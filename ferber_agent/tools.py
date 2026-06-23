"""Tools for the Ferber agent (modernized).

These mirror the tools the Ferber et al. RAG agent uses, on a modern stack. The genomic
text tools (rag/oncokb/pubmed/calculate) are pure API/CPU; the imaging tools are wired for
the multimodal ferber20 track:

  - rag                 retrieve oncology guideline passages from the Chroma knowledge base
  - oncokb              OncoKB genomic annotation (public demo endpoint, or prod with a token)
  - pubmed              PubMed literature lookup via NCBI E-utilities
  - calculate           safe arithmetic (e.g. progression ratios from segmented areas)
  - radiology_report    GPT-4V-style structured radiology report from a patient image
  - medsam              MedSAM bbox-prompted segmentation -> lesion area (in px)
  - histology_classifier  in-house KRAS/BRAF/MSI histology classifier — UNAVAILABLE
                          (unreleased upstream; returns an explicit gap message)

Cohere reranking (``rerank_passages``) restores the paper's rerank step when COHERE_API_KEY
is set; otherwise retrieval is cosine-only (a recorded deviation).
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import urllib.parse

import requests

_TIMEOUT = 20


# --- RAG retrieval -----------------------------------------------------------
class RagTool:
    def __init__(self, chroma_dir: str, collection: str, embed_model: str, k: int = 20):
        self.chroma_dir = chroma_dir
        self.collection_name = collection
        self.embed_model = embed_model
        self.k = k
        self._coll = None
        self._lock = threading.Lock()

    def _collection(self):
        if self._coll is None:
            with self._lock:
                if self._coll is None:
                    import chromadb
                    from chromadb.utils import embedding_functions

                    ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=os.environ["OPENAI_API_KEY"], model_name=self.embed_model)
                    client = chromadb.PersistentClient(path=self.chroma_dir)
                    self._coll = client.get_collection(
                        name=self.collection_name, embedding_function=ef)
        return self._coll

    def query(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.k
        coll = self._collection()
        with self._lock:
            res = coll.query(query_texts=[query], n_results=k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None] * len(docs)])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({
                "text": doc,
                "title": (meta or {}).get("title", ""),
                "source": (meta or {}).get("source", ""),
                "distance": float(dist) if dist is not None else None,
            })
        return out


# --- Cohere reranking --------------------------------------------------------
def rerank_passages(query: str, hits: list[dict], top_n: int = 10,
                    model: str = "rerank-english-v3.0") -> list[dict]:
    """Re-order retrieved passages by Cohere relevance, restoring the paper's rerank step.

    Falls back to the input order (cosine-only) when COHERE_API_KEY is absent or the call
    fails — a recorded deviation, never a hard error. Returns the (possibly reordered and
    truncated) hits, each annotated with a ``rerank_score`` when reranking was applied.
    """
    if not hits:
        return hits
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        return hits[:top_n]
    try:
        import cohere

        client = cohere.ClientV2(api_key=api_key)
        docs = [h.get("text", "") for h in hits]
        res = client.rerank(model=model, query=query, documents=docs,
                            top_n=min(top_n, len(docs)))
        ordered = []
        for r in res.results:
            h = dict(hits[r.index])
            h["rerank_score"] = float(r.relevance_score)
            ordered.append(h)
        return ordered
    except Exception:  # noqa: BLE001 — reranking is best-effort; degrade to cosine order
        return hits[:top_n]


# --- OncoKB ------------------------------------------------------------------
def oncokb_annotate(hugo_symbol: str, alteration: str) -> str:
    token = os.environ.get("ONCOKB_API_TOKEN")
    base = "https://www.oncokb.org/api/v1" if token else "https://demo.oncokb.org/api/v1"
    url = (f"{base}/annotate/mutations/byProteinChange"
           f"?hugoSymbol={urllib.parse.quote(hugo_symbol)}"
           f"&alteration={urllib.parse.quote(alteration)}")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(url, headers=headers, timeout=_TIMEOUT)
        r.raise_for_status()
        d = r.json()
    except Exception as e:  # noqa: BLE001
        return f"OncoKB lookup failed ({base.split('//')[1].split('.')[0]}): {e}"
    onc = d.get("oncogenic", "Unknown")
    eff = (d.get("mutationEffect") or {}).get("knownEffect", "")
    txs = []
    for t in d.get("treatments", [])[:6]:
        drugs = ", ".join(dr.get("drugName", "") for dr in t.get("drugs", []))
        lvl = t.get("level", "")
        inds = "; ".join(t.get("indications", []) or t.get("approvedIndications", []) or [])
        txs.append(f"{drugs} (level {lvl}) {inds}".strip())
    summary = {"oncogenic": onc, "mutationEffect": eff,
               "treatments": txs or ["none reported at this endpoint"],
               "endpoint": "prod" if token else "demo"}
    return json.dumps(summary)


# --- PubMed ------------------------------------------------------------------
def pubmed_search(query: str, k: int = 4) -> str:
    eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        es = requests.get(f"{eutils}/esearch.fcgi", params={
            "db": "pubmed", "term": query, "retmax": k, "retmode": "json",
            "sort": "relevance"}, timeout=_TIMEOUT)
        es.raise_for_status()
        ids = es.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No PubMed results."
        su = requests.get(f"{eutils}/esummary.fcgi", params={
            "db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=_TIMEOUT)
        su.raise_for_status()
        res = su.json().get("result", {})
        lines = []
        for pid in ids:
            it = res.get(pid, {})
            title = it.get("title", "")
            src = it.get("fulljournalname", it.get("source", ""))
            year = (it.get("pubdate", "") or "")[:4]
            lines.append(f"PMID {pid} ({year}, {src}): {title}")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"PubMed lookup failed: {e}"


# --- calculate ---------------------------------------------------------------
def calculate(expression: str) -> str:
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
           ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod}

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:  # noqa: BLE001
        return f"calc error: {e}"


# --- imaging: shared image resolution ----------------------------------------
def resolve_image(ref: str, image_map: dict[str, str] | None) -> str | None:
    """Resolve a model-supplied image name to an absolute path against ``image_map``.

    The vignettes refer to images by date-based names (e.g. ``September2023.png``) while the
    on-disk files are ``Surname_N.jpg``; ``image_map`` carries both aliases -> path. Matching
    is lenient: exact, basename, and stem (extension-insensitive), case-insensitive. Cross-
    patient duplicates (e.g. ferber20 Garcia_2) are simply absent from the map, so they never
    resolve. Returns None when nothing matches.
    """
    if not ref or not image_map:
        return None
    cand = ref.strip()
    # exact / case-insensitive
    for k, v in image_map.items():
        if k == cand or k.lower() == cand.lower():
            return v
    # by stem (drop extension on both sides)
    def _stem(s: str) -> str:
        s = s.replace("\\", "/").split("/")[-1]
        return s.rsplit(".", 1)[0].lower()
    cstem = _stem(cand)
    for k, v in image_map.items():
        if _stem(k) == cstem:
            return v
    return None


# --- imaging: radiology report (GPT-4V-style) --------------------------------
def _data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def radiology_report(image_path: str, clinical_question: str = "",
                     model: str = "gpt-5.1") -> str:
    """Produce a structured radiology report from a patient image (vision call).

    Mirrors the Ferber agent's GPT-4V radiology tool: the multimodal backbone reads the
    image and returns a structured report (technique, findings by organ system, measurable
    lesions with approximate sizes, impression). It does NOT see the patient vignette — it
    reports what is in the pixels — so the agent must integrate it with the clinical context.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        "You are a board-certified radiologist. Read this medical image and write a "
        "structured report with sections: Modality/Technique, Findings (by organ system or "
        "region, noting any measurable lesion and its approximate size and location), and "
        "Impression. Be specific and concise; report only what is visible in the image."
    )
    if clinical_question:
        prompt += f"\n\nClinical question: {clinical_question}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            ]}],
            max_completion_tokens=1200,
        )
        return resp.choices[0].message.content or "(empty radiology report)"
    except Exception as e:  # noqa: BLE001
        return f"radiology_report failed: {e}"


# --- imaging: MedSAM bbox segmentation -> area -------------------------------
# Uses the official Wang Lab HF mirror (transformers SamModel format) by default, so the
# weights pull cleanly via huggingface_hub (no Google-Drive .pth step). MEDSAM_MODEL_ID
# overrides the repo; a local snapshot path also works.
_MEDSAM_LOCK = threading.Lock()
_MEDSAM = None
_MEDSAM_DEFAULT_ID = "wanglab/medsam-vit-base"


def _load_medsam():
    """Lazy-load MedSAM (transformers SamModel + SamProcessor). Heavy; first-use only."""
    global _MEDSAM
    if _MEDSAM is not None:
        return _MEDSAM
    with _MEDSAM_LOCK:
        if _MEDSAM is None:
            import torch
            from transformers import SamModel, SamProcessor

            model_id = os.environ.get("MEDSAM_MODEL_ID", _MEDSAM_DEFAULT_ID)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SamModel.from_pretrained(model_id).to(device).eval()
            processor = SamProcessor.from_pretrained(model_id)
            _MEDSAM = (model, processor, device)
    return _MEDSAM


def medsam_segment(image_path: str, bbox: list[int]) -> str:
    """Segment the lesion inside ``bbox`` with MedSAM and return its area.

    ``bbox`` is ``[x_min, y_min, x_max, y_max]`` in the pixel coordinates of the provided
    image (the ferber20 vignettes quote these directly, e.g. ``[475, 250, 490, 275]``). The
    box prompts MedSAM, the mask is decoded at the image's native resolution, and the area is
    reported in native pixels plus as a fraction of the image — so two timepoints can be
    compared with ``calculate`` to get a progression ratio.
    """
    try:
        import numpy as np
        import torch
        from PIL import Image

        model, processor, device = _load_medsam()

        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        x0, y0, x1, y1 = (float(v) for v in bbox)
        x0, x1 = sorted((max(0.0, x0), min(float(W), x1)))
        y0, y1 = sorted((max(0.0, y0), min(float(H), y1)))

        inputs = processor(img, input_boxes=[[[x0, y0, x1, y1]]], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())
        mask = masks[0][0][0].numpy().astype(np.uint8)  # (H, W) at native resolution

        area_px = int(mask.sum())
        frac = area_px / float(H * W)
        return json.dumps({
            "area_px": area_px, "image_hw": [H, W], "area_fraction": round(frac, 6),
            "bbox_used": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
            "note": "lesion area from MedSAM mask; compare two timepoints with calculate",
        })
    except Exception as e:  # noqa: BLE001
        return f"medsam_segment failed: {e}"


# --- imaging: in-house histology classifier (HARD GAP) -----------------------
def histology_classifier_unavailable(marker: str = "", image_ref: str = "") -> str:
    """The Ferber in-house KRAS/BRAF/MSI histology classifier is NOT reproducible.

    Ferber et al. trained proprietary H&E-based classifiers (KRAS, BRAF, MSI) that were never
    released. Rather than fabricate predictions, this tool returns an explicit unavailability
    message and directs the agent to the molecular report. Documented as a hard faithfulness
    gap in the README.
    """
    m = f" for {marker}" if marker else ""
    return (f"Histology image classifier{m} UNAVAILABLE: the Ferber et al. in-house "
            "KRAS/BRAF/MSI H&E classifiers were never publicly released and cannot be "
            "reproduced. Use the molecular/genomic report for mutation and MSI status, and "
            "OncoKB for the therapeutic implications.")


# --- OpenAI tool schemas -----------------------------------------------------
def tool_schemas(enabled: tuple[str, ...]) -> list[dict]:
    schemas = {
        "rag": {
            "type": "function",
            "function": {
                "name": "rag",
                "description": "Retrieve oncology clinical-practice-guideline passages from "
                               "the knowledge base. Use for standard-of-care, therapy "
                               "selection, and biomarker guidance.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "guideline search query"}},
                    "required": ["query"]},
            }},
        "oncokb": {
            "type": "function",
            "function": {
                "name": "oncokb",
                "description": "Look up the oncogenicity and FDA/guideline therapy "
                               "implications of a specific gene alteration in OncoKB.",
                "parameters": {"type": "object", "properties": {
                    "hugo_symbol": {"type": "string", "description": "gene, e.g. BRAF"},
                    "alteration": {"type": "string", "description": "protein change, e.g. V600E"}},
                    "required": ["hugo_symbol", "alteration"]},
            }},
        "pubmed": {
            "type": "function",
            "function": {
                "name": "pubmed",
                "description": "Search PubMed for relevant primary literature / trials.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"}}, "required": ["query"]},
            }},
        "calculate": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a basic arithmetic expression, e.g. a progression "
                               "ratio between two segmented lesion areas.",
                "parameters": {"type": "object", "properties": {
                    "expression": {"type": "string"}}, "required": ["expression"]},
            }},
        "radiology_report": {
            "type": "function",
            "function": {
                "name": "radiology_report",
                "description": "Generate a structured radiology report from one of the "
                               "patient's imaging files by reading the image itself. Pass the "
                               "image filename as referenced in the case (e.g. "
                               "'September2023.png' or the listed file name).",
                "parameters": {"type": "object", "properties": {
                    "image_ref": {"type": "string",
                                  "description": "image filename referenced in the case"},
                    "clinical_question": {"type": "string",
                                          "description": "optional focus for the report"}},
                    "required": ["image_ref"]},
            }},
        "medsam": {
            "type": "function",
            "function": {
                "name": "medsam",
                "description": "Segment a lesion in a patient image given a bounding box and "
                               "return its area in pixels (and as a fraction of the image). "
                               "Use the bbox coordinates quoted in the radiology report "
                               "([x_min, y_min, x_max, y_max]); compare two timepoints with "
                               "'calculate' to quantify progression.",
                "parameters": {"type": "object", "properties": {
                    "image_ref": {"type": "string",
                                  "description": "image filename referenced in the case"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4,
                             "maxItems": 4,
                             "description": "[x_min, y_min, x_max, y_max] in image pixels"}},
                    "required": ["image_ref", "bbox"]},
            }},
        "histology_classifier": {
            "type": "function",
            "function": {
                "name": "histology_classifier",
                "description": "Attempt to predict KRAS/BRAF/MSI status from an H&E histology "
                               "image. NOTE: the in-house classifier is unavailable; this "
                               "returns a gap message directing you to the molecular report.",
                "parameters": {"type": "object", "properties": {
                    "image_ref": {"type": "string"},
                    "marker": {"type": "string",
                               "description": "one of KRAS, BRAF, MSI"}},
                    "required": ["marker"]},
            }},
    }
    return [schemas[t] for t in enabled if t in schemas]
