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

Faithful mode uses verbatim-faithful variants whose descriptions and parameter shapes match
the original dspy source (``*_faithful`` functions + ``faithful_tool_schemas``), and replays
the paper's pre-extracted per-case histology predictions (``histology_replay``) instead of the
unavailable in-house classifier. The retrieval embedding/rerank cache (``cached_embed`` /
``RagTool.retrieve_reranked``) is a result-preserving speedup that also makes retrieval
reproducible run-to-run.

Cohere reranking (``rerank_passages``) restores the paper's rerank step when COHERE_API_KEY
is set; otherwise retrieval is cosine-only.
"""
from __future__ import annotations

import base64
import functools
import hashlib
import json
import mimetypes
import os
import threading
import urllib.parse
from pathlib import Path

import requests

from . import faithful_prompts as _fp

_TIMEOUT = 20

# --- retrieval embedding cache (result-preserving speedup) -------------------
# text-embedding-3-large is deterministic for a given input, so caching an
# embedding can never change a retrieval result — it only avoids a repeated
# OpenAI round-trip. Two layers: a process-global in-memory dict (shared across
# the per-case worker threads, which each build their own RagTool) and a disk
# cache under EMBED_CACHE_DIR (shared across processes/runs).
_EMBED_MEM: dict[str, list] = {}
_RETR_MEM: dict[str, list] = {}
_EMBED_MEM_LOCK = threading.Lock()
_RETR_MEM_LOCK = threading.Lock()
_CACHE_STATS = {"mem_hit": 0, "disk_hit": 0, "miss": 0,
                "retr_mem_hit": 0, "retr_disk_hit": 0, "retr_miss": 0}
_STATS_LOCK = threading.Lock()


def _stat(key: str) -> None:
    with _STATS_LOCK:
        _CACHE_STATS[key] += 1


def cache_stats() -> dict:
    """Snapshot of embedding + retrieval cache hit/miss counters for this process."""
    with _STATS_LOCK:
        s = dict(_CACHE_STATS)
    et = s["mem_hit"] + s["disk_hit"] + s["miss"]
    rt = s["retr_mem_hit"] + s["retr_disk_hit"] + s["retr_miss"]
    s["total"] = et
    s["hit_rate"] = (s["mem_hit"] + s["disk_hit"]) / et if et else 0.0
    s["retr_total"] = rt
    s["retr_hit_rate"] = (s["retr_mem_hit"] + s["retr_disk_hit"]) / rt if rt else 0.0
    return s


def _embed_disk_path(cache_dir: str, embed_model: str, text: str) -> str:
    h = hashlib.sha256(f"{embed_model}\x00{text}".encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, h[:2], h + ".json")


def _atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{hashlib.md5(os.urandom(8)).hexdigest()[:8]}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def cached_embed(ef, embed_model: str, text: str, cache_dir: str | None) -> list:
    """Embed one query, reusing the in-memory then disk cache. Falls back to a
    live ``ef([text])`` call on a miss. Pure cache: the returned vector is the
    same one ``ef`` would have produced (deterministic embedding model)."""
    mem_key = f"{embed_model}\x00{text}"
    with _EMBED_MEM_LOCK:
        v = _EMBED_MEM.get(mem_key)
    if v is not None:
        _stat("mem_hit")
        return v
    disk_path = _embed_disk_path(cache_dir, embed_model, text) if cache_dir else None
    if disk_path and os.path.exists(disk_path):
        try:
            with open(disk_path) as f:
                v = json.load(f)
            with _EMBED_MEM_LOCK:
                _EMBED_MEM[mem_key] = v
            _stat("disk_hit")
            return v
        except Exception:  # noqa: BLE001 — a corrupt/partial cache file just re-embeds
            pass
    # miss: embed live via the collection's own embedding function (identical to
    # what coll.query(query_texts=...) computes internally).
    raw = ef([text])[0]
    v = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    with _EMBED_MEM_LOCK:
        _EMBED_MEM[mem_key] = v
    if disk_path:
        try:
            _atomic_write_json(disk_path, v)
        except Exception:  # noqa: BLE001 — caching is best-effort
            pass
    _stat("miss")
    return v


# --- RAG retrieval -----------------------------------------------------------
class RagTool:
    def __init__(self, chroma_dir: str, collection: str, embed_model: str, k: int = 20,
                 embed_cache_dir: str | None = None):
        self.chroma_dir = chroma_dir
        self.collection_name = collection
        self.embed_model = embed_model
        self.k = k
        self._coll = None
        self._ef = None
        self._lock = threading.Lock()
        # disk embedding cache (cross-process / cross-run). EMBED_CACHE_DIR env
        # var overrides; None disables disk caching (in-memory only).
        self.embed_cache_dir = embed_cache_dir or os.environ.get("EMBED_CACHE_DIR") or None

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
                    self._ef = ef
        return self._coll

    def query(self, query: str, k: int | None = None) -> list[dict]:
        """Retrieve top-k passages for ``query``.

        Result-preserving speedup: the embedding (the OpenAI round-trip, the slow
        part) is computed with the collection's OWN embedding function *outside* the
        lock and cached, so concurrent subquery retrievals no longer serialize on the
        embed step. The cheap in-memory Chroma vector search stays under the lock.
        Using ``query_embeddings`` with the collection's EF returns the identical hit
        set that ``query_texts`` would (verified by the equivalence test), since
        ``query_texts`` just runs that same EF internally."""
        k = k or self.k
        coll = self._collection()
        emb = cached_embed(self._ef, self.embed_model, query, self.embed_cache_dir)
        with self._lock:
            res = coll.query(query_embeddings=[emb], n_results=k)
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

    def query_texts(self, query: str, k: int | None = None) -> list[dict]:
        """Original ``query_texts`` retrieval path, kept as the reference for the
        equivalence check (to confirm the embed-path change is result-preserving)."""
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

    def retrieve_reranked(self, query: str, k: int, top_n: int, rerank: bool) -> list[dict]:
        """Full per-subquery retrieval: chroma top-k -> Cohere rerank top-n (or cosine
        truncation). Cached by (collection, embed_model, query, k, top_n, rerank).

        Why cache the RERANK output and not just the embedding: chroma retrieval is
        deterministic, but Cohere rerank scores on near-tied guideline chunks vary
        run-to-run, so the reranked top-n flips between calls. Calling Cohere live
        every run makes retrieval non-reproducible. Caching the reranked result fixes
        ONE draw and reuses it, making the agent reproducible (and the parallel fan-out
        provably exact), without biasing the result — the same sense in which the
        embedding cache is pure."""
        mem_key = f"{self.collection_name}\x00{self.embed_model}\x00{query}\x00{k}\x00{top_n}\x00{int(rerank)}"
        with _RETR_MEM_LOCK:
            cached = _RETR_MEM.get(mem_key)
        if cached is not None:
            _stat("retr_mem_hit")
            return [dict(h) for h in cached]
        disk_path = None
        if self.embed_cache_dir:
            h = hashlib.sha256(("retr\x00" + mem_key).encode("utf-8")).hexdigest()
            disk_path = os.path.join(self.embed_cache_dir, "retr", h[:2], h + ".json")
            if os.path.exists(disk_path):
                try:
                    with open(disk_path) as f:
                        cached = json.load(f)
                    with _RETR_MEM_LOCK:
                        _RETR_MEM[mem_key] = cached
                    _stat("retr_disk_hit")
                    return [dict(h) for h in cached]
                except Exception:  # noqa: BLE001
                    pass
        # miss: compute live (chroma deterministic; rerank one fixed draw)
        hits = self.query(query, k=k)
        if rerank:
            hits = rerank_passages(query, hits, top_n=top_n)
        else:
            hits = hits[:top_n]
        with _RETR_MEM_LOCK:
            _RETR_MEM[mem_key] = hits
        if disk_path:
            try:
                _atomic_write_json(disk_path, hits)
            except Exception:  # noqa: BLE001
                pass
        _stat("retr_miss")
        return [dict(h) for h in hits]


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


# --- web search (replaces the discontinued google_search) --------------------
# The paper's agent had a google_search tool (a nested llama-hub GoogleSearchToolSpec agent
# keyed on GOOGLE_API_KEY / GOOGLE_SEARCH_ENGINE). The Google Custom Search whole-web endpoint
# was discontinued, so the capability is restored via OpenAI's native web search. This function
# tool makes a nested Responses-API ``web_search`` call and returns the synthesized answer plus
# the cited source URLs — the modern equivalent of the original's nested google_search agent.
# It is used as a callable function tool on the chat.completions backend; the Responses-based
# backends instead expose the hosted ``web_search`` tool directly on the model loop.
def web_search(query: str, model: str = "gpt-5.1") -> tuple[str, int]:
    """Run a nested OpenAI web search and return ``(cited_summary, n_url_citations)``.

    The summary is a compact, source-cited string the agent can fold into its evidence; the
    integer is the number of distinct URL citations (0 means the search returned nothing
    usable). Best-effort: any failure degrades to an explicit message and never raises, so a
    transient search outage cannot sink a case.
    """
    if not (query or "").strip():
        return "web_search: empty query", 0
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=4, timeout=120.0)
        instr = (
            "You are a clinical web-search assistant for a molecular tumor board. Search the "
            "web for the query and return a concise, factual summary of the most relevant, "
            "authoritative findings (FDA labels, NCCN/ESMO/ASCO guidance, pivotal trials). "
            "Cite each claim with its source."
        )
        resp = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            input=[{"role": "system", "content": instr},
                   {"role": "user", "content": query}],
        )
        text = (resp.output_text or "").strip()
        urls = _collect_url_citations(resp)
        if urls:
            text += "\n\nSources:\n" + "\n".join(f"- {u}" for u in urls)
        return (text or "web_search: no results"), len(urls)
    except Exception as e:  # noqa: BLE001 — web search is best-effort; degrade to a message
        return f"web_search failed: {e}", 0


def _collect_url_citations(resp) -> list[str]:
    """Extract the distinct ``url_citation`` URLs from a Responses message output.

    Used both by :func:`web_search` (chat backend) and by the Responses backends to tell
    whether a hosted web-search result was actually cited in the model's answer."""
    urls: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            for ann in getattr(part, "annotations", []) or []:
                if getattr(ann, "type", None) == "url_citation":
                    u = getattr(ann, "url", None)
                    if u and u not in urls:
                        urls.append(u)
    return urls[:10]


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


# =============================================================================
# Faithful mode: the paper's described pipeline.
#
# The internal tool NAMES are kept modern (oncokb/pubmed/calculate/radiology_report/medsam/
# histology_classifier) so a downstream tool-use mapping keyed on those names is unchanged.
# What is restored to be VERBATIM-faithful is the tool *descriptions* (copied character-for-
# character from the original agent_tools.py docstrings, via faithful_prompts) and the
# *parameter shapes* (the original signatures: oncokb's change=mutation/amplification/variant,
# pubmed's search-terms list, calculate's a/b/operator, radiology's folder-of-images, segment's
# nested bbox list). The histology tool replays the paper's pre-extracted per-case MSI/KRAS/BRAF
# predictions (see histology_replay).
# =============================================================================


# --- faithful OncoKB: restore the change=mutation/amplification/variant branch ----
def oncokb_annotate_faithful(hugo_symbol: str, change: str, alteration: str) -> str:
    """Port of the original onco_kb(hugo_symbol, change, alteration): three OncoKB endpoints
    selected by ``change`` (mutation / amplification / variant). Uses the prod endpoint when an
    ONCOKB_API_TOKEN is set, else the public demo endpoint (a recorded deviation)."""
    token = os.environ.get("ONCOKB_API_TOKEN")
    base = "https://www.oncokb.org/api/v1" if token else "https://demo.oncokb.org/api/v1"
    headers = {"accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    change = (change or "mutation").strip().lower()
    try:
        if change == "amplification":
            url = f"{base}/annotate/copyNumberAlterations"
            params = {"hugoSymbol": hugo_symbol, "copyNameAlterationType": alteration.upper()}
        elif change == "variant":
            a, _, b = hugo_symbol.partition("-")
            url = f"{base}/annotate/structuralVariants"
            params = {"hugoSymbolA": a, "hugoSymbolB": b,
                      "structuralVariantType": alteration.upper(), "isFunctionalFusion": "true"}
        else:  # mutation (default)
            url = f"{base}/annotate/mutations/byProteinChange"
            params = {"hugoSymbol": hugo_symbol, "alteration": alteration}
        r = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        r.raise_for_status()
        d = r.json()
    except Exception as e:  # noqa: BLE001
        return f"OncoKB lookup failed ({change}, {'prod' if token else 'demo'}): {e}"
    onc = d.get("oncogenic", "Unknown")
    eff = (d.get("mutationEffect") or {}).get("knownEffect", "")
    txs = []
    for t in d.get("treatments", [])[:6]:
        drugs = ", ".join(dr.get("drugName", "") for dr in t.get("drugs", []))
        lvl = t.get("level", "")
        inds = "; ".join(t.get("indications", []) or t.get("approvedIndications", []) or [])
        txs.append(f"{drugs} (level {lvl}) {inds}".strip())
    return json.dumps({"oncogenic": onc, "mutationEffect": eff, "change": change,
                       "treatments": txs or ["none reported at this endpoint"],
                       "endpoint": "prod" if token else "demo"})


# --- faithful PubMed: restore the (search-terms list, query) signature ----
def pubmed_search_faithful(pubmed_search_terms, query: str = "") -> str:
    """Port of the original query_pubmed(pubmed_search_terms, query): only the first three
    search terms are used to fetch articles (per the original docstring), then the final query
    is appended for relevance."""
    terms = pubmed_search_terms if isinstance(pubmed_search_terms, list) else [pubmed_search_terms]
    terms = [str(t) for t in terms if str(t).strip()][:3]
    combined = " OR ".join(f"({t})" for t in terms) if terms else (query or "")
    if query and terms:
        combined = f"({combined}) AND ({query})"
    return pubmed_search(combined or query)


# --- faithful calculate: restore the (a, b, operator) signature ----
def calculate_faithful(a: float, b: float, operator: str) -> str:
    """Port of the original calculate(a, b, operator): +, -, *, / on two numbers."""
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return "Invalid operands. Provide numeric a and b."
    if operator == "+":
        return f"The sum of {a} and {b} is {a + b}."
    if operator == "-":
        return f"Subtracting {a} and {b} (a-b) is {a - b}."
    if operator == "*":
        return f"Multiplying {a} and {b} is {a * b}."
    if operator == "/":
        if b == 0:
            return "Division by zero is undefined."
        return f"The ratio between {a} and {b} is {a / b}."
    return "Invalid operator. Please use one of the following: +, -, *, /."


# --- faithful radiology: restore the folder-of-images multi-image compare ----
def radiology_report_folder(image_paths: list[str], query: str = "",
                            model: str = "gpt-5.1") -> str:
    """Port of the original gen_radiology_report(path_to_img_folder, query): read EACH image in
    the patient's folder separately, then — when more than one is present — run a comparison
    pass over all of them (the original's single-vision then multi-vision two-stage flow). Our
    folder is the case's resolved image set."""
    if not image_paths:
        return "No images found in the provided folder."
    out = ""
    for p in image_paths:
        name = Path(p).name
        rep = radiology_report(p, query, model=model)
        out += f"Radiology Report for {name}\n{rep}\n\n" + "*" * 10 + "\n\n"
    if len(image_paths) > 1:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            content = [{"type": "text", "text": (
                "You are a board-certified radiologist. Compare the following medical images "
                "of the same patient across timepoints. Describe any change in measurable "
                "lesions (size/number/location) and give an overall impression of "
                "progression, stability, or response.")}]
            for p in image_paths:
                content.append({"type": "image_url",
                                "image_url": {"url": _data_url(p)}})
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": content}],
                max_completion_tokens=1200)
            out += "Radiology Report for comparing images\n" + (resp.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            out += f"Radiology comparison failed: {e}"
    return out


# --- faithful histology REPLAY: return the paper's pre-extracted predictions ----
# Packaged default lookup (the ferber20 cases, extracted from the public paper supplementary).
# Override with the HISTOLOGY_LOOKUP env var to point at your own lookup JSON.
_HISTOLOGY_LOOKUP_DEFAULT = Path(__file__).resolve().parent / "data" / "histology_lookup.json"


@functools.lru_cache(maxsize=1)
def _histology_lookup() -> dict:
    """Load the per-case MSI/KRAS/BRAF prediction lookup, keyed by case surname.

    Resolution order: the ``HISTOLOGY_LOOKUP`` env var if it points at an existing file,
    otherwise the lookup bundled with the package (``ferber_agent/data/histology_lookup.json``,
    built from the paper supplementary by ``scripts/build_histology_lookup.py``)."""
    env_path = os.environ.get("HISTOLOGY_LOOKUP")
    path = Path(env_path) if env_path else _HISTOLOGY_LOOKUP_DEFAULT
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def histology_replay(case_key: str, targets=None) -> str:
    """Replay the paper's check_mutations predictions for ``case_key`` (the case surname).

    The original ran proprietary H&E classifiers (never released). The paper states it
    pre-extracted those predictions for convenience; we replay the documented per-case
    MSI/KRAS/BRAF (label, probability) from the supplementary. Returns an explicit gap message
    when a case has no documented prediction."""
    lut = _histology_lookup()
    rec = (lut.get("cases") or {}).get(case_key) if "cases" in lut else lut.get(case_key)
    if not rec or not rec.get("available", True) or not rec.get("predictions"):
        return (f"No histology-based genetic prediction is documented for this case "
                f"({case_key}). Use the molecular/genomic report for mutation and MSI status.")
    preds = rec["predictions"]
    want = None
    if targets:
        tl = targets if isinstance(targets, list) else [str(targets)]
        want = {str(t).strip().upper() for chunk in tl for t in str(chunk).split(",")}
    out = "Genetic predictions from histopathology images:\n"
    out += "*" * (len(out) - 1) + "\n"
    for marker in ("MSI", "KRAS", "BRAF"):
        if marker not in preds:
            continue
        if want and marker not in want:
            continue
        p = preds[marker]
        out += f"Target is {marker}:\n"
        out += f"prediction: {p.get('label', 'n/a')}\n"
        if p.get("probability") is not None:
            out += f"probability: {p['probability']}\n"
        out += "\n"
    return out.strip() or (f"No requested targets documented for {case_key}.")


# --- faithful OpenAI tool schemas (verbatim descriptions, restored params) ----
def faithful_tool_schemas(enabled: tuple[str, ...]) -> list[dict]:
    """Schemas whose ``description`` is the VERBATIM original docstring and whose parameters
    match the original signatures. Internal names stay modern for Table-1 mapping."""
    schemas = {
        "oncokb": {"type": "function", "function": {
            "name": "oncokb", "description": _fp.TOOL_ONCOKB_DOC,
            "parameters": {"type": "object", "properties": {
                "hugo_symbol": {"type": "string"},
                "change": {"type": "string", "enum": ["mutation", "amplification", "variant"]},
                "alteration": {"type": "string"}},
                "required": ["hugo_symbol", "change", "alteration"]}}},
        "pubmed": {"type": "function", "function": {
            "name": "pubmed", "description": _fp.TOOL_PUBMED_DOC,
            "parameters": {"type": "object", "properties": {
                "pubmed_search_terms": {"type": "array", "items": {"type": "string"}},
                "query": {"type": "string"}},
                "required": ["pubmed_search_terms", "query"]}}},
        "calculate": {"type": "function", "function": {
            "name": "calculate", "description": _fp.TOOL_CALCULATE_DOC,
            "parameters": {"type": "object", "properties": {
                "a": {"type": "number"}, "b": {"type": "number"},
                "operator": {"type": "string", "enum": ["+", "-", "*", "/"]}},
                "required": ["a", "b", "operator"]}}},
        "radiology_report": {"type": "function", "function": {
            "name": "radiology_report", "description": _fp.TOOL_RADIOLOGY_DOC,
            "parameters": {"type": "object", "properties": {
                "path_to_img_folder": {"type": "string"},
                "query": {"type": "string"}},
                "required": ["path_to_img_folder", "query"]}}},
        "medsam": {"type": "function", "function": {
            "name": "medsam", "description": _fp.TOOL_SEGMENT_DOC,
            "parameters": {"type": "object", "properties": {
                "path_to_img": {"type": "string"},
                "bbox_coordinates": {"type": "array",
                    "items": {"type": "array", "items": {"type": "number"}}}},
                "required": ["path_to_img", "bbox_coordinates"]}}},
        "histology_classifier": {"type": "function", "function": {
            "name": "histology_classifier", "description": _fp.TOOL_CHECKMUTATIONS_DOC,
            "parameters": {"type": "object", "properties": {
                "patient_id": {"type": "string"},
                "targets": {"type": "array", "items": {"type": "string"}}},
                "required": ["patient_id"]}}},
        # web_search replaces the paper's discontinued google_search. It is additive and not
        # part of the verbatim-faithful byte-diff set, so its description is a modern
        # web-search equivalent of the original google_search tool rather than a vendored block.
        "web_search": {"type": "function", "function": {
            "name": "web_search", "description": (
                "Search the web for current authoritative oncology information (FDA approvals, "
                "NCCN/ESMO/ASCO guidance, pivotal trials, drug labels) when the knowledge base "
                "and other tools are insufficient. Returns a source-cited summary."),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "the web search query"}},
                "required": ["query"]}}},
    }
    return [schemas[t] for t in enabled if t in schemas]


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
        "web_search": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current authoritative oncology information "
                               "(FDA approvals, NCCN/ESMO/ASCO guidance, pivotal trials, drug "
                               "labels). Returns a source-cited summary.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "the web search query"}},
                    "required": ["query"]},
            }},
    }
    return [schemas[t] for t in enabled if t in schemas]
