"""Tools for the Ferber agent (modernized).

These mirror the tools the Ferber et al. RAG agent uses on the genomic track, on a modern
lightweight stack (no torch). The original fork's imaging/pathology vision tools and the
Google-search tool are out of scope for the MSK genomic text track (stubbed/omitted).

  - rag       retrieve oncology guideline passages from the Chroma knowledge base
  - oncokb    OncoKB genomic annotation (public demo endpoint, or prod with a token)
  - pubmed    PubMed literature lookup via NCBI E-utilities
  - calculate safe arithmetic
"""
from __future__ import annotations

import json
import os
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

    def _collection(self):
        if self._coll is None:
            import chromadb
            from chromadb.utils import embedding_functions

            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"], model_name=self.embed_model)
            client = chromadb.PersistentClient(path=self.chroma_dir)
            self._coll = client.get_collection(name=self.collection_name, embedding_function=ef)
        return self._coll

    def query(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.k
        res = self._collection().query(query_texts=[query], n_results=k)
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
                "description": "Evaluate a basic arithmetic expression.",
                "parameters": {"type": "object", "properties": {
                    "expression": {"type": "string"}}, "required": ["expression"]},
            }},
    }
    return [schemas[t] for t in enabled if t in schemas]
