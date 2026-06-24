"""Pluggable guideline-retrieval engines for the faithful pipeline's Stage 2.

The faithful Ferber pipeline expands the question into subqueries and retrieves guideline
passages for each. This module isolates *how* that retrieval is done, holding every other
stage (verbatim prompts, Stage-1 tool gathering, Stage-2 synthesis, model, cases) fixed. Each
engine implements the same ``retrieve(query, retrieve_k, top_n, usage=None)`` contract and
returns a list of normalized passage dicts:

    {"text", "title", "source", "score", "engine", "chunk_chars"}

  ``chroma_cosine``              Chroma top-``retrieve_k`` cosine -> top-``top_n`` (no rerank).
  ``chroma_cohere``             Chroma top-``retrieve_k`` cosine -> Cohere rerank -> top-``top_n``.
  ``openai_filesearch_responses`` OpenAI Responses ``file_search`` tool over a vector store of
                                the source docs; OpenAI does its own chunking/embedding/retrieval.
  ``openai_filesearch_chat``    The SAME vector store, retrieved through a chat.completions
                                function-tool wrapper: the model emits a ``file_search`` call,
                                the handler runs ``vector_stores.search`` and returns passages.

The two ``chroma_*`` engines delegate to the agent's :meth:`RagTool.retrieve_reranked`, so they
reuse its embedding/rerank cache (a result-preserving speedup that also makes retrieval
reproducible run-to-run) and produce output identical to the inline Chroma path. The
``openai_filesearch_*`` engines require a vector store id (build one with
``scripts/build_vector_store.py``).

Every engine optionally takes a :class:`~ferber_agent.usage.UsageAccumulator` to record
per-engine retrieval cost and latency (rerank calls, file_search calls, vector-store searches,
model tokens, seconds).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

# Canonical engine names (the public retrieval-engine switch).
ENGINES = ("chroma_cosine", "chroma_cohere", "openai_filesearch_responses",
           "openai_filesearch_chat")
CHROMA_ENGINES = ("chroma_cosine", "chroma_cohere")
OPENAI_FS_ENGINES = ("openai_filesearch_responses", "openai_filesearch_chat")
# Deprecated short aliases accepted by make_engine for backward compatibility.
_ALIASES = {"openai_fs_responses": "openai_filesearch_responses",
            "openai_fs_chat": "openai_filesearch_chat"}


@lru_cache(maxsize=1)
def _oai_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=6, timeout=180.0)


# --- shared passage normalization ------------------------------------------------------
def _passage(text: str, title: str, source: str, score, engine: str) -> dict:
    text = text or ""
    return {
        "text": text,
        "title": title or "",
        "source": source or "",
        "score": (float(score) if score is not None else None),
        "engine": engine,
        "chunk_chars": len(text),  # for the OpenAI-chunking-vs-fixed-chunk diagnostic
    }


def _source_from(filename: str, attributes) -> tuple[str, str]:
    """Map an OpenAI vector-store filename / attributes back to ``(source, title)``.

    Files are uploaded as ``{source}__{doc_id}.txt`` with attributes {source, title, doc_id},
    so attributes are authoritative; the filename prefix is the fallback."""
    src = ""
    title = ""
    if isinstance(attributes, dict):
        src = attributes.get("source", "") or ""
        title = attributes.get("title", "") or ""
    if not src and filename:
        base = filename.rsplit("/", 1)[-1]
        src = base.split("__", 1)[0]
    return src, title


def _fs_result_text(r) -> str:
    """Extract text from a file_search / vector-store result (handles both result shapes)."""
    t = getattr(r, "text", None)
    if isinstance(t, str) and t:
        return t
    out = []
    for part in getattr(r, "content", None) or []:
        if getattr(part, "type", None) == "text":
            out.append(getattr(part, "text", "") or "")
        elif isinstance(part, dict) and part.get("type") == "text":
            out.append(part.get("text", "") or "")
    return "\n".join(out)


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- engines ---------------------------------------------------------------------------
class ChromaEngine:
    """Chroma cosine retrieval, optionally followed by Cohere rerank.

    ``rerank=True`` restores the paper's rerank step (``chroma_cohere``); ``rerank=False`` is
    cosine-only (``chroma_cosine``). Retrieval is delegated to the agent's
    :meth:`RagTool.retrieve_reranked`, so the embedding/rerank cache is reused and the output is
    identical to the inline Chroma path."""

    def __init__(self, chroma_dir: str, collection: str, embed_model: str, rerank: bool,
                 rag=None):
        from .tools import RagTool

        self.name = "chroma_cohere" if rerank else "chroma_cosine"
        self.rerank = rerank and bool(os.environ.get("COHERE_API_KEY"))
        # Reuse the agent's RagTool when given (avoids a second Chroma client per agent and
        # shares its embedding/rerank cache).
        self._rag = rag if rag is not None else RagTool(chroma_dir, collection, embed_model)

    def retrieve(self, query: str, retrieve_k: int, top_n: int, usage=None) -> list[dict]:
        timer = usage.retrieval_timer() if usage is not None else _nullctx()
        with timer:
            # Cached, reproducible chroma retrieval (chroma top-k -> rerank/truncate top-n).
            hits = self._rag.retrieve_reranked(query, retrieve_k, top_n, self.rerank)
        if usage is not None:
            usage.bump("retrieval_calls")
            if self.rerank:
                usage.bump("rerank_calls")
        return [_passage(h.get("text", ""), h.get("title", ""), h.get("source", ""),
                         h.get("rerank_score", h.get("distance")), self.name)
                for h in hits]


class OpenAIFileSearchResponsesEngine:
    """OpenAI Responses ``file_search`` tool over the corpus vector store.

    Per subquery, one Responses call is forced to use file_search and the retrieved chunks are
    captured via ``include=['file_search_call.results']`` (the model's free-text answer is
    discarded — only the passages feed the identical Stage-2 synthesis). OpenAI does its own
    chunking/embedding/retrieval and may rewrite the query: the "hand it to OpenAI's hosted
    RAG" engine."""

    name = "openai_filesearch_responses"

    def __init__(self, vector_store_id: str, llm_model: str):
        self.vs_id = vector_store_id
        self.llm_model = llm_model

    def retrieve(self, query: str, retrieve_k: int, top_n: int, usage=None) -> list[dict]:
        client = _oai_client()
        timer = usage.retrieval_timer() if usage is not None else _nullctx()
        results = []
        resp = None
        with timer:
            resp = client.responses.create(
                model=self.llm_model,
                input=("Search the oncology clinical-guideline knowledge base and surface the "
                       f"passages most relevant to this query:\n{query}"),
                tools=[{"type": "file_search", "vector_store_ids": [self.vs_id],
                        "max_num_results": retrieve_k}],
                tool_choice="required",
                include=["file_search_call.results"],
                max_output_tokens=512,
            )
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) == "file_search_call":
                    results = getattr(item, "results", None) or []
                    if usage is not None:
                        usage.bump("filesearch_calls")
        if usage is not None:
            usage.bump("retrieval_calls")
            usage.add_responses(resp, stage="retrieval")
        out = []
        for r in results[:top_n]:
            src, title = _source_from(getattr(r, "filename", ""), getattr(r, "attributes", None))
            out.append(_passage(_fs_result_text(r), title, src, getattr(r, "score", None),
                                self.name))
        return out


class OpenAIFileSearchChatEngine:
    """The SAME vector store, retrieved via a chat.completions function-tool wrapper.

    A chat.completions turn is given a ``file_search`` function tool and forced to call it with
    the subquery; the handler runs ``vector_stores.search`` over the same store and returns the
    passages. Tests whether the API surface (chat function-tool vs Responses-native
    file_search) changes retrieval."""

    name = "openai_filesearch_chat"

    _TOOL = [{
        "type": "function",
        "function": {
            "name": "file_search",
            "description": ("Search the oncology clinical-guideline knowledge base for passages "
                            "relevant to a query."),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "the guideline search query"}},
                "required": ["query"]},
        }}]

    def __init__(self, vector_store_id: str, llm_model: str):
        self.vs_id = vector_store_id
        self.llm_model = llm_model

    def retrieve(self, query: str, retrieve_k: int, top_n: int, usage=None) -> list[dict]:
        client = _oai_client()
        timer = usage.retrieval_timer() if usage is not None else _nullctx()
        with timer:
            resp = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": ("You retrieve guideline passages. Call the "
                                                    "file_search tool with the user's query.")},
                    {"role": "user", "content": query}],
                tools=self._TOOL,
                tool_choice={"type": "function", "function": {"name": "file_search"}},
                max_completion_tokens=256,
            )
            if usage is not None:
                usage.add_chat(resp, stage="retrieval")
            # the function-tool query the model emitted (falls back to the raw subquery)
            fs_query = query
            tcs = resp.choices[0].message.tool_calls or []
            if tcs:
                try:
                    fs_query = json.loads(tcs[0].function.arguments or "{}").get("query") or query
                except json.JSONDecodeError:
                    pass
            search = client.vector_stores.search(
                vector_store_id=self.vs_id, query=fs_query, max_num_results=retrieve_k)
            if usage is not None:
                usage.bump("vstore_searches")
        if usage is not None:
            usage.bump("retrieval_calls")
        out = []
        for r in (getattr(search, "data", None) or [])[:top_n]:
            src, title = _source_from(getattr(r, "filename", ""), getattr(r, "attributes", None))
            out.append(_passage(_fs_result_text(r), title, src, getattr(r, "score", None),
                                self.name))
        return out


def make_engine(name: str, *, chroma_dir: str | None = None, collection: str = "oncology_db",
                embed_model: str = "text-embedding-3-large",
                vector_store_id: str | None = None, llm_model: str = "gpt-5.1", rag=None):
    """Build the retrieval engine for ``name`` (one of :data:`ENGINES`).

    The deprecated aliases ``openai_fs_responses`` / ``openai_fs_chat`` are accepted and mapped
    to their canonical ``openai_filesearch_*`` names. The ``openai_filesearch_*`` engines
    require ``vector_store_id``."""
    name = _ALIASES.get(name, name)
    if name == "chroma_cosine":
        return ChromaEngine(chroma_dir, collection, embed_model, rerank=False, rag=rag)
    if name == "chroma_cohere":
        return ChromaEngine(chroma_dir, collection, embed_model, rerank=True, rag=rag)
    if name == "openai_filesearch_responses":
        if not vector_store_id:
            raise ValueError("openai_filesearch_responses requires vector_store_id")
        return OpenAIFileSearchResponsesEngine(vector_store_id, llm_model)
    if name == "openai_filesearch_chat":
        if not vector_store_id:
            raise ValueError("openai_filesearch_chat requires vector_store_id")
        return OpenAIFileSearchChatEngine(vector_store_id, llm_model)
    raise ValueError(f"unknown retrieval engine {name!r}; expected one of {ENGINES}")
