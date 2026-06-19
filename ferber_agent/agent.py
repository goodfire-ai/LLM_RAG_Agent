"""FerberAgent: autonomous tool-use + guideline-RAG, on a modern OpenAI-SDK stack.

A modernized reimplementation of the Ferber et al. (Nature Cancer 2025) RAG agent's
method for the genomic text track. The original used dspy + llama-index 0.9 + a vendored
OpenAI agent + cohere rerank; those APIs are deprecated/removed, so the *method* is
reimplemented faithfully on the current OpenAI SDK (function-calling) + chromadb:

  Stage 1 — autonomous tool use: the model decides which tools to call (OncoKB genomic
            annotation, PubMed, guideline RAG, calculate) and gathers evidence.
  Stage 2 — RAG-grounded answer: guideline passages are retrieved and the model produces a
            grounded clinical answer citing them.

Reranking falls back to cosine-only without COHERE_API_KEY (a recorded deviation).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from .result import FerberResult
from .tools import RagTool, calculate, oncokb_annotate, pubmed_search, tool_schemas

_SYSTEM = (
    "You are an expert molecular tumor board assistant. Given a patient's clinical and "
    "genomic context and a question, gather evidence with the available tools before "
    "answering: use OncoKB for the oncogenicity and therapy implications of specific gene "
    "alterations, PubMed for relevant trials/literature, and the guideline RAG tool for "
    "standard-of-care guidance. Call tools when they would help; then reason carefully."
)
_SYNTH = (
    "Using the patient context, the evidence gathered from tools, and the retrieved "
    "clinical-guideline passages below, give a concise, clinically grounded answer to the "
    "question. Cite guideline passages by their [source: title] when you rely on them."
)


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _is_reasoning(model: str) -> bool:
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _create(model: str, messages: list[dict], tools=None, max_tokens: int = 6000):
    kw: dict = {"model": model, "messages": messages}
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
    if _is_reasoning(model):
        kw["max_completion_tokens"] = max_tokens
    else:
        kw["max_tokens"] = max_tokens
        kw["temperature"] = 0.1
    return _client().chat.completions.create(**kw)


class FerberAgent:
    def __init__(self, chroma_dir: str, collection: str = "oncology_db",
                 llm_model: str = "gpt-5.1", embed_model: str = "text-embedding-3-large",
                 tools: tuple[str, ...] = ("rag", "oncokb", "pubmed", "calculate"),
                 rerank: bool = False, retrieve_k: int = 20, max_tool_iters: int = 6):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.tool_names = tuple(tools)
        self.rerank = rerank and bool(os.environ.get("COHERE_API_KEY"))
        self.retrieve_k = retrieve_k
        self.max_tool_iters = max_tool_iters
        self._rag = RagTool(chroma_dir, collection, embed_model, k=retrieve_k) \
            if "rag" in self.tool_names else None

    # --- tool dispatch -------------------------------------------------------
    def _dispatch(self, name: str, args: dict, retrieved_acc: list) -> str:
        if name == "rag" and self._rag is not None:
            hits = self._rag.query(args.get("query", ""))
            retrieved_acc.extend(hits)
            return "\n\n".join(
                f"[{h['source']}: {h['title']}] {h['text'][:500]}" for h in hits[:8]) or "no hits"
        if name == "oncokb":
            return oncokb_annotate(args.get("hugo_symbol", ""), args.get("alteration", ""))
        if name == "pubmed":
            return pubmed_search(args.get("query", ""))
        if name == "calculate":
            return calculate(args.get("expression", ""))
        return f"unknown tool {name}"

    def answer(self, context: str, question: str) -> FerberResult:
        schemas = tool_schemas(self.tool_names)
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Patient context:\n{context}\n\nQuestion:\n{question}"},
        ]
        tool_calls: list[dict] = []
        retrieved: list[dict] = []

        # Stage 1: autonomous tool use
        for _ in range(self.max_tool_iters):
            resp = _create(self.llm_model, messages, tools=schemas)
            msg = resp.choices[0].message
            tcs = msg.tool_calls or []
            assistant_msg = {"role": "assistant", "content": msg.content or ""}
            if tcs:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tcs
                ]
            messages.append(assistant_msg)
            if not tcs:
                break
            for tc in tcs:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._dispatch(tc.function.name, args, retrieved)
                tool_calls.append({"tool": tc.function.name, "args": args, "result": result[:1500]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result[:6000]})

        # Stage 2: ensure guideline grounding, then synthesize a cited answer
        if self._rag is not None and not retrieved:
            retrieved.extend(self._rag.query(f"{question}\n{context[:500]}"))
        passages = "\n\n".join(
            f"[{h['source']}: {h['title']}] {h['text'][:600]}" for h in retrieved[:10])
        synth_messages = [
            {"role": "system", "content": _SYNTH},
            {"role": "user", "content": (
                f"Patient context:\n{context}\n\nQuestion:\n{question}\n\n"
                f"Retrieved guideline passages:\n{passages or '(none retrieved)'}")},
        ]
        final = _create(self.llm_model, synth_messages)
        answer_text = final.choices[0].message.content or ""

        citations = [{"source": h["source"], "title": h["title"]} for h in retrieved[:10]]
        return FerberResult(answer_text=answer_text, citations=citations,
                            tool_calls=tool_calls, retrieved=retrieved)
