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
from .tools import (
    RagTool,
    calculate,
    histology_classifier_unavailable,
    medsam_segment,
    oncokb_annotate,
    pubmed_search,
    radiology_report,
    rerank_passages,
    resolve_image,
    tool_schemas,
)

_SYSTEM_BASE = (
    "You are an expert molecular tumor board assistant. Given a patient's clinical and "
    "genomic context and a question, gather evidence with the available tools before "
    "answering: use OncoKB for the oncogenicity and therapy implications of specific gene "
    "alterations, PubMed for relevant trials/literature, and the guideline RAG tool for "
    "standard-of-care guidance."
)
_SYSTEM_IMAGING = (
    " For imaging, call radiology_report on a referenced image to get a structured read, and "
    "medsam with a lesion bounding box (the [x_min, y_min, x_max, y_max] coordinates quoted "
    "in the report) to measure its area in pixels; compare two timepoints' areas with "
    "calculate to quantify progression. The in-house histology KRAS/BRAF/MSI classifier is "
    "unavailable — rely on the molecular report for those."
)
_SYSTEM_TAIL = " Call tools when they would help; then reason carefully."


def _system_prompt(tool_names: tuple[str, ...]) -> str:
    msg = _SYSTEM_BASE
    if any(t in tool_names for t in ("radiology_report", "medsam", "histology_classifier")):
        msg += _SYSTEM_IMAGING
    return msg + _SYSTEM_TAIL
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
                 rerank: bool = False, retrieve_k: int = 20, rerank_top_n: int = 10,
                 max_tool_iters: int = 6, vision_model: str | None = None):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.tool_names = tuple(tools)
        self.rerank = rerank and bool(os.environ.get("COHERE_API_KEY"))
        self.retrieve_k = retrieve_k
        self.rerank_top_n = rerank_top_n
        self.max_tool_iters = max_tool_iters
        self.vision_model = vision_model or llm_model
        self._rag = RagTool(chroma_dir, collection, embed_model, k=retrieve_k) \
            if "rag" in self.tool_names else None
        # per-call image map (referenced filename -> absolute path); set in answer()
        self._images: dict[str, str] = {}

    def _retrieve(self, query: str) -> list[dict]:
        """Cosine retrieval, then Cohere rerank when enabled (else cosine order)."""
        hits = self._rag.query(query)
        if self.rerank:
            hits = rerank_passages(query, hits, top_n=self.rerank_top_n)
        return hits

    # --- tool dispatch -------------------------------------------------------
    def _dispatch(self, name: str, args: dict, retrieved_acc: list) -> str:
        if name == "rag" and self._rag is not None:
            hits = self._retrieve(args.get("query", ""))
            retrieved_acc.extend(hits)
            return "\n\n".join(
                f"[{h['source']}: {h['title']}] {h['text'][:500]}" for h in hits[:8]) or "no hits"
        if name == "oncokb":
            return oncokb_annotate(args.get("hugo_symbol", ""), args.get("alteration", ""))
        if name == "pubmed":
            return pubmed_search(args.get("query", ""))
        if name == "calculate":
            return calculate(args.get("expression", ""))
        if name == "radiology_report":
            path = resolve_image(args.get("image_ref", ""), self._images)
            if path is None:
                return (f"image '{args.get('image_ref', '')}' not available "
                        f"(known: {sorted(set(self._images))})")
            return radiology_report(path, args.get("clinical_question", ""),
                                    model=self.vision_model)
        if name == "medsam":
            path = resolve_image(args.get("image_ref", ""), self._images)
            if path is None:
                return (f"image '{args.get('image_ref', '')}' not available "
                        f"(known: {sorted(set(self._images))})")
            bbox = args.get("bbox") or []
            if len(bbox) != 4:
                return "medsam needs bbox=[x_min, y_min, x_max, y_max]"
            return medsam_segment(path, bbox)
        if name == "histology_classifier":
            return histology_classifier_unavailable(args.get("marker", ""),
                                                    args.get("image_ref", ""))
        return f"unknown tool {name}"

    def answer(self, context: str, question: str,
               images: dict[str, str] | None = None) -> FerberResult:
        self._images = dict(images or {})
        schemas = tool_schemas(self.tool_names)
        messages: list[dict] = [
            {"role": "system", "content": _system_prompt(self.tool_names)},
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
            retrieved.extend(self._retrieve(f"{question}\n{context[:500]}"))
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
