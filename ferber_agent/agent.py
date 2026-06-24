"""FerberAgent: autonomous tool-use + guideline-RAG, on a modern OpenAI-SDK stack.

A modernized reimplementation of the Ferber et al. (Nature Cancer 2025) RAG agent's
method. The original used dspy + llama-index 0.9 + a vendored OpenAI agent + cohere rerank;
those APIs are deprecated/removed, so the *method* is reimplemented on the current OpenAI SDK
(function-calling) + chromadb. The agent exposes two modes:

Default mode (``faithful=False``) — a compact two-stage loop:
  Stage 1 — autonomous tool use: the model decides which tools to call (OncoKB genomic
            annotation, PubMed, guideline RAG, calculate) and gathers evidence.
  Stage 2 — RAG-grounded answer: guideline passages are retrieved and the model produces a
            grounded clinical answer citing them.

Faithful mode (``faithful=True``) — the paper's full multi-stage pipeline, with the verbatim
upstream prompt strings (see ``faithful_prompts``):
  Stage 1 — autonomous tool gathering (oncokb / pubmed / calculate, plus imaging when images
            are supplied); guideline ``rag`` is NOT a callable tool here.
  Stage 2 — mandatory guideline grounding: Search subquery fan-out (up to ``n_subqueries``)
            -> retrieve top-k per subquery -> Cohere rerank top-n -> dedup union ->
            AnswerStrategy -> GenerateCitedResponse -> optional one-pass citation
            self-evaluation -> Suggestions.

The independent per-subquery retrievals and per-statement citation checks fan out across a
thread pool with order-preserving reassembly, so the concurrency is result-preserving (see
``_map_parallel``). Reranking falls back to cosine-only without ``COHERE_API_KEY``.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from . import faithful_prompts as _fp
from .result import FerberResult
from .tools import (
    RagTool,
    calculate,
    calculate_faithful,
    faithful_tool_schemas,
    histology_classifier_unavailable,
    histology_replay,
    medsam_segment,
    oncokb_annotate,
    oncokb_annotate_faithful,
    pubmed_search,
    pubmed_search_faithful,
    radiology_report,
    radiology_report_folder,
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

    # More retries + a generous timeout for concurrent per-case fan-out: the SDK default
    # max_retries=2 is too few to ride out transient 429/5xx when many cases run at once.
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=6, timeout=180.0)


def _is_reasoning(model: str) -> bool:
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _create(model: str, messages: list[dict], tools=None, max_tokens: int = 6000,
            temperature: float = 0.1):
    """One chat.completions call. Reasoning models (gpt-5/o*) drop temperature and use
    ``max_completion_tokens``; GPT-4-era models take ``temperature`` (the paper specifies
    0.2 for the agent stage, 0.1 for the RAG stages — passed in by the caller)."""
    kw: dict = {"model": model, "messages": messages}
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
    if _is_reasoning(model):
        kw["max_completion_tokens"] = max_tokens
    else:
        kw["max_tokens"] = max_tokens
        kw["temperature"] = temperature
    return _client().chat.completions.create(**kw)


class FerberAgent:
    def __init__(self, chroma_dir: str, collection: str = "oncology_db",
                 llm_model: str = "gpt-5.1", embed_model: str = "text-embedding-3-large",
                 tools: tuple[str, ...] = ("rag", "oncokb", "pubmed", "calculate"),
                 rerank: bool = False, retrieve_k: int = 20, rerank_top_n: int = 10,
                 max_tool_iters: int = 6, vision_model: str | None = None,
                 faithful: bool = False, n_subqueries: int = 12,
                 max_function_calls: int = 10, synth_max_tokens: int = 4096,
                 agent_temp: float = 0.2, rag_temp: float = 0.1,
                 citation_selfeval: bool | None = None):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.tool_names = tuple(tools)
        self.rerank = rerank and bool(os.environ.get("COHERE_API_KEY"))
        self.retrieve_k = retrieve_k
        self.rerank_top_n = rerank_top_n
        self.max_tool_iters = max_tool_iters
        self.vision_model = vision_model or llm_model
        # --- faithful mode: the paper's full multi-stage pipeline ---
        self.faithful = faithful
        self.n_subqueries = n_subqueries
        # Result-preserving fan-out: the independent per-statement citation checks and the
        # per-subquery retrievals run across a thread pool. Outputs are reassembled in their
        # original order, so only the execution is concurrent (see _map_parallel).
        self.rag_workers = int(os.environ.get("RAG_WORKERS", "12"))
        self.citation_workers = int(os.environ.get("CITATION_WORKERS", "12"))
        self.max_function_calls = max_function_calls
        self.synth_max_tokens = synth_max_tokens
        self.agent_temp = agent_temp
        self.rag_temp = rag_temp
        # The paper runs a single iteration of citation self-evaluation after synthesis.
        self.citation_selfeval = (bool(os.environ.get("FERBER_CITATION_SELFEVAL"))
                                  if citation_selfeval is None else bool(citation_selfeval))
        self._case_key: str | None = None  # set via answer(case_key=...) for the histology replay
        # In faithful mode guideline retrieval is the MANDATORY grounding stage (not a model-
        # callable tool), so the retriever is always built and "rag" is never a callable tool.
        self._rag = RagTool(chroma_dir, collection, embed_model,
                            k=(retrieve_k if not faithful else self.retrieve_k),
                            embed_cache_dir=os.environ.get("EMBED_CACHE_DIR")) \
            if (chroma_dir and (faithful or "rag" in self.tool_names)) else None
        # per-call image map (referenced filename -> absolute path); set in answer()
        self._images: dict[str, str] = {}

    _IMAGING_TOOLS = ("radiology_report", "medsam", "histology_classifier")

    def _active_tools(self) -> tuple[str, ...]:
        """Imaging-tool schemas are exposed only when the call carries an image map, so the
        text-only genomic track (e.g. MTBBench) is unaffected and behavior-preserving."""
        if self._images:
            return self.tool_names
        return tuple(t for t in self.tool_names if t not in self._IMAGING_TOOLS)

    def _faithful_tools(self) -> tuple[str, ...]:
        """Faithful callable tool set: the patient-evidence tools (oncokb/pubmed/calculate +
        imaging when images are present). Guideline ``rag`` is NOT callable — it is the
        mandatory Stage-2 grounding step in the published method."""
        base = ["oncokb", "pubmed", "calculate"]
        if self._images:
            base += ["radiology_report", "medsam", "histology_classifier"]
        return tuple(base)

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

    # === faithful mode: the paper's described pipeline =====================
    @staticmethod
    def _map_parallel(fn, items, workers: int) -> list:
        """Apply ``fn`` to each item concurrently, returning results in INPUT order.

        Result-preserving fan-out: a thread pool runs the independent calls in parallel
        but the output list mirrors the input ordering exactly, so downstream code sees
        the same sequence it would from a serial loop. Exceptions propagate (parity with
        the serial loops, which do not swallow them)."""
        items = list(items)
        if not items:
            return []
        w = max(1, min(int(workers), len(items)))
        if w == 1:
            return [fn(x) for x in items]
        results: list = [None] * len(items)
        with ThreadPoolExecutor(max_workers=w) as ex:
            futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        return results

    @staticmethod
    def _parse_query_list(raw: str) -> list[str]:
        """Parse the Search-stage reply (a python list as a string) into query strings, with a
        tolerant fallback for bulleted/quoted output."""
        import ast as _ast
        raw = (raw or "").strip()
        if raw.startswith("```"):
            body = raw[3:]
            if "```" in body:
                body = body[: body.index("```")]
            raw = body[4:].strip() if body.lstrip().lower().startswith("json") else body.strip()
        for parser in (json.loads, _ast.literal_eval):
            try:
                data = parser(raw)
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except Exception:  # noqa: BLE001
                pass
        out = []
        for line in raw.splitlines():
            line = line.strip().lstrip("-*0123456789.) ").strip().strip("'").strip('"').strip()
            if line and not line.startswith(("[", "]", "{", "}")):
                out.append(line)
        return out

    def _dispatch_faithful(self, name: str, args: dict, retrieved_acc: list) -> str:
        """Faithful tool dispatch: restored parameter shapes mapped to implementations."""
        if name == "oncokb":
            return oncokb_annotate_faithful(args.get("hugo_symbol", ""),
                                            args.get("change", "mutation"),
                                            args.get("alteration", ""))
        if name == "pubmed":
            return pubmed_search_faithful(args.get("pubmed_search_terms", []),
                                          args.get("query", ""))
        if name == "calculate":
            return calculate_faithful(args.get("a", 0), args.get("b", 0),
                                      args.get("operator", "+"))
        if name == "radiology_report":
            # The original took a patient-folder path; our "folder" is the case's resolved image
            # set (every attached image), read each + a comparison pass.
            paths = sorted(set(self._images.values()))
            if not paths:
                return "No images available for this patient."
            return radiology_report_folder(paths, args.get("query", ""), model=self.vision_model)
        if name == "medsam":
            path = resolve_image(args.get("path_to_img", ""), self._images) \
                or (sorted(set(self._images.values()))[0] if self._images else None)
            if path is None:
                return "no image available for segmentation"
            boxes = args.get("bbox_coordinates") or []
            box = boxes[0] if boxes and isinstance(boxes[0], list) else boxes
            if not box or len(box) != 4:
                return "medsam needs bbox_coordinates=[[x_min, y_min, x_max, y_max]]"
            return medsam_segment(path, box)
        if name == "histology_classifier":
            return histology_replay(self._case_key or args.get("patient_id", ""),
                                    args.get("targets"))
        return f"unknown tool {name}"

    def _stage(self, doc: str, field_name: str, field_desc: str, user: str,
               max_tokens: int = 2000) -> str:
        """Run one dspy-style signature stage as a single chat call: the verbatim docstring is
        the instruction, the verbatim OutputField desc specifies the output, the user message
        carries the (labelled) input fields. Returns the model's text."""
        system = f"{doc}\n\nProduce the field `{field_name}`: {field_desc}"
        resp = _create(self.llm_model,
                       [{"role": "system", "content": system},
                        {"role": "user", "content": user}],
                       max_tokens=max_tokens, temperature=self.rag_temp)
        return resp.choices[0].message.content or ""

    def _citation_selfeval(self, cited: str, retrieved: list[dict], cited_user: str,
                           max_checks: int = 12) -> tuple[str, dict]:
        """Paper's single-iteration citation self-evaluation.

        For each cited statement, run CheckCitationFaithfulness (verbatim signature) to verify it
        is supported by its retrieved context; if any statement is flagged unfaithful, re-run
        GenerateCitedResponse ONCE to revise those statements (no backtracking loop — a single
        iteration, mirroring the paper). Returns (possibly revised answer, record)."""
        src_text = {i: (h.get("text", "") or "")[:700] for i, h in enumerate(retrieved, start=1)}
        full_ctx = "\n\n".join(f"[{i}] {t}" for i, t in src_text.items())[:6000]
        stmts: list[tuple[str, list[int]]] = []
        for sent in re.split(r"(?<=[.!?])\s+", cited):
            idxs = [int(x) for x in re.findall(r"\[(\d+)\]", sent)]
            if idxs and sent.strip():
                stmts.append((sent.strip(), idxs))
        # fallback so the verbatim faithfulness prompt is exercised even if no [n] citations parse
        if not stmts:
            first = next((s.strip() for s in re.split(r"(?<=[.!?])\s+", cited) if s.strip()),
                         cited[:300])
            stmts = [(first, list(src_text.keys())[:3])]

        # The per-statement faithfulness checks are independent (each sees only its own
        # statement + cited context), so they fan out across a thread pool. Verdicts are
        # collected in statement order and the unfaithful list is rebuilt in that same
        # order, so the result is identical to the serial loop — only faster.
        targets = stmts[:max_checks]

        def _check(item):
            stmt, idxs = item
            ctx = "\n\n".join(src_text.get(i, "") for i in idxs if i in src_text) or full_ctx
            check_user = (f"context ({_fp.CHECK_CITATION_CONTEXT_DESC}):\n{ctx}\n\n"
                          f"text ({_fp.CHECK_CITATION_TEXT_DESC}): {stmt}")
            return self._stage(_fp.CHECK_CITATION_DOC, "faithfulness",
                               _fp.CHECK_CITATION_FAITHFULNESS_DESC, check_user,
                               max_tokens=600)

        verdicts = self._map_parallel(_check, targets, self.citation_workers)
        checked = len(targets)
        unfaithful: list[str] = []
        for (stmt, _idxs), verdict in zip(targets, verdicts):
            low = (verdict or "").strip().lower()[:60]
            if any(w in low for w in ("false", "unfaithful", "not faithful", "not supported")):
                unfaithful.append(stmt)
        record = {"checked": checked, "unfaithful": len(unfaithful), "revised": False}
        if unfaithful:
            note = ("\n\nThe following statements in your previous response were flagged as NOT "
                    "faithful to the cited context. Revise the response so every cited claim is "
                    "supported by its [source], correcting or removing each unsupported claim:\n"
                    + "\n".join(f"- {s}" for s in unfaithful))
            revised = self._stage(_fp.GENCITED_DOC, "response", _fp.GENCITED_RESPONSE_DESC,
                                  cited_user + note, max_tokens=self.synth_max_tokens)
            if revised.strip():
                record["revised"] = True
                return revised, record
        return cited, record

    def _answer_faithful(self, context: str, question: str) -> FerberResult:
        active = self._faithful_tools()
        schemas = faithful_tool_schemas(active)
        # --- Stage 1: autonomous tool gathering (verbatim agent + chat_ext prompts) ----
        instruction = _fp.CHAT_EXT_INSTRUCTION.replace("{question}", question)
        messages: list[dict] = [
            {"role": "system", "content": _fp.AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n{instruction}"},
        ]
        tool_calls: list[dict] = []
        retrieved: list[dict] = []
        nudged = False
        last_content = ""
        for _ in range(self.max_function_calls):
            resp = _create(self.llm_model, messages, tools=schemas,
                           max_tokens=self.synth_max_tokens, temperature=self.agent_temp)
            msg = resp.choices[0].message
            tcs = msg.tool_calls or []
            last_content = msg.content or last_content
            assistant_msg = {"role": "assistant", "content": msg.content or ""}
            if tcs:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tcs]
            messages.append(assistant_msg)
            if not tcs:
                # mirror the original _should_continue: nudge once to use ALL tools, then stop.
                if not nudged:
                    nudged = True
                    messages.append({"role": "user", "content": _fp.MUST_USE_ALL_TOOLS_NUDGE})
                    continue
                break
            for tc in tcs:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._dispatch_faithful(tc.function.name, args, retrieved)
                tool_calls.append({"tool": tc.function.name, "args": args, "result": result[:1500]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result[:6000]})
        tool_results = last_content or "No tools were used."

        # --- Stage 2: mandatory guideline RAG (Search -> retrieve -> strategy -> cited -> sugg) --
        # 2a. Search: expand into focused subqueries (verbatim Search signature).
        sub_user = (f"question: {question}\ncontext: {context[:3000]}\n"
                    f"tool_results: {tool_results[:3000]}")
        raw_subs = self._stage(_fp.SEARCH_DOC, "searches", _fp.SEARCH_SEARCHES_DESC,
                               sub_user, max_tokens=1500)
        subqueries: list[str] = []
        for q in [question, *self._parse_query_list(raw_subs)]:
            q = (q or "").strip()
            if q and q not in subqueries:
                subqueries.append(q)
        subqueries = subqueries[: self.n_subqueries]

        # 2b. retrieve top-K per subquery -> rerank top-N -> union/dedup.
        # The 12 subquery retrievals are independent, so they fan out across a thread pool.
        # Each subquery's retrieve+rerank is unchanged; the per-subquery hit
        # lists are returned in subquery order and the dedup'd union is built in that SAME
        # order, so the resulting passage list is byte-identical to the serial loop —
        # the [n] citation indices downstream are unaffected. Embedding is cached and runs
        # outside the Chroma lock, so the slow OpenAI embed step is what actually parallelizes.
        def _fetch(sq: str) -> list[dict]:
            try:
                # cached chroma-retrieve + rerank: deterministic per (query,k,top_n),
                # so the parallel fan-out is provably exact and the agent is reproducible.
                return self._rag.retrieve_reranked(sq, self.retrieve_k, self.rerank_top_n,
                                                   self.rerank)
            except Exception:  # noqa: BLE001 — drop this subquery (parity with serial `continue`)
                return []

        per_subquery = self._map_parallel(_fetch, subqueries, self.rag_workers)
        seen: set = set()
        union: list[dict] = []
        for hits in per_subquery:  # per_subquery preserves subquery order
            for h in hits:
                key = (h.get("source", ""), h.get("title", ""), (h.get("text", "") or "")[:200])
                if key in seen:
                    continue
                seen.add(key)
                union.append(h)
        retrieved = union
        # passages prefixed "Source {idx}:" so the [x] citation instruction resolves.
        passages = "\n\n".join(
            f"Source {i}: [{h.get('source','')}: {h.get('title','')}] {(h.get('text','') or '')[:700]}"
            for i, h in enumerate(retrieved, start=1)) or "Source 1: (no guideline passages retrieved)"

        patient_field = "Patient:\n" + context
        tool_field = "Tool:\n" + tool_results
        question_field = "Question:\n" + question
        tools_desc = "These tools are available to you:" + str(
            [s["function"]["description"] for s in schemas])

        # 2c. AnswerStrategy (verbatim).
        strat_user = (f"context: {passages}\npatient: {patient_field}\n"
                      f"tool_results: {tool_field}\nquestion: {question_field}")
        strategy = self._stage(_fp.STRATEGY_DOC, "response", _fp.STRATEGY_RESPONSE_DESC,
                               strat_user, max_tokens=2500)

        # 2d. RequireInput (verbatim) — context omitted, mirroring rag.py.
        req_user = (f"patient: {patient_field}\ntool_results: {tool_field}\n"
                    f"tools: {tools_desc}\nquestion: {question_field}")
        ask_for_more = self._stage(_fp.REQUIREINPUT_DOC, "response",
                                   _fp.REQUIREINPUT_RESPONSE_DESC, req_user, max_tokens=1200)

        # 2e. GenerateCitedResponse (verbatim) — the main answer.
        cited_user = (
            f"strategy: {strategy}\n"
            f"context ({_fp.GENCITED_CONTEXT_DESC}):\n{passages}\n"
            f"patient ({_fp.GENCITED_PATIENT_DESC}): {patient_field}\n"
            f"tool_results ({_fp.GENCITED_TOOLRESULTS_DESC}): {tool_field}\n"
            f"question: {question_field}")
        cited = self._stage(_fp.GENCITED_DOC, "response", _fp.GENCITED_RESPONSE_DESC,
                            cited_user, max_tokens=self.synth_max_tokens)

        # 2e'. Citation self-evaluation (paper's single iteration) — verify each cited
        # statement against its retrieved context and revise unfaithful ones exactly once.
        selfeval_rec: dict = {"enabled": False}
        if self.citation_selfeval:
            cited, selfeval_rec = self._citation_selfeval(cited, retrieved, cited_user)
            selfeval_rec["enabled"] = True

        # 2f. Suggestions (verbatim) — appended to the cited response.
        sugg_user = f"response: {cited}\nrecommendations: {ask_for_more}"
        suggestions = self._stage(_fp.SUGGESTIONS_DOC, "suggestions",
                                  _fp.SUGGESTIONS_SUGGESTIONS_DESC, sugg_user, max_tokens=1000)

        answer_text = (cited + "\n\n" + suggestions).strip()
        # record the mandatory grounding as a synthetic tool entry for retrieval-sanity review
        # (rag has no Table-1 counterpart, so it does not affect tool-use fidelity).
        tool_calls.append({
            "tool": "rag",
            "args": {"mode": "mandatory_stage2_grounding", "n_subqueries": len(subqueries),
                     "subqueries": subqueries, "citation_selfeval": selfeval_rec},
            "result": f"retrieved {len(retrieved)} passages from "
                      f"{sorted({h.get('source','') for h in retrieved})}"[:1500],
        })
        citations = [{"source": h.get("source", ""), "title": h.get("title", "")}
                     for h in retrieved[:10]]
        return FerberResult(answer_text=answer_text, citations=citations,
                            tool_calls=tool_calls, retrieved=retrieved)

    def answer(self, context: str, question: str,
               images: dict[str, str] | None = None,
               case_key: str | None = None) -> FerberResult:
        self._images = dict(images or {})
        if case_key is not None:
            self._case_key = case_key
        if self.faithful:
            return self._answer_faithful(context, question)
        active = self._active_tools()
        schemas = tool_schemas(active)
        messages: list[dict] = [
            {"role": "system", "content": _system_prompt(active)},
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
