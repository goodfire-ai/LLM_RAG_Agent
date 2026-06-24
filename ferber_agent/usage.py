"""Per-rollout usage / cost / latency accounting, shared across the three backends.

Token counts are the ground truth (read straight off the API ``usage`` objects). Dollar
cost is APPROXIMATE: it multiplies tokens by a configurable price table (USD per 1M tokens)
plus a per-call web-search / file-search price. Prices are overridable via env
(``OPENAI_PRICE_*``), default to gpt-5.1 list prices, and are labelled approximate everywhere
they surface. Latency is wall-clock seconds spent inside API calls for one case.

One :class:`UsageAccumulator` lives per :class:`~ferber_agent.agent.FerberAgent` and is reset
at the start of every ``answer()`` call. Agents answer one case at a time within a thread, so
the accumulator is not shared across concurrent cases and needs no locking.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field


def _price(env_key: str, default: float) -> float:
    try:
        return float(os.environ.get(env_key, default))
    except (TypeError, ValueError):
        return default


def price_table() -> dict:
    """USD per 1M tokens (approximate gpt-5.1 list prices; override via env). The
    ``*_per_call`` entries are per hosted-tool invocation, not per token."""
    return {
        "input_per_m": _price("OPENAI_PRICE_INPUT_PER_M", 1.25),
        "cached_input_per_m": _price("OPENAI_PRICE_CACHED_INPUT_PER_M", 0.125),
        "output_per_m": _price("OPENAI_PRICE_OUTPUT_PER_M", 10.0),
        "web_search_per_call": _price("OPENAI_PRICE_WEB_SEARCH_PER_CALL", 0.01),
        "embed_per_m": _price("OPENAI_PRICE_EMBED_PER_M", 0.13),
        # OpenAI file_search tool call price (~$2.50 / 1000 calls).
        "filesearch_per_call": _price("OPENAI_PRICE_FILESEARCH_PER_CALL", 0.0025),
    }


@dataclass
class UsageAccumulator:
    """Mutable per-case tally of tokens, hosted-tool calls, latency, and approximate cost."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    web_search_calls: int = 0
    web_search_cited: int = 0          # web_search calls whose results were cited in an answer
    n_api_calls: int = 0               # backbone (chat/responses) calls
    n_tool_exec: int = 0               # function tools actually executed (oncokb/pubmed/etc.)
    # retrieval-engine accounting (per-engine cost / latency of the retrieval step).
    retrieval_calls: int = 0           # subquery retrievals issued (all engines)
    rerank_calls: int = 0              # Cohere rerank calls (chroma_cohere only)
    filesearch_calls: int = 0          # OpenAI file_search tool invocations (Responses engine)
    vstore_searches: int = 0           # OpenAI vector_stores.search calls (chat file-search engine)
    retrieval_seconds: float = 0.0     # wall-clock inside the retrieval step
    api_seconds: float = 0.0           # wall-clock inside API calls
    by_stage: dict = field(default_factory=dict)  # stage -> {calls, in, out}
    _t0: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        # The two faithful-mode workloads (per-subquery retrieval and per-statement citation
        # checks) fan out across a thread pool, so every counter mutation is guarded by this
        # lock. Guarding only keeps the accounting numbers exact; the agent's answer is
        # result-preserving regardless (retrieval is deterministic).
        self._lock = threading.Lock()

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def bump(self, attr: str, n: int = 1) -> None:
        """Thread-safe ``self.<attr> += n`` for the integer counters mutated during fan-out."""
        with self._lock:
            setattr(self, attr, getattr(self, attr) + n)

    # --- chat.completions usage (chat_completions backend) ---
    def add_chat(self, resp, stage: str = "") -> None:
        u = getattr(resp, "usage", None)
        pt = ct = cached = rt = 0
        if u is not None:
            pt = int(getattr(u, "prompt_tokens", 0) or 0)
            ct = int(getattr(u, "completion_tokens", 0) or 0)
            details = getattr(u, "prompt_tokens_details", None)
            if details is not None:
                cached = int(getattr(details, "cached_tokens", 0) or 0)
            cdetails = getattr(u, "completion_tokens_details", None)
            if cdetails is not None:
                rt = int(getattr(cdetails, "reasoning_tokens", 0) or 0)
        with self._lock:
            self.n_api_calls += 1
            self._add(pt, cached, ct, rt, stage)

    # --- responses usage (responses_faithful / native_agentic backends) ---
    def add_responses(self, resp, stage: str = "") -> None:
        u = getattr(resp, "usage", None)
        it = ot = cached = rt = 0
        if u is not None:
            it = int(getattr(u, "input_tokens", 0) or 0)
            ot = int(getattr(u, "output_tokens", 0) or 0)
            idet = getattr(u, "input_tokens_details", None)
            if idet is not None:
                cached = int(getattr(idet, "cached_tokens", 0) or 0)
            odet = getattr(u, "output_tokens_details", None)
            if odet is not None:
                rt = int(getattr(odet, "reasoning_tokens", 0) or 0)
        n_web = sum(1 for item in (getattr(resp, "output", []) or [])
                    if getattr(item, "type", None) == "web_search_call")
        with self._lock:
            self.n_api_calls += 1
            self._add(it, cached, ot, rt, stage)
            self.web_search_calls += n_web

    def _add(self, in_tok: int, cached: int, out_tok: int, reasoning: int, stage: str) -> None:
        """Apply token deltas. Caller must hold ``self._lock``."""
        self.input_tokens += in_tok
        self.cached_input_tokens += cached
        self.output_tokens += out_tok
        self.reasoning_tokens += reasoning
        if stage:
            s = self.by_stage.setdefault(stage, {"calls": 0, "in": 0, "out": 0})
            s["calls"] += 1
            s["in"] += in_tok
            s["out"] += out_tok

    def add_web_search_call(self, n: int = 1) -> None:
        """For the chat backend, where web_search runs as a nested Responses call inside a
        function tool and so is not visible as a ``web_search_call`` output item."""
        self.bump("web_search_calls", n)

    def retrieval_timer(self) -> "UsageAccumulator._RetrievalTimer":
        """Context manager: adds wall-clock spent in the with-block to ``retrieval_seconds``."""
        return UsageAccumulator._RetrievalTimer(self)

    class _RetrievalTimer:
        def __init__(self, acc: "UsageAccumulator"):
            self.acc = acc

        def __enter__(self):
            self._t = time.time()
            return self

        def __exit__(self, *exc):
            with self.acc._lock:
                self.acc.retrieval_seconds += time.time() - self._t
            return False

    class _Timer:
        def __init__(self, acc: "UsageAccumulator"):
            self.acc = acc

        def __enter__(self):
            self._t = time.time()
            return self

        def __exit__(self, *exc):
            with self.acc._lock:
                self.acc.api_seconds += time.time() - self._t
            return False

    def timer(self, *_ignored) -> "_Timer":
        """Context manager that adds wall-clock spent in the with-block to ``api_seconds``."""
        return UsageAccumulator._Timer(self)

    def cost_usd(self) -> float:
        """Approximate USD cost for this case from the configurable price table."""
        p = price_table()
        billed_input = max(0, self.input_tokens - self.cached_input_tokens)
        return (
            billed_input / 1e6 * p["input_per_m"]
            + self.cached_input_tokens / 1e6 * p["cached_input_per_m"]
            + self.output_tokens / 1e6 * p["output_per_m"]
            + self.web_search_calls * p["web_search_per_call"]
            + self.filesearch_calls * p["filesearch_per_call"]
        )

    def summary(self) -> dict:
        """Serializable snapshot of this case's usage (suitable for a result record)."""
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "web_search_calls": self.web_search_calls,
            "web_search_cited": self.web_search_cited,
            "n_api_calls": self.n_api_calls,
            "n_tool_exec": self.n_tool_exec,
            "retrieval_calls": self.retrieval_calls,
            "rerank_calls": self.rerank_calls,
            "filesearch_calls": self.filesearch_calls,
            "vstore_searches": self.vstore_searches,
            "retrieval_seconds": round(self.retrieval_seconds, 2),
            "api_seconds": round(self.api_seconds, 2),
            "approx_cost_usd": round(self.cost_usd(), 4),
            "by_stage": self.by_stage,
            "price_table": price_table(),
        }
