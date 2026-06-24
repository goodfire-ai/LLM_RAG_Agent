"""Shared hermetic fixtures for the faithful-pipeline tests.

These build a FerberAgent whose three external surfaces are replaced by deterministic stubs,
so the full faithful pipeline runs offline (no OpenAI / Chroma / Cohere / GPU):

  * ``ferber_agent.agent._create`` — the Stage-1 tool-gathering chat call (module function).
  * ``agent._stage``               — every Stage-2 dspy-style chat call (instance method).
  * ``agent._rag``                 — the retriever (``retrieve_reranked`` / ``query``).

Because the stubs are deterministic, the only thing that varies between a serial run
(``workers=1``) and a parallel run (``workers=12``) is the thread-pool fan-out — which is
exactly what the equivalence tests assert is result-preserving.
"""
from __future__ import annotations

import types

import pytest

from ferber_agent import faithful_prompts as fp
from ferber_agent.agent import FerberAgent

# A fixed pool of guideline passages. Each subquery selects an overlapping window from the
# pool (deterministically), so the dedup'd union is non-trivial and order matters.
_POOL = [
    {"source": f"src{i}", "title": f"Guideline {i}",
     "text": f"Passage {i}: clinical guidance number {i}. " * 6}
    for i in range(1, 11)
]


class FakeRag:
    """Deterministic stand-in for RagTool: hits depend only on (query, k, top_n)."""

    def __init__(self):
        self.calls: list[tuple] = []

    def _hits_for(self, query: str, k: int) -> list[dict]:
        start = sum(ord(c) for c in query) % len(_POOL)
        return [_POOL[(start + j) % len(_POOL)] for j in range(k)]

    def retrieve_reranked(self, query: str, k: int, top_n: int, rerank: bool) -> list[dict]:
        self.calls.append((query, k, top_n, rerank))
        return [dict(h) for h in self._hits_for(query, k)[:top_n]]

    def query(self, query: str, k: int | None = None) -> list[dict]:
        return [dict(h) for h in self._hits_for(query, k or 20)]


def _fake_tool_call(call_id: str, name: str, arguments: str):
    fn = types.SimpleNamespace(name=name, arguments=arguments)
    return types.SimpleNamespace(id=call_id, function=fn)


def _make_response(content: str, tool_calls=None):
    msg = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def make_create_stub():
    """Stage-1 stub: first call invokes the pure ``calculate`` tool, then summarises (no calls).

    Deterministic and offline (``calculate`` needs no network), so it exercises the real
    tool-dispatch path without hitting any API."""
    state = {"n": 0}

    def fake_create(model, messages, tools=None, max_tokens=6000, temperature=0.1):
        state["n"] += 1
        if state["n"] == 1 and tools:
            tc = _fake_tool_call("call_1", "calculate",
                                 '{"a": 120, "b": 30, "operator": "/"}')
            return _make_response("Let me compute the progression ratio.", [tc])
        return _make_response("Summary: tools gathered the relevant evidence.", None)

    return fake_create


def make_stage_stub():
    """Stage-2 stub: deterministic text per dspy signature ``doc``.

    The CheckCitationFaithfulness verdict depends ONLY on the statement carried in ``user``
    (mirroring the real per-statement check), so it is identical regardless of execution
    order — the property the equivalence test relies on."""
    def fake_stage(doc, field_name, field_desc, user, max_tokens=2000):
        if doc == fp.SEARCH_DOC:
            return ('["MSI-high colorectal immunotherapy", "KRAS G12C targeted therapy", '
                    '"BRAF V600E encorafenib cetuximab", "anti-EGFR RAS wild-type first line"]')
        if doc == fp.CHECK_CITATION_DOC:
            stmt = user.split("text (")[-1]
            score = sum(ord(c) for c in stmt) % 3
            return ("FALSE — not supported by the cited context" if score == 0
                    else "TRUE — faithful to the cited context")
        if doc == fp.GENCITED_DOC:
            if "flagged as NOT" in user:  # the single revise pass
                return "REVISED ANSWER [1]. Every claim now matches its cited source [2]."
            return ("The tumor is MSI-high [1]. KRAS is wild-type [2]. BRAF V600E is present [3]. "
                    "Pembrolizumab is indicated [4]. Anti-EGFR therapy is appropriate [5].")
        if doc == fp.STRATEGY_DOC:
            return "Strategy: synthesize molecular status, then map to guideline therapy options."
        if doc == fp.REQUIREINPUT_DOC:
            return "Provide histology images for the image-based prediction tool."
        if doc == fp.SUGGESTIONS_DOC:
            return "Utilizing further resources, I can assist if you provide additional imaging."
        return "stub"

    return fake_stage


@pytest.fixture
def make_offline_agent(monkeypatch):
    """Return a factory: ``make(workers=12, citation_selfeval=True) -> FerberAgent`` wired with
    the deterministic offline stubs above."""
    def make(workers: int = 12, citation_selfeval: bool = True) -> FerberAgent:
        monkeypatch.setattr("ferber_agent.agent._create", make_create_stub())
        agent = FerberAgent(
            chroma_dir="/tmp/unused_index", llm_model="gpt-5.1",
            faithful=True, n_subqueries=12, retrieve_k=40, rerank_top_n=10,
            citation_selfeval=citation_selfeval,
        )
        agent._rag = FakeRag()
        agent._stage = make_stage_stub()  # type: ignore[assignment]
        agent.rag_workers = workers
        agent.citation_workers = workers
        return agent

    return make
