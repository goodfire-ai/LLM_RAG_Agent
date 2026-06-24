"""Hermetic end-to-end check that faithful mode executes every stage of the paper's pipeline.

This is the offline reproduction check for the export: on a single stubbed sample, the faithful
pipeline must run all stages end-to-end — Stage-1 tool gathering and dispatch, Stage-2 subquery
generation, retrieval, AnswerStrategy, GenerateCitedResponse, the citation self-evaluation, and
Suggestions — and the histology tool must return a replayed prediction. The full ferber20
numeric result (faithful vs bare completeness) needs the live corpus/keys/GPU and is the source
experiment's job; this asserts the *machinery*, deterministically, in CI.
"""
from __future__ import annotations

import json

from ferber_agent import tools


def test_faithful_runs_all_stages(make_offline_agent):
    agent = make_offline_agent(workers=12, citation_selfeval=True)
    ctx = ("58-year-old with metastatic colorectal cancer. Molecular: KRAS wild-type, "
           "BRAF V600E, MSI-high.")
    q = "Which guideline-supported therapies apply?"

    res = agent.answer(ctx, q, case_key="Adams")

    # Stage 1: the tool-gathering loop dispatched the (pure) calculate tool.
    assert any(c["tool"] == "calculate" for c in res.tool_calls)

    # Stage 2 grounding bookkeeping: subqueries generated + retrieval non-empty.
    rag_rec = next(c for c in res.tool_calls if c["tool"] == "rag")
    assert rag_rec["args"]["n_subqueries"] > 1
    assert len(rag_rec["args"]["subqueries"]) > 1
    assert len(res.retrieved) >= 1
    assert res.citations and all("source" in c for c in res.citations)

    # Citation self-evaluation fired and checked at least one statement.
    sev = rag_rec["args"]["citation_selfeval"]
    assert sev["enabled"] is True
    assert sev["checked"] >= 1

    # GenerateCitedResponse + Suggestions both contributed to the final answer.
    assert "[1]" in res.answer_text
    assert "provide additional imaging" in res.answer_text.lower()


def test_faithful_selfeval_can_be_disabled(make_offline_agent):
    agent = make_offline_agent(workers=4, citation_selfeval=False)
    res = agent.answer("patient context", "question?")
    rag_rec = next(c for c in res.tool_calls if c["tool"] == "rag")
    assert rag_rec["args"]["citation_selfeval"]["enabled"] is False


def test_histology_replay_returns_bundled_prediction():
    # The packaged ferber20 lookup ships with the agent, so the replay tool returns a
    # prediction out of the box without any HISTOLOGY_LOOKUP override.
    tools._histology_lookup.cache_clear()
    out = tools.histology_replay("Adams")
    assert "MSI" in out and ("prediction" in out.lower())
    # A surname with no documented prediction yields an explicit gap message, never a fabrication.
    gap = tools.histology_replay("NoSuchPatient12345")
    assert "no histology-based genetic prediction is documented" in gap.lower()


def test_histology_replay_env_override(tmp_path, monkeypatch):
    lut = {"cases": {"Doe": {"available": True, "predictions": {
        "KRAS": {"label": "mutated", "probability": 0.91}}, "source": "test"}}}
    p = tmp_path / "lookup.json"
    p.write_text(json.dumps(lut))
    monkeypatch.setenv("HISTOLOGY_LOOKUP", str(p))
    tools._histology_lookup.cache_clear()
    try:
        out = tools.histology_replay("Doe", targets=["KRAS"])
        assert "KRAS" in out and "mutated" in out
        assert "0.91" in out
    finally:
        tools._histology_lookup.cache_clear()
