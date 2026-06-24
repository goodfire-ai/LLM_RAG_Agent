"""Deterministic serial-vs-parallel equivalence tests for faithful mode.

Faithful mode fans out two independent workloads across a thread pool: the per-statement
citation-faithfulness checks and the per-subquery retrievals. This is only a sound speedup if
it is *result-preserving* — the parallel run must produce byte-identical output to the serial
run. These tests prove that with deterministic stubs (no OpenAI / Chroma / Cohere / GPU), so
they run in CI. They are the hermetic port of the experiment's ``equiv_check.py`` gate.
"""
from __future__ import annotations

from ferber_agent.agent import FerberAgent

CITED = (
    "The tumor is MSI-high [1]. KRAS is wild-type [2]. BRAF V600E is present [3]. "
    "Pembrolizumab is indicated for MSI-high disease [4]. Anti-EGFR therapy is "
    "appropriate given RAS wild-type status [5]. The lesion progressed on imaging [6]. "
    "First-line chemotherapy should be FOLFOX [7]. Consider a clinical trial [8]."
)


def _retrieved(n: int = 8) -> list[dict]:
    return [{"source": f"src{i}", "title": f"Guideline {i}",
             "text": f"Passage {i}: clinical guidance text number {i} " * 8}
            for i in range(1, n + 1)]


def _passage_keys(hits: list[dict]) -> list[tuple]:
    return [(h.get("source", ""), h.get("title", ""), (h.get("text", "") or "")[:200])
            for h in hits]


def test_map_parallel_preserves_input_order():
    # workers > 1 must return results in INPUT order, identical to the serial path.
    items = list(range(50))
    def fn(x):
        return x * x
    serial = FerberAgent._map_parallel(fn, items, workers=1)
    parallel = FerberAgent._map_parallel(fn, items, workers=12)
    assert serial == [x * x for x in items]
    assert parallel == serial


def test_map_parallel_empty():
    assert FerberAgent._map_parallel(lambda x: x, [], workers=8) == []


def test_citation_selfeval_parallel_equals_serial(make_offline_agent):
    # The deterministic per-statement verdict depends only on each statement's text, so the
    # parallel fan-out (workers=12) must match the serial path (workers=1) every time.
    retrieved = _retrieved()
    cited_user = "strategy: x\ncontext: y\npatient: z\nquestion: q"

    serial_agent = make_offline_agent(workers=1)
    serial_out, serial_rec = serial_agent._citation_selfeval(CITED, retrieved, cited_user)
    # the stub flags at least one statement, so the revise pass is genuinely exercised
    assert serial_rec["checked"] >= 1

    for _ in range(8):  # repeat to rule out a race
        par_agent = make_offline_agent(workers=12)
        out, rec = par_agent._citation_selfeval(CITED, retrieved, cited_user)
        assert out == serial_out
        assert rec == serial_rec


def test_faithful_pipeline_parallel_equals_serial(make_offline_agent):
    # End-to-end: the full faithful answer (subquery fan-out + retrieval dedup + citation
    # self-eval) must be byte-identical between serial and parallel execution.
    ctx = ("65-year-old with metastatic colorectal adenocarcinoma. Molecular: BRAF V600E "
           "mutation; microsatellite instability-high.")
    q = "What targeted and immunotherapy options are supported by guidelines?"

    serial = make_offline_agent(workers=1).answer(ctx, q)
    parallel = make_offline_agent(workers=12).answer(ctx, q)

    assert serial.answer_text == parallel.answer_text
    assert _passage_keys(serial.retrieved) == _passage_keys(parallel.retrieved)
    assert serial.citations == parallel.citations
    # dedup actually removed overlap across the subquery windows
    assert len(serial.retrieved) == len(set(_passage_keys(serial.retrieved)))
