"""Byte-fidelity test: the faithful prompts must match the vendored dspy source exactly.

Faithful mode's whole claim is that it sends the paper's prompts *verbatim*. This test re-runs
``scripts/extract_prompts.py`` against the original dspy source vendored in this repo
(``RAGent/DSPY``) and asserts every extracted block is byte-identical to the corresponding
constant baked into ``ferber_agent/faithful_prompts.py``. Any hand-edit or upstream drift fails
here. Hermetic: pure ``ast`` parsing, no network.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DSPY_SRC = _REPO_ROOT / "RAGent" / "DSPY"


def _load_extract_prompts():
    path = _REPO_ROOT / "scripts" / "extract_prompts.py"
    spec = importlib.util.spec_from_file_location("extract_prompts", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extract_prompts"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_faithful_prompts_byte_identical_to_source():
    ep = _load_extract_prompts()
    from ferber_agent import faithful_prompts as fp

    extracted = ep.extract_all(_DSPY_SRC)
    baked = fp.ALL_VERBATIM_BLOCKS

    # same set of blocks
    assert set(extracted) == set(baked), (
        f"block set differs: only-in-source={set(extracted) - set(baked)}, "
        f"only-in-module={set(baked) - set(extracted)}")

    drift = [k for k in extracted if extracted[k] != baked[k]]
    assert not drift, f"verbatim drift in blocks: {drift}"


def test_all_blocks_referenced_in_package():
    # Every verbatim block must actually be used by the agent/tools (no dead constants).
    from ferber_agent import faithful_prompts as fp

    src = ((_REPO_ROOT / "ferber_agent" / "agent.py").read_text()
           + (_REPO_ROOT / "ferber_agent" / "tools.py").read_text())
    unused = [k for k in fp.ALL_VERBATIM_BLOCKS if f".{k}" not in src and k not in src]
    assert not unused, f"verbatim blocks never referenced: {unused}"
