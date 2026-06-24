"""Extract the Ferber agent's VERBATIM prompt strings from the original dspy source.

Faithful mode reproduces the paper's pipeline with prompts copied character-for-character
from the original code (vendored in this repo under ``RAGent/DSPY``). To eliminate any
hand-copy drift, the long prompt blocks are not retyped: this script parses the three
original source files with ``ast`` and emits their exact string values into
``ferber_agent/faithful_prompts.py``. ``tests/test_prompt_fidelity.py`` re-runs this same
extraction and asserts byte-equality against the generated module.

Sources (under ``RAGent/DSPY``):
  signatures.py   : Search / AnswerStrategy / RequireInput / Suggestions /
                    GenerateCitedResponse / CheckCitationFaithfulness
                    (class docstring = instruction; OutputField.desc = output spec)
  med_agent.py    : AGENT_SYSTEM_PROMPT, the chat() exhaustive-tool instruction, the
                    "MUST use ALL tools" continue-nudge
  agent_tools.py  : tool docstrings (onco_kb / query_pubmed / calculate /
                    gen_radiology_report / segment_image / check_mutations)

Usage:  python scripts/extract_prompts.py [--dspy-src DIR] [--out FILE] [--print]
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

# repo root = parent of this scripts/ directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DSPY_SRC = _REPO_ROOT / "RAGent" / "DSPY"
_DEFAULT_OUT = _REPO_ROOT / "ferber_agent" / "faithful_prompts.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_docstring(mod: ast.Module, name: str) -> str:
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            doc = ast.get_docstring(node, clean=False)
            if doc is None:
                raise ValueError(f"class {name} has no docstring")
            return doc
    raise KeyError(f"class {name} not found")


def _class_dunder_doc(mod: ast.Module, name: str) -> str:
    """Return the verbatim ``__doc__ = ...`` assignment value of a class.

    Some signatures (e.g. CheckCitationFaithfulness) set the instruction via an explicit
    ``__doc__ = f"..."`` assignment rather than a bare docstring literal, so ast.get_docstring
    misses it. Handles both a plain str Constant and a no-interpolation f-string (JoinedStr with
    a single Constant)."""
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == "__doc__"):
                    v = stmt.value
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        return v.value
                    if (isinstance(v, ast.JoinedStr) and len(v.values) == 1
                            and isinstance(v.values[0], ast.Constant)):
                        return v.values[0].value
                    raise ValueError(f"class {name} __doc__ is not a plain string literal")
            raise KeyError(f"class {name} has no __doc__ assignment")
    raise KeyError(f"class {name} not found")


def _func_docstring(mod: ast.Module, name: str) -> str:
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            doc = ast.get_docstring(node, clean=False)
            if doc is None:
                raise ValueError(f"function {name} has no docstring")
            return doc
    raise KeyError(f"function {name} not found")


def _field_desc(mod: ast.Module, class_name: str, field_name: str) -> str:
    """Return the verbatim ``desc=`` string of a dspy Input/OutputField assignment."""
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == field_name
                        and isinstance(stmt.value, ast.Call)):
                    for kw in stmt.value.keywords:
                        if kw.arg == "desc" and isinstance(kw.value, ast.Constant):
                            return kw.value.value
            raise KeyError(f"{class_name}.{field_name} has no desc= string")
    raise KeyError(f"class {class_name} not found")


def _module_assign_str(mod: ast.Module, name: str) -> str:
    for node in mod.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
                and isinstance(node.value, ast.Constant)):
            return node.value.value
    raise KeyError(f"module-level str assignment {name} not found")


def _is_instruction_target(stmt) -> bool:
    if isinstance(stmt, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == "instruction" for t in stmt.targets)
    if isinstance(stmt, ast.AugAssign):
        return isinstance(stmt.target, ast.Name) and stmt.target.id == "instruction"
    return False


def _collect_str_parts(node) -> list[str]:
    """Flatten a (possibly parenthesised) string-concatenation expression into its literal
    pieces, replacing the ``question`` Name reference with the ``{question}`` placeholder."""
    out: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    elif isinstance(node, ast.Name) and node.id == "question":
        out.append("{question}")
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        out.extend(_collect_str_parts(node.left))
        out.extend(_collect_str_parts(node.right))
    return out


def _chat_ext_instruction(mod: ast.Module) -> str:
    """Reconstruct the verbatim chat() exhaustive-tool instruction template.

    In med_agent.py the instruction is built by ``instruction = (<lit> + question)`` followed by
    a run of ``instruction += <lit>``. We collect those string literals in source order and join
    them, substituting ``{question}`` for the one ``question`` variable reference, so the static
    template is captured byte-for-byte.
    """
    func = None
    for node in ast.walk(mod):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assigns = [s for s in ast.walk(node)
                       if isinstance(s, (ast.Assign, ast.AugAssign))]
            if any(_is_instruction_target(s) for s in assigns):
                func = node
                break
    if func is None:
        raise KeyError("function with instruction assignments not found")

    parts: list[str] = []
    for stmt in func.body:
        if isinstance(stmt, ast.Assign) and _is_instruction_target(stmt):
            parts.extend(_collect_str_parts(stmt.value))
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name) \
                and stmt.target.id == "instruction":
            parts.extend(_collect_str_parts(stmt.value))
    return "".join(parts)


def _find_const_str_startswith(mod: ast.Module, prefix: str) -> str:
    for node in ast.walk(mod):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.startswith(prefix):
            return node.value
    raise KeyError(f"no string constant starting with {prefix!r}")


def extract_all(dspy_src: Path) -> dict[str, str]:
    """Extract every verbatim prompt block from the dspy source directory."""
    sig = _parse(dspy_src / "signatures.py")
    med = _parse(dspy_src / "med_agent.py")
    tools = _parse(dspy_src / "agent_tools.py")

    out: dict[str, str] = {}
    # --- signatures.py: docstrings (instructions) ---
    out["SEARCH_DOC"] = _class_docstring(sig, "Search")
    out["SEARCH_SEARCHES_DESC"] = _field_desc(sig, "Search", "searches")
    out["STRATEGY_DOC"] = _class_docstring(sig, "AnswerStrategy")
    out["STRATEGY_RESPONSE_DESC"] = _field_desc(sig, "AnswerStrategy", "response")
    out["REQUIREINPUT_DOC"] = _class_docstring(sig, "RequireInput")
    out["REQUIREINPUT_RESPONSE_DESC"] = _field_desc(sig, "RequireInput", "response")
    out["SUGGESTIONS_DOC"] = _class_docstring(sig, "Suggestions")
    out["SUGGESTIONS_SUGGESTIONS_DESC"] = _field_desc(sig, "Suggestions", "suggestions")
    out["GENCITED_DOC"] = _class_docstring(sig, "GenerateCitedResponse")
    out["GENCITED_RESPONSE_DESC"] = _field_desc(sig, "GenerateCitedResponse", "response")
    out["GENCITED_CONTEXT_DESC"] = _field_desc(sig, "GenerateCitedResponse", "context")
    out["GENCITED_PATIENT_DESC"] = _field_desc(sig, "GenerateCitedResponse", "patient")
    out["GENCITED_TOOLRESULTS_DESC"] = _field_desc(sig, "GenerateCitedResponse", "tool_results")

    # --- citation self-evaluation (paper's single-iteration faithfulness check) ---
    out["CHECK_CITATION_DOC"] = _class_dunder_doc(sig, "CheckCitationFaithfulness")
    out["CHECK_CITATION_CONTEXT_DESC"] = _field_desc(sig, "CheckCitationFaithfulness", "context")
    out["CHECK_CITATION_TEXT_DESC"] = _field_desc(sig, "CheckCitationFaithfulness", "text")
    out["CHECK_CITATION_FAITHFULNESS_DESC"] = _field_desc(
        sig, "CheckCitationFaithfulness", "faithfulness")

    # --- med_agent.py: agent prompts ---
    out["AGENT_SYSTEM_PROMPT"] = _module_assign_str(med, "AGENT_SYSTEM_PROMPT")
    out["CHAT_EXT_INSTRUCTION"] = _chat_ext_instruction(med)
    out["MUST_USE_ALL_TOOLS_NUDGE"] = _find_const_str_startswith(
        med, "Check again if you have used all available tools")

    # --- agent_tools.py: tool docstrings ---
    out["TOOL_ONCOKB_DOC"] = _func_docstring(tools, "onco_kb")
    out["TOOL_PUBMED_DOC"] = _func_docstring(tools, "query_pubmed")
    out["TOOL_CALCULATE_DOC"] = _func_docstring(tools, "calculate")
    out["TOOL_RADIOLOGY_DOC"] = _func_docstring(tools, "gen_radiology_report")
    out["TOOL_SEGMENT_DOC"] = _func_docstring(tools, "segment_image")
    out["TOOL_CHECKMUTATIONS_DOC"] = _func_docstring(tools, "check_mutations")
    return out


_HEADER = '''\
"""VERBATIM Ferber prompt strings — GENERATED by scripts/extract_prompts.py. DO NOT EDIT BY HAND.

Each constant is the exact string value of a docstring / dspy field ``desc`` / instruction
literal in the original source (RAGent/DSPY/{signatures,med_agent,agent_tools}.py). The
prompt-fidelity test (tests/test_prompt_fidelity.py) re-extracts from that source and asserts
byte-equality against these constants, so any drift here fails the test. Regenerate with::

    python scripts/extract_prompts.py
"""
# fmt: off
# ruff: noqa
'''


def render_module(consts: dict[str, str]) -> str:
    lines = [_HEADER, ""]
    for k in consts:  # preserve insertion order
        lines.append(f"{k} = {consts[k]!r}")
    lines.append("")
    lines.append("ALL_VERBATIM_BLOCKS = {")
    for k in consts:
        lines.append(f"    {k!r}: {k},")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dspy-src", default=str(_DEFAULT_DSPY_SRC),
                    help="directory holding the original dspy source (default: RAGent/DSPY)")
    ap.add_argument("--out", default=str(_DEFAULT_OUT),
                    help="output module path (default: ferber_agent/faithful_prompts.py)")
    ap.add_argument("--print", action="store_true", help="print a summary instead of writing")
    args = ap.parse_args()

    consts = extract_all(Path(args.dspy_src))
    if args.print:
        for k, v in consts.items():
            print(f"=== {k} ({len(v)} chars) ===")
            print(v[:200].replace("\n", "\\n") + ("..." if len(v) > 200 else ""))
            print()
        return
    Path(args.out).write_text(render_module(consts), encoding="utf-8")
    print(f"wrote {args.out} with {len(consts)} verbatim blocks "
          f"({sum(len(v) for v in consts.values())} total chars)")


if __name__ == "__main__":
    main()
