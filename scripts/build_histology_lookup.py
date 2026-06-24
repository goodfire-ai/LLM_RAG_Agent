"""Build the per-case histology replay lookup from a paper supplementary.

The Ferber agent's ``check_mutations`` tool ran proprietary H&E image classifiers that were
never released; the paper instead pre-extracted those MSI/KRAS/BRAF predictions. Faithful mode
*replays* them via ``ferber_agent.tools.histology_replay``, which reads a lookup JSON. This
script (re)builds that lookup: for each named case it asks an LLM to report ONLY the
image-based prediction-tool result (MSI / KRAS / BRAF, with probability when stated), sourced
from the authoritative supplementary text, and marks a genuine gap when the tool was not run.

Output JSON (the schema ``histology_replay`` consumes), keyed by case surname::

    {"dataset": "...", "n_cases": N, "model": "...", "source": "...",
     "cases": {"<surname>": {"available": bool,
                             "predictions": {"MSI":  {"label": "MSI-High|MSS", "probability": <num|null>},
                                             "KRAS": {"label": "mutated|wild-type", "probability": <num|null>},
                                             "BRAF": {"label": "mutated|wild-type", "probability": <num|null>}},
                             "source": "<one sentence: provenance, or why absent>"}}}

The bundled ``ferber_agent/data/histology_lookup.json`` was produced this way from the public
Ferber et al. supplementary. The in-house classifiers themselves are NOT reproducible — this
replays the paper's documented predictions and never fabricates one.

Requires ANTHROPIC_API_KEY (uses the Claude SDK). Usage::

    python scripts/build_histology_lookup.py \
        --supplementary ferber_supp.txt \
        --surnames Adams Smith Garcia ... \
        --out ferber_agent/data/histology_lookup.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-4-6"

_SYS = (
    "You extract structured genetic-test predictions from the supplementary of a molecular tumor "
    "board paper. The agent used a 'check mutations' / genetic-modeling TOOL that predicts MSI, "
    "KRAS, and BRAF status from histopathology images, sometimes with a probability. In the "
    "supplementary, the agent's model responses cite each such tool prediction inline, usually "
    "tagged with a [Tool] marker (e.g. 'the checkmutations tool predicting wild-type BRAF with a "
    "probability of 0.48 [Tool]', 'MSI-High status with a probability of 0.95'). "
    "Your job: report ONLY what that image-based prediction tool reported for the named patient. "
    "Do NOT report a marker whose status came from a molecular/genomic report, a liver/tumor "
    "biopsy, OncoKB, or the patient's own history/recollection (those are tagged [Patient] or "
    "described as molecular analysis, not [Tool]). If the tool errored, was not run, or the agent "
    "only ASKED for images to run it, treat all markers as absent. Never invent a probability that "
    "is not explicitly stated; use null when no probability is given. Normalize labels to: MSI -> "
    "'MSI-High' or 'MSS'; KRAS and BRAF -> 'mutated' or 'wild-type'. Return ONLY a JSON object."
)

_SCHEMA = (
    'Return JSON exactly: {"available": <true if the image prediction tool reported at least one '
    'marker for THIS patient else false>, "predictions": {"MSI": {"label": "MSI-High|MSS", '
    '"probability": <number|null>}, "KRAS": {"label": "mutated|wild-type", "probability": '
    '<number|null>}, "BRAF": {"label": "mutated|wild-type", "probability": <number|null>}}, '
    '"source": "<one sentence: where each reported marker came from, or why absent>"}. '
    "Omit any marker the prediction tool did not report (do not include it in predictions)."
)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = text[: text.rfind("```")] if "```" in text else text
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def extract_case(client, surname: str, supp: str, model: str) -> dict:
    """Ask the LLM for the image-tool MSI/KRAS/BRAF prediction for one case."""
    user = (
        f"Target patient surname: {surname}\n"
        f"(The supplementary may refer to them as 'Mr./Mrs./Ms. {surname}' or '{surname}, "
        f"<first name>'. Use only that patient's section / model responses.)\n\n"
        f"=== FULL SUPPLEMENTARY TEXT ===\n{supp}\n=== END ===\n\n{_SCHEMA}"
    )
    msg = client.messages.create(
        model=model, max_tokens=1200, system=_SYS,
        messages=[{"role": "user", "content": user}])
    raw = msg.content[0].text
    try:
        rec = _extract_json(raw)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "predictions": {}, "source": f"parse_error: {e}"}
    preds = {k: v for k, v in (rec.get("predictions") or {}).items()
             if isinstance(v, dict) and v.get("label")}
    rec["predictions"] = preds
    rec["available"] = bool(preds) and rec.get("available", True)
    return rec


def build_lookup(supplementary: Path, surnames: list[str], model: str = DEFAULT_MODEL,
                 dataset: str = "custom", workers: int = 6) -> dict:
    """Build the full lookup dict for ``surnames`` from the supplementary text."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    supp = supplementary.read_text(errors="replace")

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_case, client, s, supp, model): s for s in surnames}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
    out = {s: out[s] for s in surnames}  # stable order = input order
    n_avail = sum(1 for r in out.values() if r["available"])
    return {"dataset": dataset, "n_cases": len(out), "model": model,
            "source": supplementary.name, "cases": out, "n_available": n_avail}


def _load_surnames(args) -> list[str]:
    if args.surnames:
        return list(args.surnames)
    cases = json.loads(Path(args.cases).read_text())
    rows = cases["cases"] if isinstance(cases, dict) and "cases" in cases else cases
    return [c["surname"] for c in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--supplementary", required=True, type=Path,
                    help="path to the supplementary text file")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--surnames", nargs="+", help="case surnames to extract")
    src.add_argument("--cases", help="JSON file with a list/`cases` of {surname: ...} objects")
    ap.add_argument("--out", required=True, type=Path, help="output lookup JSON path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default="custom")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    surnames = _load_surnames(args)
    print(f"{len(surnames)} cases; supplementary={args.supplementary}")
    lookup = build_lookup(args.supplementary, surnames, model=args.model,
                          dataset=args.dataset, workers=args.workers)

    for s in surnames:
        rec = lookup["cases"][s]
        markers = ", ".join(f"{k}={v.get('label')}@{v.get('probability')}"
                            for k, v in rec["predictions"].items())
        flag = "" if rec["available"] else "  (no tool prediction)"
        print(f"{s:12s} {markers}{flag}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lookup, indent=2))
    print(f"\nwrote {args.out}: {lookup['n_available']}/{lookup['n_cases']} cases "
          f"have a documented tool prediction")


if __name__ == "__main__":
    main()
