"""Build (or reuse) the OpenAI vector store backing the file_search retrieval engines.

The ``openai_filesearch_responses`` / ``openai_filesearch_chat`` retrieval engines query an
OpenAI vector store of the guideline corpus instead of the local Chroma index. This script
uploads the corpus to a vector store — one file per document, with ``{source, title, doc_id}``
attributes so retrieved chunks carry their provenance — and records the store id so the agent
can be pointed at it (``--vector-store-id`` / ``OPENAI_VECTOR_STORE_ID``). OpenAI does its own
chunking / embedding / retrieval; that is exactly the "hand the docs to OpenAI's hosted RAG"
comparison from experiment #24.

The corpus is a directory of ``{source}.jsonl`` files (one JSON object per line) with at least
a ``clean_text`` field, plus optional ``id`` / ``unique_article_uuid`` and ``title`` — the same
format the Chroma index is built from. Idempotent: an already-fully-indexed recorded store is
reused unless ``--force``.

Usage (pure network I/O; no GPU)::

    python scripts/build_vector_store.py --corpus-dir /path/to/corpus --out vector_store.json
    # then: FerberAgent(..., retrieval_engine="openai_filesearch_responses",
    #                    vector_store_id=<recorded id>)

Requires ``OPENAI_API_KEY``.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Default corpus sources (the five-source oncology guideline corpus). Override with --sources.
DEFAULT_SOURCES = ("asco", "esmo", "meditron", "onkopedia_de", "onkopedia_en")


def load_docs(corpus_dir: str, sources: tuple[str, ...]) -> list[dict]:
    """Load the corpus documents from ``{corpus_dir}/{source}.jsonl`` for each source.

    Each record needs a ``clean_text`` field; ``id``/``unique_article_uuid`` and ``title`` are
    optional. Returns a list of ``{source, doc_id, title, text}`` dicts."""
    docs: list[dict] = []
    for src in sources:
        path = Path(corpus_dir) / f"{src}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"corpus file not found: {path}")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                docs.append({
                    "source": src,
                    "doc_id": (d.get("id") or d.get("unique_article_uuid") or "")[:32],
                    "title": (d.get("title") or "")[:240],
                    "text": d.get("clean_text") or "",
                })
    return docs


def build_vector_store(corpus_dir: str, out_path: str, *, name: str,
                       sources: tuple[str, ...] = DEFAULT_SOURCES, workers: int = 16,
                       force: bool = False, client=None) -> dict:
    """Upload the corpus to an OpenAI vector store and record the result to ``out_path``.

    Returns the recorded dict (vector_store_id, status, file counts, composition). If a recorded
    store at ``out_path`` is already fully indexed and ``force`` is False, it is reused and its
    record returned unchanged. ``client`` is an ``openai.OpenAI`` instance (constructed if None).
    """
    if client is None:
        from openai import OpenAI

        client = OpenAI(timeout=180.0, max_retries=5)
    out = Path(out_path)

    # Idempotency: reuse a fully-indexed recorded store unless --force.
    if out.exists() and not force:
        rec = json.loads(out.read_text())
        vs_id = rec.get("vector_store_id")
        try:
            st = client.vector_stores.retrieve(vs_id)
            if st.status == "completed" and st.file_counts.completed == rec.get("n_files"):
                print(f"[vs] reuse existing store {vs_id} "
                      f"({st.file_counts.completed} files indexed)", flush=True)
                return rec
            print(f"[vs] recorded store {vs_id} not fully indexed "
                  f"({st.file_counts.completed}/{rec.get('n_files')}); rebuilding", flush=True)
        except Exception as e:  # noqa: BLE001 — a missing/expired store just triggers a rebuild
            print(f"[vs] recorded store unusable ({e!r}); rebuilding", flush=True)

    docs = load_docs(corpus_dir, sources)
    total_bytes = sum(len(d["text"].encode("utf-8")) for d in docs)
    per_source = {s: sum(1 for d in docs if d["source"] == s) for s in sources}
    print(f"[vs] {len(docs)} docs, {total_bytes/1e6:.1f} MB, per-source={per_source}", flush=True)

    vs = client.vector_stores.create(name=name)
    print(f"[vs] created vector store {vs.id}", flush=True)

    def upload(d: dict) -> str:
        fname = f"{d['source']}__{d['doc_id'] or 'na'}.txt"
        bio = io.BytesIO(d["text"].encode("utf-8"))
        bio.name = fname
        f = client.files.create(file=bio, purpose="assistants")
        client.vector_stores.files.create(
            vector_store_id=vs.id, file_id=f.id,
            attributes={"source": d["source"], "title": d["title"], "doc_id": d["doc_id"]})
        return f.id

    file_ids: list[str] = []
    errors: list[tuple] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(upload, d): d for d in docs}
        for i, fut in enumerate(as_completed(futs), 1):
            d = futs[fut]
            try:
                file_ids.append(fut.result())
            except Exception as e:  # noqa: BLE001 — record per-file upload failures, keep going
                errors.append((d["source"], d["doc_id"], repr(e)))
            if i % 50 == 0 or i == len(docs):
                print(f"[vs] uploaded {i}/{len(docs)} ({time.time()-t0:.0f}s, "
                      f"{len(errors)} errors)", flush=True)

    print(f"[vs] waiting for indexing of {len(file_ids)} files...", flush=True)
    for _ in range(300):  # up to ~10 min
        st = client.vector_stores.retrieve(vs.id)
        fc = st.file_counts
        if fc.in_progress == 0 and (fc.completed + fc.failed) >= len(file_ids):
            break
        time.sleep(2)
    st = client.vector_stores.retrieve(vs.id)
    fc = st.file_counts
    print(f"[vs] status={st.status} completed={fc.completed} failed={fc.failed} "
          f"total={fc.total}", flush=True)

    rec = {
        "vector_store_id": vs.id,
        "name": name,
        "status": st.status,
        "n_files": len(file_ids),
        "completed": fc.completed,
        "failed": fc.failed,
        "total_bytes": total_bytes,
        "per_source": per_source,
        "embed": "openai-hosted (file_search default chunking + embedding)",
        "source_corpus": corpus_dir,
        "upload_errors": errors[:20],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"[vs] recorded -> {out}", flush=True)
    if fc.failed or errors:
        print(f"[vs] WARNING: {fc.failed} indexing failures, {len(errors)} upload errors",
              flush=True)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", required=True,
                    help="directory of {source}.jsonl corpus files (need a clean_text field)")
    ap.add_argument("--out", default="vector_store.json",
                    help="path to write the recorded store id + composition (default: ./vector_store.json)")
    ap.add_argument("--name", default="ferber-oncology-guidelines",
                    help="vector store display name")
    ap.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES),
                    help=f"corpus source stems to upload (default: {' '.join(DEFAULT_SOURCES)})")
    ap.add_argument("--workers", type=int, default=16, help="parallel upload workers")
    ap.add_argument("--force", action="store_true", help="rebuild even if a recorded store exists")
    args = ap.parse_args(argv)

    rec = build_vector_store(args.corpus_dir, args.out, name=args.name,
                             sources=tuple(args.sources), workers=args.workers, force=args.force)
    print(f"VS_BUILD_OK {rec['vector_store_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
