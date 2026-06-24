"""Offline tests for the pluggable retrieval-engine switch.

``make_engine`` selection (canonical names + deprecated aliases + errors), the Chroma engine's
delegation to the cached ``retrieve_reranked`` path, and both OpenAI file_search engines are
exercised with stubbed clients — no network, no Chroma, no vector store.
"""
from __future__ import annotations

import types

import pytest

from ferber_agent import retrieval
from ferber_agent.retrieval import (
    ChromaEngine, OpenAIFileSearchChatEngine, OpenAIFileSearchResponsesEngine, make_engine,
)
from ferber_agent.usage import UsageAccumulator


class FakeRag:
    """Deterministic RagTool stand-in exposing the cached retrieve_reranked contract."""

    def __init__(self):
        self.calls: list[tuple] = []

    def retrieve_reranked(self, query, k, top_n, rerank):
        self.calls.append((query, k, top_n, rerank))
        return [{"text": f"passage {i} for {query}", "title": f"T{i}", "source": "asco",
                 "distance": 0.1 * i} for i in range(top_n)]


# --- make_engine selection ---------------------------------------------------
def test_make_engine_selects_canonical_names():
    assert isinstance(make_engine("chroma_cosine", rag=FakeRag()), ChromaEngine)
    assert isinstance(make_engine("chroma_cohere", rag=FakeRag()), ChromaEngine)
    assert isinstance(
        make_engine("openai_filesearch_responses", vector_store_id="vs_1"),
        OpenAIFileSearchResponsesEngine)
    assert isinstance(
        make_engine("openai_filesearch_chat", vector_store_id="vs_1"),
        OpenAIFileSearchChatEngine)


def test_make_engine_accepts_deprecated_aliases():
    assert make_engine("openai_fs_responses", vector_store_id="vs_1").name == \
        "openai_filesearch_responses"
    assert make_engine("openai_fs_chat", vector_store_id="vs_1").name == "openai_filesearch_chat"


def test_make_engine_errors():
    with pytest.raises(ValueError):
        make_engine("nope")
    with pytest.raises(ValueError):  # file_search needs a vector store id
        make_engine("openai_filesearch_responses")


# --- chroma engine: delegates to retrieve_reranked, normalizes passages ------
def test_chroma_engine_delegates_and_normalizes():
    rag = FakeRag()
    eng = make_engine("chroma_cosine", rag=rag)
    usage = UsageAccumulator()
    out = eng.retrieve("BRAF therapy", retrieve_k=20, top_n=3, usage=usage)
    assert rag.calls == [("BRAF therapy", 20, 3, False)]  # cosine => rerank False
    assert len(out) == 3
    # normalized passage schema
    assert set(out[0]) == {"text", "title", "source", "score", "engine", "chunk_chars"}
    assert out[0]["engine"] == "chroma_cosine"
    assert out[0]["chunk_chars"] == len(out[0]["text"])
    assert usage.retrieval_calls == 1 and usage.rerank_calls == 0


def test_chroma_cohere_sets_rerank_when_key_present(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "x")
    rag = FakeRag()
    eng = make_engine("chroma_cohere", rag=rag)
    usage = UsageAccumulator()
    eng.retrieve("q", retrieve_k=10, top_n=2, usage=usage)
    assert rag.calls[0][3] is True  # rerank requested
    assert usage.rerank_calls == 1


# --- OpenAI file_search engines: stubbed clients -----------------------------
def _fs_result(text, source, title, score):
    return types.SimpleNamespace(text=text, filename=f"{source}__id.txt",
                                 attributes={"source": source, "title": title}, score=score)


def test_filesearch_responses_engine(monkeypatch):
    results = [_fs_result("guideline text A", "esmo", "ESMO CRC", 0.9),
               _fs_result("guideline text B", "asco", "ASCO CRC", 0.8)]
    fake_resp = types.SimpleNamespace(
        output=[types.SimpleNamespace(type="file_search_call", results=results)],
        usage=None, output_text="")
    fake_client = types.SimpleNamespace(
        responses=types.SimpleNamespace(create=lambda **kw: fake_resp))
    monkeypatch.setattr(retrieval, "_oai_client", lambda: fake_client)

    eng = make_engine("openai_filesearch_responses", vector_store_id="vs_1")
    usage = UsageAccumulator()
    out = eng.retrieve("colorectal guidelines", retrieve_k=10, top_n=5, usage=usage)
    assert [p["source"] for p in out] == ["esmo", "asco"]
    assert [p["title"] for p in out] == ["ESMO CRC", "ASCO CRC"]
    assert all(p["engine"] == "openai_filesearch_responses" for p in out)
    assert usage.filesearch_calls == 1 and usage.retrieval_calls == 1


def test_filesearch_chat_engine(monkeypatch):
    tc = types.SimpleNamespace(function=types.SimpleNamespace(arguments='{"query": "crc"}'))
    chat_resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(tool_calls=[tc]))],
        usage=None)
    search_res = types.SimpleNamespace(data=[_fs_result("vstore text", "meditron", "Med", 0.7)])
    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kw: chat_resp)),
        vector_stores=types.SimpleNamespace(search=lambda **kw: search_res))
    monkeypatch.setattr(retrieval, "_oai_client", lambda: fake_client)

    eng = make_engine("openai_filesearch_chat", vector_store_id="vs_1")
    usage = UsageAccumulator()
    out = eng.retrieve("colorectal guidelines", retrieve_k=10, top_n=5, usage=usage)
    assert len(out) == 1 and out[0]["source"] == "meditron"
    assert out[0]["engine"] == "openai_filesearch_chat"
    assert usage.vstore_searches == 1 and usage.retrieval_calls == 1
