"""Offline tests for the execution-backend switch (chat / responses / native).

These assert backend validation, the default-preserving configuration, the callable tool set
each backend exposes, and that ``answer()`` routes to the right backend method — all without
any network (the backend answer methods are stubbed with sentinels for the routing test).
"""
from __future__ import annotations

import pytest

from ferber_agent.agent import (
    BACKEND_CHAT, BACKEND_NATIVE, BACKEND_RESPONSES, VALID_BACKENDS, FerberAgent,
)

_IDX = "/tmp/nonexistent_index"


def _agent(**kw) -> FerberAgent:
    return FerberAgent(chroma_dir=_IDX, faithful=True, **kw)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        _agent(backend="bogus")


def test_default_backend_and_engine_preserve_existing_behavior():
    # Defaults must match the pre-switch behavior: chat backend, cosine retrieval, no web search.
    a = _agent()
    assert a.backend == BACKEND_CHAT
    assert a.retrieval_engine == "chroma_cosine"
    assert a.web_search is False


def test_retrieval_engine_derived_from_rerank_flag(monkeypatch):
    # Back-compat: rerank=True (with a Cohere key) derives chroma_cohere; explicit wins.
    monkeypatch.setenv("COHERE_API_KEY", "x")
    assert _agent(rerank=True).retrieval_engine == "chroma_cohere"
    assert _agent(rerank=False).retrieval_engine == "chroma_cosine"
    assert _agent(rerank=True, retrieval_engine="chroma_cosine").retrieval_engine == "chroma_cosine"


def test_web_search_tool_only_on_chat_backend():
    # Web search is a callable function tool on the chat backend; on the Responses backend it is
    # the hosted tool, so it is NOT listed among the callable function tools.
    chat = _agent(backend=BACKEND_CHAT, web_search=True)
    assert "web_search" in chat._faithful_tools()
    resp = _agent(backend=BACKEND_RESPONSES, web_search=True)
    assert "web_search" not in resp._faithful_tools()
    # off by default
    assert "web_search" not in _agent(backend=BACKEND_CHAT)._faithful_tools()


@pytest.mark.parametrize("backend,method", [
    (BACKEND_CHAT, "_answer_faithful"),
    (BACKEND_RESPONSES, "_answer_faithful_responses"),
    (BACKEND_NATIVE, "_answer_native_agentic"),
])
def test_answer_routes_to_backend(monkeypatch, backend, method):
    a = _agent(backend=backend)
    called = {}
    for name in ("_answer_faithful", "_answer_faithful_responses", "_answer_native_agentic"):
        monkeypatch.setattr(a, name,
                            lambda c, q, _n=name: called.setdefault("which", _n) or "ok")
    a.answer("ctx", "q")
    assert called["which"] == method


def test_valid_backends_tuple_is_the_three_arms():
    assert set(VALID_BACKENDS) == {BACKEND_CHAT, BACKEND_RESPONSES, BACKEND_NATIVE}
