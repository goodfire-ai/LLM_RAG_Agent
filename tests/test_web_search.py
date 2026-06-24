"""Offline unit tests for the web-search tool (replaces the discontinued google_search).

``web_search`` makes a nested OpenAI Responses ``web_search`` call; these tests stub the OpenAI
client so the tool surface, URL-citation extraction, and failure handling are exercised without
any network. The schema presence is also asserted in both schema builders.
"""
from __future__ import annotations

import types

from ferber_agent import tools


def _fake_responses_obj(text: str, urls: list[str]):
    """Build a fake Responses object whose message output carries url_citation annotations."""
    annotations = [types.SimpleNamespace(type="url_citation", url=u) for u in urls]
    part = types.SimpleNamespace(annotations=annotations)
    message = types.SimpleNamespace(type="message", content=[part])
    return types.SimpleNamespace(output_text=text, output=[message])


class _FakeOpenAI:
    def __init__(self, resp, *a, **k):
        self._resp = resp
        self.responses = types.SimpleNamespace(create=lambda **kw: self._resp)


def test_collect_url_citations_dedups_and_caps():
    resp = _fake_responses_obj("answer", ["https://a.org", "https://a.org", "https://b.org"])
    urls = tools._collect_url_citations(resp)
    assert urls == ["https://a.org", "https://b.org"]
    # non-message items are ignored
    resp2 = types.SimpleNamespace(output=[types.SimpleNamespace(type="web_search_call")])
    assert tools._collect_url_citations(resp2) == []


def test_web_search_returns_summary_and_citation_count(monkeypatch):
    resp = _fake_responses_obj("Encorafenib + cetuximab is standard for BRAF V600E mCRC.",
                               ["https://nccn.org/x", "https://fda.gov/y"])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", lambda *a, **k: _FakeOpenAI(resp))
    text, n = tools.web_search("BRAF V600E colorectal therapy")
    assert n == 2
    assert "Encorafenib" in text
    assert "Sources:" in text and "https://nccn.org/x" in text


def test_web_search_empty_query_is_guarded():
    text, n = tools.web_search("   ")
    assert n == 0 and "empty query" in text


def test_web_search_degrades_on_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("openai.OpenAI", _boom)
    text, n = tools.web_search("anything")
    assert n == 0 and text.startswith("web_search failed")


def test_web_search_in_both_schema_builders():
    chat = {s["function"]["name"] for s in tools.tool_schemas(("rag", "web_search"))}
    assert "web_search" in chat
    faithful = {s["function"]["name"]
                for s in tools.faithful_tool_schemas(("oncokb", "web_search"))}
    assert "web_search" in faithful
    # the web_search schema declares a single required `query` string
    ws = next(s for s in tools.tool_schemas(("web_search",))
              if s["function"]["name"] == "web_search")
    assert ws["function"]["parameters"]["required"] == ["query"]
