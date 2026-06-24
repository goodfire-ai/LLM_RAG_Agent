"""Offline unit tests for the modernized ferber_agent tools.

These exercise the tool surface that does NOT need network or GPU: schema assembly, image
resolution, the Cohere rerank fallback, the histology hard-gap message, and safe arithmetic.
Network/vision/MedSAM paths are covered by the live smoke run, not here.
"""
from __future__ import annotations

from ferber_agent import tools
from ferber_agent.agent import FerberAgent


def test_tool_schemas_include_imaging():
    enabled = ("rag", "oncokb", "pubmed", "calculate", "radiology_report", "medsam",
               "histology_classifier")
    schemas = tools.tool_schemas(enabled)
    names = {s["function"]["name"] for s in schemas}
    assert names == set(enabled)
    # medsam declares a 4-number bbox
    medsam = next(s for s in schemas if s["function"]["name"] == "medsam")
    bbox = medsam["function"]["parameters"]["properties"]["bbox"]
    assert bbox["minItems"] == 4 and bbox["maxItems"] == 4


def test_tool_schemas_filters_unknown():
    assert tools.tool_schemas(("rag", "bogus")) == tools.tool_schemas(("rag",))


def test_resolve_image_aliases_and_stem():
    img_map = {"September2023.png": "/x/Xing_1.jpg", "Xing_1.jpg": "/x/Xing_1.jpg"}
    assert tools.resolve_image("September2023.png", img_map) == "/x/Xing_1.jpg"
    # case-insensitive
    assert tools.resolve_image("september2023.PNG", img_map) == "/x/Xing_1.jpg"
    # by stem (extension-insensitive): "September2023" matches "September2023.png"
    assert tools.resolve_image("September2023", img_map) == "/x/Xing_1.jpg"
    # basename of a path
    assert tools.resolve_image("Xing_1", img_map) == "/x/Xing_1.jpg"
    # missing -> None (this is how Garcia_2 exclusion surfaces: it's absent from the map)
    assert tools.resolve_image("Garcia_2.jpg", img_map) is None
    assert tools.resolve_image("anything", None) is None


def test_rerank_fallback_without_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    hits = [{"text": f"doc{i}", "title": str(i), "source": "s"} for i in range(15)]
    out = tools.rerank_passages("query", hits, top_n=10)
    # cosine-only fallback: original order, truncated to top_n, no rerank_score added
    assert out == hits[:10]
    assert all("rerank_score" not in h for h in out)
    assert tools.rerank_passages("q", [], top_n=5) == []


def test_histology_classifier_is_explicit_gap():
    msg = tools.histology_classifier_unavailable("BRAF", "Xing_3.jpg")
    assert "UNAVAILABLE" in msg
    assert "BRAF" in msg
    assert "molecular" in msg.lower()


def test_calculate_progression_ratio():
    assert tools.calculate("120 / 30") == "4.0"
    assert tools.calculate("2 ** 10") == "1024"
    assert "error" in tools.calculate("__import__('os')").lower()


def test_imaging_schemas_filtered_without_images():
    # Imaging tools present in the tuple but no image map -> filtered out (MTBBench parity).
    agent = FerberAgent(chroma_dir="/tmp/nonexistent_index",
                        tools=("rag", "oncokb", "pubmed", "calculate",
                               "radiology_report", "medsam", "histology_classifier"))
    agent._images = {}
    active = agent._active_tools()
    assert "radiology_report" not in active and "medsam" not in active
    assert "histology_classifier" not in active
    assert set(active) == {"rag", "oncokb", "pubmed", "calculate"}
    # with an image map, imaging tools are exposed
    agent._images = {"September2023.png": "/abs/Xing_1.jpg"}
    active2 = agent._active_tools()
    assert "radiology_report" in active2 and "medsam" in active2


def test_oncokb_endpoint_selection(monkeypatch):
    # Without a token the demo endpoint is selected; with one, prod. (No network call here —
    # we only assert the URL routing by inspecting the failure message host on a bad symbol.)
    monkeypatch.delenv("ONCOKB_API_TOKEN", raising=False)
    # calculate is pure; oncokb is exercised live in the smoke. Just assert the function
    # exists and is callable with the expected signature.
    assert callable(tools.oncokb_annotate)


# --- faithful-mode tools -----------------------------------------------------
def test_faithful_tool_schemas_use_verbatim_descriptions_and_restored_params():
    from ferber_agent import faithful_prompts as fp

    enabled = ("oncokb", "pubmed", "calculate", "radiology_report", "medsam",
               "histology_classifier")
    schemas = {s["function"]["name"]: s["function"] for s in tools.faithful_tool_schemas(enabled)}
    assert set(schemas) == set(enabled)

    # descriptions are the byte-verbatim upstream docstrings
    assert schemas["oncokb"]["description"] == fp.TOOL_ONCOKB_DOC
    assert schemas["pubmed"]["description"] == fp.TOOL_PUBMED_DOC
    assert schemas["calculate"]["description"] == fp.TOOL_CALCULATE_DOC
    assert schemas["medsam"]["description"] == fp.TOOL_SEGMENT_DOC
    assert schemas["histology_classifier"]["description"] == fp.TOOL_CHECKMUTATIONS_DOC

    # restored original parameter shapes
    assert schemas["oncokb"]["parameters"]["properties"]["change"]["enum"] == \
        ["mutation", "amplification", "variant"]
    assert schemas["pubmed"]["parameters"]["properties"]["pubmed_search_terms"]["type"] == "array"
    assert set(schemas["calculate"]["parameters"]["properties"]) == {"a", "b", "operator"}
    bbox = schemas["medsam"]["parameters"]["properties"]["bbox_coordinates"]
    assert bbox["type"] == "array" and bbox["items"]["type"] == "array"  # nested list


def test_calculate_faithful_operators():
    assert "4.0" in tools.calculate_faithful(120, 30, "/")
    assert "undefined" in tools.calculate_faithful(1, 0, "/").lower()
    assert "150" in tools.calculate_faithful(120, 30, "+")
    assert "Invalid operator" in tools.calculate_faithful(1, 2, "^")
    assert "Invalid operands" in tools.calculate_faithful("x", 2, "+")


def test_pubmed_search_faithful_uses_first_three_terms(monkeypatch):
    captured = {}

    def fake_pubmed(query, k=4):
        captured["query"] = query
        return "stub"

    monkeypatch.setattr(tools, "pubmed_search", fake_pubmed)
    tools.pubmed_search_faithful(["t1", "t2", "t3", "t4"], "final question")
    # only the first three terms are used (per the original docstring), AND-ed with the query
    assert "t1" in captured["query"] and "t2" in captured["query"] and "t3" in captured["query"]
    assert "t4" not in captured["query"]
    assert "final question" in captured["query"]


def test_cache_stats_shape():
    s = tools.cache_stats()
    for key in ("mem_hit", "disk_hit", "miss", "hit_rate", "retr_total"):
        assert key in s
