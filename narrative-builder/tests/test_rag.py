"""Unit tests for the grounding guard and JSON extraction (no vLLM/LangGraph needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import extract_json
from rag_graph import check_grounding


# --- extract_json --------------------------------------------------------

def test_extract_json_plain():
    assert extract_json('{"summary": "hi", "timeline": []}') == {"summary": "hi", "timeline": []}


def test_extract_json_fenced():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nthanks'
    assert extract_json(text) == {"a": 1}


def test_extract_json_embedded_in_prose():
    text = 'Sure. {"summary": "x", "timeline": [{"id": 5}]} Done.'
    assert extract_json(text) == {"summary": "x", "timeline": [{"id": 5}]}


def test_extract_json_none_on_garbage():
    assert extract_json("no json here at all") is None
    assert extract_json("") is None


# --- check_grounding -----------------------------------------------------

ARTICLES = [{"id": 10}, {"id": 20}, {"id": 30}]


def test_grounding_all_valid():
    summary = "Event happened [1] and continued [3]."
    timeline = [{"id": 10, "why_it_matters": "a"}, {"id": 30, "why_it_matters": "b"}]
    res = check_grounding(ARTICLES, summary, timeline)
    assert res["problems"] == []
    assert len(res["clean_timeline"]) == 2


def test_grounding_out_of_range_citation():
    res = check_grounding(ARTICLES, "Claim [5].", [{"id": 10}])
    assert any("non-existent source numbers" in p for p in res["problems"])


def test_grounding_missing_citations():
    res = check_grounding(ARTICLES, "No citations here.", [{"id": 10}])
    assert any("no [n] citations" in p for p in res["problems"])


def test_grounding_fabricated_timeline_id_is_dropped():
    timeline = [{"id": 10, "why_it_matters": "ok"}, {"id": 999, "why_it_matters": "fake"}]
    res = check_grounding(ARTICLES, "Cited [1].", timeline)
    assert any("not among the" in p for p in res["problems"])
    # The fabricated id=999 entry is removed from the clean timeline.
    assert [e["id"] for e in res["clean_timeline"]] == [10]
