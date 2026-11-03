import json

import pytest

from supersede.answer import answer, context_ids
from supersede.corpus import Doc
from supersede.judge import judge
from supersede.lineage import LineageGraph
from supersede.llm import MockLLM
from supersede.metrics import cohen_kappa

DOCS = {
    "A1": Doc("A1", "Widget protocol", "Uses UDP.", year=1990, obsoleted_by=("A2",)),
    "A2": Doc("A2", "Widget protocol v2", "Uses TCP.", year=2000, obsoletes=("A1",)),
    "B": Doc("B", "Sprocket format", "Unrelated.", year=1995),
}
GRAPH = LineageGraph(DOCS.values())
Q = {
    "text": "Which transport does the widget protocol use?",
    "answer_current": "TCP",
    "answer_stale": "UDP",
}


def test_context_by_policy():
    ranking = ["A1", "B"]
    assert context_ids(ranking, GRAPH, "naive", 5) == ["A1", "B"]
    assert context_ids(ranking, GRAPH, "latest_only", 5) == ["A2"]
    assert context_ids(ranking, GRAPH, "show_conflict", 5) == ["A2", "A1"]
    with pytest.raises(ValueError):
        context_ids(ranking, GRAPH, "nope", 5)


def test_answer_marks_superseded_only_for_show_conflict():
    seen = []

    def respond(system, prompt):
        seen.append(prompt)
        return json.dumps({"answer": "TCP", "citations": ["A2", "ZZZ"]})

    llm = MockLLM(respond)
    a = answer(llm, Q["text"], ["A1", "B"], DOCS, GRAPH, "show_conflict")
    assert "SUPERSEDED by A2" in seen[-1] and "[A2] Widget protocol v2 (2000)" in seen[-1]
    assert a.citations == ("A2",)  # citations outside the context are dropped
    answer(llm, Q["text"], ["A1", "B"], DOCS, GRAPH, "naive")
    assert "SUPERSEDED" not in seen[-1]


def test_answer_falls_back_to_raw_text():
    a = answer(MockLLM(lambda s, p: "plain text"), "q", ["B"], DOCS, GRAPH, "naive")
    assert a.text == "plain text" and a.citations == ()


def test_judge_parses_and_guards_labels():
    ok = judge(MockLLM(lambda s, p: '{"label": "stale", "rationale": "says UDP"}'), Q, "UDP")
    assert ok.label == "stale"
    bad = judge(MockLLM(lambda s, p: '{"label": "sorta"}'), Q, "UDP")
    assert bad.label == "other"


def test_cohen_kappa():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert cohen_kappa(["a"], ["a"]) == 1.0
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])


def test_blind_context_hides_version_cues_and_maps_citations_back():
    docs = {
        **DOCS,
        "A2": Doc("A2", "Widget protocol v2", "Uses TCP. Obsoletes RFC 1.", year=2000),
    }
    seen = []

    def respond(system, prompt):
        seen.append(prompt)
        return json.dumps({"answer": "TCP", "citations": ["D1", "A2"]})

    a = answer(MockLLM(respond), Q["text"], ["A1", "B"], docs, GRAPH, "show_conflict", blind=True)
    prompt = seen[-1]
    assert "(2000)" not in prompt and "RFC 1" not in prompt and "[REF]" in prompt
    assert "[D1] Widget protocol v2" in prompt and "SUPERSEDED by D1" in prompt
    assert "[A2]" not in prompt
    assert a.citations == ("A2",)  # D1 maps back; the raw ID "A2" is not a label
