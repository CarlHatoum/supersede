import json

from supersede.corpus import write_jsonl
from supersede.experiments.closed_book import joint, run
from supersede.llm import CachedLLM, MockLLM


def test_joint_override_and_rescue():
    cb = {"q1": "current", "q2": "current", "q3": "stale", "q4": "abstain"}
    rag = [
        {"qid": "q1", "policy": "naive", "label": "stale"},
        {"qid": "q2", "policy": "naive", "label": "current"},
        {"qid": "q3", "policy": "naive", "label": "current"},
        {"qid": "q4", "policy": "naive", "label": "stale"},
    ]
    d = joint(cb, rag)["naive"]
    assert d["override"]["rate"] == 0.5 and d["override"]["n"] == 2
    assert d["rescue"]["rate"] == 0.5 and d["rescue"]["n"] == 2


def test_run_with_mock(tmp_path):
    data, res = tmp_path / "data", tmp_path / "results"
    data.mkdir(), res.mkdir()
    qs = [
        {"qid": f"q{i}", "text": f"What is X{i}?", "answer_current": "new", "answer_stale": "old"}
        for i in range(3)
    ]
    write_jsonl(data / "questions.jsonl", qs)
    write_jsonl(
        res / "rfc_answer_staleness_hybrid_items.jsonl",
        [{"qid": "q0", "policy": "naive", "label": "stale"}],
    )
    answerer = MockLLM(lambda s, p: json.dumps({"answer": "new"}))
    judge = MockLLM(lambda s, p: json.dumps({"label": "current", "rationale": "ok"}))
    llm = CachedLLM(answerer, tmp_path / "c.sqlite")
    jl = CachedLLM(judge, tmp_path / "c.sqlite")
    r = run(data, res, llm, jl, workers=1)
    assert r["labels"]["current"] == 3 and r["current"]["rate"] == 1.0
    assert "documents" not in answerer.calls[0][1].lower()
    assert r["rag"]["standard"]["naive"]["override"]["rate"] == 1.0
