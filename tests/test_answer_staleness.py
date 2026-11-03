import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from supersede.bench.rfc import build as build_rfc  # noqa: E402
from supersede.corpus import read_jsonl, write_jsonl  # noqa: E402
from supersede.experiments import answer_staleness as exp  # noqa: E402
from supersede.llm import CachedLLM, MockLLM  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "rfc-index-mini.xml"


def test_run_export_and_kappa(tmp_path):
    build_rfc(FIXTURE, tmp_path)
    write_jsonl(
        tmp_path / "questions.jsonl",
        [
            {
                "qid": "q1",
                "text": "widget transfer protocol framing",
                "current": ["RFC3"],
                "stale": ["RFC1", "RFC2"],
                "answer_current": "framed",
                "answer_stale": "unframed",
            }
        ],
    )

    def answerer(system, prompt):
        superseded_shown = "SUPERSEDED" in prompt
        return json.dumps({"answer": "framed" if superseded_shown else "unframed"})

    def judger(system, prompt):
        return json.dumps({"label": "current" if "\nframed" in prompt else "stale"})

    llm = CachedLLM(MockLLM(answerer, "a"), tmp_path / "c.sqlite")
    judge_llm = CachedLLM(MockLLM(judger, "j"), tmp_path / "c.sqlite")
    result = exp.run(tmp_path, "bm25", llm, judge_llm, workers=1)
    rows = {r["policy"]: r for r in result["rows"]}
    assert rows["naive"]["stale_outcome"] == 1.0
    assert rows["show_conflict"]["current_outcome"] == 1.0
    assert result["n_missing"] == 0

    items_path = tmp_path / "items.jsonl"
    write_jsonl(items_path, result["items"])
    out = tmp_path / "to_label.jsonl"
    assert exp.export_labeling(items_path, 2, out) == 2
    rows = list(read_jsonl(out))
    assert all("label" not in r and r["human_label"] == "" for r in rows)
    judged = {(i["qid"], i["policy"]): i["label"] for i in result["items"]}
    write_jsonl(out, [{**r, "human_label": judged[(r["qid"], r["policy"])]} for r in rows])
    assert exp.kappa(out, items_path)["accuracy"] == 1.0


def test_export_csv_roundtrip(tmp_path):
    import csv

    items = tmp_path / "items.jsonl"
    write_jsonl(
        items,
        [
            {
                "qid": f"q{i}",
                "policy": "naive",
                "question": "Q, with comma",
                "answer_current": "a",
                "answer_stale": "b",
                "answer": 'says "a"',
                "label": lab,
            }
            for i, lab in enumerate(["current", "stale", "current"])
        ],
    )
    out = tmp_path / "to_label.csv"
    assert exp.export_labeling(items, 3, out) == 3
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["qid"] for r in rows} == {"q0", "q1", "q2"}
    assert rows[0]["question"] == "Q, with comma" and "label" not in rows[0]
    gold = {"q0": "current", "q1": "stale", "q2": "current"}
    for r in rows:
        r["human_label"] = gold[r["qid"]]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert exp.kappa(out, items) == {"n": 3, "kappa": 1.0, "accuracy": 1.0}
