import json

from supersede.report.paper import Numbers, build


def test_numbers_reject_tex_specials():
    n = Numbers()
    n["ok.key_1"] = 1
    assert r"\csname res@ok.key_1\endcsname{1}" in n.tex()
    try:
        n["bad%key"] = 1
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_build_from_agent_results(tmp_path):
    res, data, out = tmp_path / "results", tmp_path / "data", tmp_path / "gen"
    (res / "agent" / "history").mkdir(parents=True)
    data.mkdir()
    m = {
        "tpr": 0.9,
        "tnr": 0.8,
        "balanced_accuracy": 0.85,
        "est_precision": 0.5,
        "balanced_accuracy_ci95": [0.7, 0.95],
    }
    run = {
        "experiment": "agent_lineage",
        "model": "ollama:org/model-x:latest",
        "setting": "strict",
        "prompt_version": "agent-v2",
        "band": [0.8, 0.98],
        "band_positive_rate": 0.14,
        "band_positive_rate_lineage": 0.25,
        "cheap_threshold": 0.94,
        "n_sampled": 2,
        "n_failed": 0,
        "labels": {"supersedes": 1, "unrelated": 1},
        "mean_steps": 2.0,
        "mean_seconds": 10.0,
        "agent": m,
        "cheap_scorer": m,
        "lineage_truth": {"agent": m, "cheap_scorer": m},
    }
    (res / "agent" / "history" / "run.json").write_text(json.dumps(run))
    rows = [
        {
            "old": "A",
            "new": "B",
            "gold": 1,
            "label": "supersedes",
            "confidence": 0.9,
            "rationale": "same scope, 100% revised",
            "trace": ["→ compare({})"],
        },
    ]
    (res / "agent" / "history" / "run.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    values = build(res, data, out)
    assert values["agent.model-x.agent-v2.strict.ba"] == "0.85"
    assert values["agent.model-x.agent-v2.strict.ba.ci"] == "[0.70, 0.95]"
    assert "agent: \\texttt{model-x}" in (out / "tab_agent.tex").read_text()
    traces = (out / "agent_traces.tex").read_text()
    assert "100\\% revised" in traces and "$\\rightarrow$ compare" in traces
