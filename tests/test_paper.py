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
    assert "agent: \\texttt{model-x}" in (out / "tab_agent_history.tex").read_text()
    traces = (out / "agent_traces.tex").read_text()
    assert "100\\% revised" in traces and "$\\rightarrow$ compare" in traces


def test_build_from_agent_compare(tmp_path):
    res, data, out = tmp_path / "results", tmp_path / "data", tmp_path / "gen"
    (res / "agent").mkdir(parents=True)
    data.mkdir()
    m = {
        "tpr": 0.9,
        "tnr": 0.7,
        "balanced_accuracy": 0.8,
        "est_precision": 0.3,
        "balanced_accuracy_ci95": [0.7, 0.9],
    }
    cost = {
        "seconds": 35.1,
        "calls": 1.0,
        "input_tokens": 595.0,
        "output_tokens": 186.0,
        "tool_calls": 0.0,
    }
    paired = {
        "n": 100,
        "a_only_correct": 5,
        "b_only_correct": 3,
        "mcnemar_p": 0.727,
        "ba_diff": 0.04,
        "ba_diff_ci95": [-0.05, 0.12],
    }
    r = {
        "n": 100,
        "band": [0.8, 0.98],
        "band_size": 2735,
        "band_positive_rate": 0.14,
        "band_positive_rate_lineage": 0.3,
        "cheap_threshold": 0.94,
        "cheap_scorer": {"strict": m, "lineage": m},
        "methods": [
            {
                "model": "ollama:x/qwen",
                "alias": "qwen",
                "variant": "llm_only",
                "prompt": "agent-v2",
                "n_decided": 100,
                "labels": {"supersedes": 60, "unrelated": 40},
                "cost": cost,
                "strict": m,
                "lineage": m,
            }
        ],
        "paired": [
            {
                "a": "qwen/llm_only/agent-v2",
                "b": "cheap scorer",
                "strict": paired,
                "lineage": paired,
            }
        ],
    }
    (res / "agent" / "compare_strict_n100.json").write_text(json.dumps(r))
    values = build(res, data, out)
    assert values["ag.qwen.llm_only.strict.ba"] == "0.80"
    assert values["ag.bandsize"] == "2{,}735"
    assert values["ag.pair.qwen.llm_only.cheap.strict.p"] == "0.73"
    assert values["ag.pair.qwen.llm_only.cheap.strict.diff.ci"] == "[-0.05, +0.12]"
    assert values["ag.qwen.llm_only.yes"] == "60\\%"
    assert "LLM only" in (out / "tab_agent.tex").read_text()
    assert "595" in (out / "tab_agent_cost.tex").read_text()
