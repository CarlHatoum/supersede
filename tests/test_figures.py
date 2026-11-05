from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from supersede.report.figures import build  # noqa: E402

RESULTS = Path(__file__).parents[1] / "results"


@pytest.mark.skipif(not (RESULTS / "rfc_staleness_test.json").exists(), reason="no results")
def test_figures_render_from_committed_results(tmp_path):
    made = build(RESULTS, tmp_path)
    core = ("fig_retrieval", "fig_answers", "fig_lineage", "fig_agent")
    assert all(made[name] for name in core), made
    if (RESULTS / "rfc_recency_noise.json").exists():
        assert made["fig_noise"]
    assert all((tmp_path / f"{n}.pdf").stat().st_size > 1000 for n, ok in made.items() if ok)


def test_missing_results_skip_figures(tmp_path):
    (tmp_path / "agent").mkdir()
    assert not any(build(tmp_path, tmp_path / "out").values())
