from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from supersede.report.figures import build  # noqa: E402

RESULTS = Path(__file__).parents[1] / "results"


@pytest.mark.skipif(not (RESULTS / "rfc_staleness_test.json").exists(), reason="no results")
def test_figures_render_from_committed_results(tmp_path):
    made = build(RESULTS, tmp_path)
    assert all(made.values()), made
    assert all((tmp_path / f"{name}.pdf").stat().st_size > 1000 for name in made)


def test_missing_results_skip_figures(tmp_path):
    (tmp_path / "agent").mkdir()
    assert not any(build(tmp_path, tmp_path / "out").values())
