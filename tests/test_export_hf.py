import importlib.util
import json
from pathlib import Path

from supersede.bench.export_hf import export
from supersede.bench.rfc import build

FIXTURE = Path(__file__).parent / "fixtures" / "rfc-index-mini.xml"


def test_export_ships_verbatim_text_and_a_matching_masking_script(tmp_path):
    data, out = tmp_path / "data", tmp_path / "hf"
    build(FIXTURE, data)
    counts = export(data, out, with_pairs=False)
    assert counts["corpus"]["corpus"] == 7
    assert sum(counts["lineage"].values()) == 4
    rows = [json.loads(line) for line in (out / "corpus" / "corpus.jsonl").open()]
    assert not any(k.endswith(("_masked", "_strict")) for k in rows[0])
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-4.0") and "config_name: corpus" in card

    spec = importlib.util.spec_from_file_location("hfmask", out / "masking.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    row = {"title": "RFC 1 bis", "abstract": "New framing. This document obsoletes RFC 1."}
    assert mod.apply(row, "masked")["abstract"] == "New framing. This document obsoletes [REF]."
    assert mod.apply(row, "strict") == {"title": "[REF] bis", "abstract": "New framing."}


def test_card_with_figures_and_numbers(tmp_path):
    import yaml

    from supersede.bench.export_hf import ASSETS, card, copy_figures, plain

    figs = tmp_path / "figs"
    figs.mkdir()
    (figs / "fig_card_problem.png").write_bytes(b"png")
    (figs / "fig_retrieval.png").write_bytes(b"png")
    assets = copy_figures(figs, tmp_path / "out" / "assets")
    assert assets == {"fig_card_problem", "fig_retrieval"}

    manifest = {
        "n_docs": 7,
        "n_supersession_edges": 4,
        "n_components": 2,
        "index_sha256": "abc",
        "built_at": "2026-10-30T00:00:00",
    }
    counts = {"corpus": {"corpus": 7}, "questions": {"dev": 0, "test": 1}}
    q = {"change_type": "value", "reference_new_is_superseded": False, "text": "Q?"}
    nums = {
        k: plain(v)
        for k, v in {
            "qret.hybrid.none.stale1.pct": r"65\%",
            "qret.hybrid.none.stale1.cipct": r"[54\%, 77\%]",
            "qret.bm25.none.stale1.pct": r"57\%",
            "qret.dense.none.stale1.pct": r"55\%",
        }.items()
    }
    text = card(manifest, {}, counts, {"dev": [], "test": [q]}, nums, assets, {"questions": q})

    header = text.split("---")[1]
    assert yaml.safe_load(header)["configs"][0]["config_name"] == "corpus"
    assert "**obsolete version first 65% of the time** (95% CI [54%, 77%])" in text
    assert "Promoting the current version" not in text  # its numbers are missing
    assert "??" not in text and "\\%" not in text
    assert f"{ASSETS}/fig_card_problem.png" in text
    assert "fig_answers.png" not in text  # not copied, not linked
    assert 'load_dataset("hatoum/supersede-rfc", "questions", split="test")' in text
    assert "<summary><b>Example row: <code>questions</code></b></summary>" in text
