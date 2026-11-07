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


def test_card_with_figures_and_examples(tmp_path):
    import yaml

    from supersede.bench.export_hf import ASSETS, card, copy_figures

    figs = tmp_path / "figs"
    figs.mkdir()
    (figs / "fig_card_example.png").write_bytes(b"png")
    (figs / "fig_retrieval.png").write_bytes(b"png")  # a results figure: never shipped
    assets = copy_figures(figs, tmp_path / "out" / "assets")
    assert assets == {"fig_card_example"}

    manifest = {
        "n_docs": 7,
        "n_supersession_edges": 4,
        "n_components": 2,
        "max_component_size": 3,
        "index_sha256": "abc",
        "built_at": "2026-10-30T00:00:00",
    }
    counts = {"corpus": {"corpus": 7}, "questions": {"dev": 0, "test": 1}}
    q = {"change_type": "value", "reference_new_is_superseded": False, "text": "Q?"}
    edge = {"old": "RFC1", "new": "RFC2", "component": "RFC1", "split": "test"}
    text = card(
        manifest, {}, counts, {"dev": [], "test": [q]}, assets, {"questions": q, "lineage": edge}
    )

    header = text.split("---")[1]
    assert yaml.safe_load(header)["configs"][0]["config_name"] == "corpus"
    assert f"{ASSETS}/fig_card_example.png" in text
    assert "fig_retrieval" not in text and "paper" not in text.lower()
    assert 'load_dataset("hatoum/supersede-rfc", "queries", split="test")' in text
    assert "<summary><b>Example row: <code>lineage</code></b></summary>" in text
    assert "<code>pairs</code>" not in text.split("## Fields")[0]  # no pairs example given
