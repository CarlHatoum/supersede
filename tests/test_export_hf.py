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
