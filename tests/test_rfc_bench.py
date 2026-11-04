import json
from pathlib import Path

import pytest

from supersede.bench.rfc import build, parse_index
from supersede.corpus import read_jsonl
from supersede.lineage import LineageGraph

FIXTURE = Path(__file__).parent / "fixtures" / "rfc-index-mini.xml"


@pytest.fixture(scope="module")
def docs():
    return parse_index(FIXTURE)


def test_parse_keeps_only_rfc_entries(docs):
    assert [d.doc_id for d in docs] == ["RFC1", "RFC2", "RFC3", "RFC10", "RFC11", "RFC12", "RFC20"]


def test_parse_fields(docs):
    rfc2 = next(d for d in docs if d.doc_id == "RFC2")
    assert rfc2.abstract == "Second version of the widget transfer protocol. Adds framing."
    assert rfc2.year == 2000
    assert rfc2.obsoletes == ("RFC1",)
    assert next(d for d in docs if d.doc_id == "RFC3").updates == ("RFC20",)


def test_lineage_uses_both_declaration_sides(docs):
    g = LineageGraph(docs)
    # RFC2 -> RFC3 is declared only on RFC3's side.
    assert g.current_versions("RFC1") == {"RFC3"}
    assert g.asymmetric_edges == 1
    # RFC99 is not in the corpus.
    assert g.dangling_edges == 1


def test_descendants_follow_the_chain(docs):
    g = LineageGraph(docs)
    assert g.descendants("RFC1") == {"RFC2", "RFC3"}
    assert g.descendants("RFC3") == frozenset()


def test_lineage_split_and_untouched(docs):
    g = LineageGraph(docs)
    assert g.current_versions("RFC10") == {"RFC11", "RFC12"}
    assert g.current_versions("RFC20") == {"RFC20"}
    assert not g.is_superseded("RFC20")  # "updates" is not supersession
    assert g.component("RFC2") == {"RFC1", "RFC2", "RFC3"}
    assert len(g.components()) == 2


def test_build_writes_consistent_artifacts(tmp_path):
    manifest = build(FIXTURE, tmp_path)
    assert manifest["n_docs"] == 7
    assert manifest["n_supersession_edges"] == 4
    queries = list(read_jsonl(tmp_path / "queries.jsonl"))
    # "Gadget Naming" (old title of both RFC10 edges, same gold set) is kept once.
    assert len(queries) == manifest["n_queries"] == 7
    q = next(q for q in queries if q["qid"] == "rfc-old_title-RFC1-RFC2")
    assert q["text"] == "Widget Transfer Protocol"
    assert q["current"] == ["RFC3"]
    assert q["stale"] == ["RFC1", "RFC2"]
    # Every query of a lineage component lands in the same split.
    for comp in {q["component"] for q in queries}:
        assert len({q["split"] for q in queries if q["component"] == comp}) == 1
    assert json.loads((tmp_path / "manifest.json").read_text())["index_sha256"]


def test_clean_text_undoes_typesetting_artefacts():
    from supersede.bench.rfc import clean_text

    assert clean_text("a \\%application- level  protocol") == "a application-level protocol"
    assert clean_text("inter- and intra-domain") == "inter- and intra-domain"
    assert clean_text("self- organizing") == "self-organizing"
