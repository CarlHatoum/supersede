import pytest

np = pytest.importorskip("numpy")

from supersede.corpus import Doc  # noqa: E402
from supersede.librarian import masking  # noqa: E402
from supersede.librarian.candidates import doc_order, neighbour_pairs  # noqa: E402
from supersede.librarian.classifiers import (  # noqa: E402
    LogisticRegression,
    edge_prf,
    heuristic_score,
    select_edges,
)
from supersede.librarian.features import FEATURES, explicit_cue, pair_features  # noqa: E402
from supersede.lineage import LineageGraph  # noqa: E402

OLD = Doc("RFC2581", "TCP Congestion Control", "Defines congestion control.", year=1999)
NEW = Doc(
    "RFC5681",
    "TCP Congestion Control",
    "Specifies four algorithms. This document obsoletes RFC 2581 and updates RFC 793.",
    year=2009,
)


def test_masking_hides_all_reference_numbers():
    assert masking.mask_text("obsoletes RFCs 2818, 7231 and 7232; see RFC-793.") == (
        "obsoletes [REF]; see [REF]."
    )
    masked = masking.apply([NEW], "masked")[0]
    assert "2581" not in masked.abstract and masked.title == NEW.title
    assert masking.apply([NEW], "declared")[0] is NEW
    with pytest.raises(ValueError):
        masking.apply([NEW], "other")


def test_explicit_cue_needs_verb_and_number_in_one_sentence():
    assert explicit_cue(NEW, OLD)
    other = Doc("RFC793", "TCP", year=1981)
    assert not explicit_cue(NEW, other)  # "updates", not a supersession verb
    assert not explicit_cue(masking.apply([NEW], "masked")[0], OLD)


def test_pair_features_and_heuristic():
    f = pair_features(OLD, NEW, title_cos=1.0, text_cos=0.9, rank=1)
    assert set(f) == set(FEATURES)
    assert f["title_jaccard"] == 1.0 and f["year_gap"] == 1.0 and f["explicit_cue"] == 1.0
    assert heuristic_score(f) == 1.0
    weak = {**f, "explicit_cue": 0.0, "title_cos": 0.5}
    assert heuristic_score(weak) == 0.0


def test_doc_order_and_neighbour_pairs_are_old_to_new():
    docs = [NEW, OLD, Doc("RFC9999", "Unrelated", year=2020)]
    assert doc_order(OLD) < doc_order(NEW)
    emb = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0]], dtype=np.float32)
    pairs = neighbour_pairs(docs, emb, n_neighbours=1)
    assert pairs[("RFC2581", "RFC5681")] == 1


def test_logreg_learns_separable_signal():
    rows = [{k: 0.0 for k in FEATURES} | {"title_cos": v} for v in (0.1, 0.2, 0.8, 0.9)]
    model = LogisticRegression(steps=500).fit(rows, [0, 0, 1, 1])
    p = model.predict_proba(rows)
    assert p[0] < 0.5 < p[3]
    assert model.weights()["title_cos"] > 0


def test_select_edges_keeps_best_successor_and_prf():
    pairs = [("a", "b"), ("a", "c"), ("d", "e")]
    assert select_edges(pairs, [0.6, 0.9, 0.2], 0.5) == [("a", "c")]
    prf = edge_prf({("a", "c")}, {("a", "c"), ("d", "e")})
    assert prf == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "tp": 1}


def test_lineage_from_edges():
    g = LineageGraph.from_edges([OLD, NEW], [("RFC2581", "RFC5681")])
    assert g.current_versions("RFC2581") == {"RFC5681"}
    assert LineageGraph.from_edges([OLD, NEW], []).edges == []
