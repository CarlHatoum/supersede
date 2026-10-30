from supersede.corpus import Doc
from supersede.lineage import LineageGraph
from supersede.metrics import (
    bootstrap_ci,
    current_at_k,
    reciprocal_rank,
    stale_at_1,
    stale_before_current,
)
from supersede.rerank import filter_superseded, promote_current
from supersede.retrieval import BM25, tokenize

DOCS = [
    Doc("A1", "Widget Transfer Protocol", obsoleted_by=("A2",)),
    Doc("A2", "Widget Transfer Protocol Version 2", obsoletes=("A1",), obsoleted_by=("A3",)),
    Doc("A3", "Widget Transfer Protocol Version 3", obsoletes=("A2",)),
    Doc("B", "Sprocket Format"),
]
GRAPH = LineageGraph(DOCS)


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The HTTP/1.1 Semantics, of the Web") == ["http", "1", "1", "semantics", "web"]


def test_bm25_prefers_shortest_exact_title():
    index = BM25([(d.doc_id, d.text) for d in DOCS])
    hits = [d for d, _ in index.search("widget transfer protocol", k=10)]
    assert hits[0] == "A1"
    assert "B" not in hits


def test_bm25_is_deterministic_on_ties():
    index = BM25([("x", "same text"), ("y", "same text")])
    assert [d for d, _ in index.search("same", 2)] == ["x", "y"]


def test_promote_puts_current_first_and_keeps_length():
    assert promote_current(["A1", "B"], GRAPH) == ["A3", "A1"]
    assert promote_current(["A1", "A2", "A3"], GRAPH) == ["A3", "A1", "A2"]


def test_filter_drops_superseded():
    assert filter_superseded(["A1", "A2", "B"], GRAPH) == ["B"]


def test_metrics():
    current, stale = {"A3"}, {"A1", "A2"}
    assert stale_at_1(["A1", "A3"], current, stale) == 1.0
    assert stale_before_current(["B", "A3", "A1"], current, stale) == 0.0
    assert stale_before_current(["A2", "B"], current, stale) == 1.0
    assert current_at_k(["A1", "A3"], current, 1) == 0.0
    assert reciprocal_rank(["A1", "A3"], current) == 0.5


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert lo < 0.5 < hi
