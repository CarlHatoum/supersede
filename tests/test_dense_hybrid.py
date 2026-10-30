import hashlib

import pytest

np = pytest.importorskip("numpy")

from supersede.retrieval import BM25, DenseRetriever, HybridRRF, tokenize  # noqa: E402

DOCS = [
    ("A1", "widget transfer protocol"),
    ("A2", "widget transfer protocol version two"),
    ("B", "sprocket file format"),
    ("C", "gadget naming registry"),
]


class FakeEncoder:
    """Deterministic hashed bag-of-words vectors, counting how often it is called."""

    def __init__(self, dim: int = 64):
        self.dim, self.calls = dim, 0

    def __call__(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def test_dense_ranks_closest_text_first():
    dense = DenseRetriever(DOCS, "fake", encoder=FakeEncoder(), query_prefix="")
    assert dense.search("sprocket format", k=2)[0][0] == "B"
    assert [h[0][0] for h in dense.search_many(["gadget registry", "widget protocol"], 1)] == [
        "C",
        "A1",
    ]


def test_dense_cache_is_reused_and_keyed_on_corpus(tmp_path):
    enc = FakeEncoder()
    DenseRetriever(DOCS, "fake", cache_dir=tmp_path, encoder=enc, query_prefix="")
    assert enc.calls == 1
    enc2 = FakeEncoder()
    DenseRetriever(DOCS, "fake", cache_dir=tmp_path, encoder=enc2, query_prefix="")
    assert enc2.calls == 0  # documents read from cache
    DenseRetriever(DOCS[:3], "fake", cache_dir=tmp_path, encoder=enc2, query_prefix="")
    assert enc2.calls == 1  # different corpus, re-encoded
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_bge_models_get_query_prefix():
    dense = DenseRetriever(DOCS, "BAAI/bge-small-en-v1.5", encoder=FakeEncoder())
    assert dense.query_prefix == DenseRetriever.BGE_QUERY_PREFIX


def test_hybrid_rrf_fuses_and_is_deterministic():
    bm25 = BM25(DOCS)
    dense = DenseRetriever(DOCS, "fake", encoder=FakeEncoder(), query_prefix="")
    hybrid = HybridRRF([bm25, dense], rrf_k=60)
    hits = hybrid.search("widget transfer protocol", k=3)
    assert hits[0][0] == "A1"
    # Top doc of both lists gets 2 / (60 + 1).
    assert hits[0][1] == pytest.approx(2 / 61)
    assert hybrid.search_many(["widget transfer protocol"], 3)[0] == hits
    assert hybrid.name == "rrf(bm25(k1=1.2,b=0.75),dense(fake);k=60)"
