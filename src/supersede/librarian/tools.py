"""The librarian's tools over a corpus: search, read a document, compare two documents.

These are plain Python methods. ``supersede.serve.mcp_server`` serves them over
MCP, and the agent can also call them in-process (faster, and used in tests).

They never expose lineage metadata (``obsoletes``, ``obsoleted_by``, status),
since that is what the agent has to infer. In the ``masked`` setting, document
numbers in titles and abstracts are hidden as well.
"""

from __future__ import annotations

from supersede.corpus import Doc
from supersede.librarian import masking
from supersede.librarian.features import _jaccard
from supersede.retrieval import Retriever, tokenize

MAX_ABSTRACT_CHARS = 2000


class CorpusTools:
    def __init__(self, docs: list[Doc], retriever: Retriever, setting: str = "masked"):
        self.setting = setting
        self.docs = {d.doc_id: d for d in masking.apply(docs, setting)}
        self.retriever = retriever

    def _card(self, d: Doc) -> dict:
        return {"doc_id": d.doc_id, "title": d.title, "year": d.year}

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Documents most relevant to ``query``: ID, title and year, best first."""
        k = max(1, min(int(k), 25))
        return [
            {**self._card(self.docs[doc_id]), "score": round(score, 4)}
            for doc_id, score in self.retriever.search(query, k)
        ]

    def read_doc(self, doc_id: str) -> dict:
        """Title, year, keywords and abstract of one document."""
        d = self.docs.get(doc_id)
        if d is None:
            return {"error": f"unknown document {doc_id!r}"}
        return {
            **self._card(d),
            "keywords": list(d.keywords),
            "abstract": d.abstract[:MAX_ABSTRACT_CHARS] or "(no abstract)",
        }

    def find_later(self, doc_id: str, k: int = 8) -> list[dict]:
        """Other documents on the same topic as ``doc_id`` published in the same year or later.

        Candidates for its replacement, best match first. Searches with the
        document's title and keywords.
        """
        d = self.docs.get(doc_id)
        if d is None:
            return [{"error": f"unknown document {doc_id!r}"}]
        k = max(1, min(int(k), 25))
        query = " ".join([d.title, *d.keywords])
        out = []
        for hit in self.search(query, 25):
            other = self.docs[hit["doc_id"]]
            if other.doc_id != doc_id and (not d.year or not other.year or other.year >= d.year):
                out.append(hit)
            if len(out) == k:
                break
        return out

    def compare(self, doc_a: str, doc_b: str) -> dict:
        """Two documents side by side, with simple overlap signals and their date order."""
        a, b = self.docs.get(doc_a), self.docs.get(doc_b)
        if a is None or b is None:
            missing = [x for x, d in ((doc_a, a), (doc_b, b)) if d is None]
            return {"error": f"unknown document(s): {', '.join(missing)}"}
        if a.year and b.year:
            older = doc_a if a.year < b.year else doc_b if b.year < a.year else "same year"
        else:
            older = "unknown"
        return {
            "a": self.read_doc(doc_a),
            "b": self.read_doc(doc_b),
            "title_word_overlap": _overlap(a.title, b.title),
            "abstract_word_overlap": _overlap(a.abstract, b.abstract),
            "older": older,
        }


def _overlap(x: str, y: str) -> float:
    return round(_jaccard(set(tokenize(x)), set(tokenize(y))), 3)


def load_corpus_tools(data, setting: str = "masked", retriever: str = "hybrid") -> CorpusTools:
    """Corpus tools over ``data/corpus.jsonl``, searching the same (masked or declared) text."""
    from pathlib import Path

    from supersede.corpus import load_docs
    from supersede.experiments.staleness import build_retrievers

    data = Path(data)
    docs = load_docs(data / "corpus.jsonl")
    shown = masking.apply(docs, setting)
    index = build_retrievers([retriever], [(d.doc_id, d.text) for d in shown], data)[retriever]
    return CorpusTools(docs, index, setting)
