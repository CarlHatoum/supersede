"""Pairwise features for "does ``new`` supersede ``old``?"."""

from __future__ import annotations

import re

from supersede.corpus import Doc
from supersede.retrieval import tokenize

FEATURES = (
    "title_jaccard",
    "title_cos",
    "text_cos",
    "year_gap",
    "same_year",
    "version_cue",
    "keyword_jaccard",
    "inv_rank",
    "explicit_cue",
    "mentions_old",
)

_VERSION = re.compile(r"\b(version\s*\d+|v\d+|bis|update[ds]?|revised|revision|edition)\b", re.I)
_SUPERSESSION_VERB = re.compile(r"\b(obsolet\w*|supersed\w*|replac\w*)\b", re.I)
# Any verb that opens a new clause about other documents ("... and updates RFC 793").
_CLAUSE_VERB = re.compile(
    r"\b(obsolet\w*|supersed\w*|replac\w*|updat\w*|defin\w*|specif\w*|describ\w*)\b", re.I
)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _number(doc_id: str) -> str | None:
    m = re.search(r"\d+", doc_id)
    return m.group() if m else None


def mentions(text: str, doc_id: str) -> bool:
    num = _number(doc_id)
    return bool(num) and re.search(rf"(?<!\d){num}(?!\d)", text) is not None


def explicit_cue(new: Doc, old: Doc) -> bool:
    """The old document's number appears in a clause opened by a supersession verb.

    The clause runs from the verb to the next clause verb or the end of the
    sentence, so in "obsoletes RFC 2581 and updates RFC 793" only 2581 counts.
    """
    if not _number(old.doc_id):
        return False
    for sentence in re.split(r"(?<=[.;])\s+", new.abstract):
        verbs = list(_CLAUSE_VERB.finditer(sentence))
        for i, v in enumerate(verbs):
            if not _SUPERSESSION_VERB.fullmatch(v.group()):
                continue
            end = verbs[i + 1].start() if i + 1 < len(verbs) else len(sentence)
            if mentions(sentence[v.end() : end], old.doc_id):
                return True
    return False


def pair_features(
    old: Doc, new: Doc, title_cos: float, text_cos: float, rank: int
) -> dict[str, float]:
    gap = (new.year - old.year) if new.year and old.year else 0
    return {
        "title_jaccard": _jaccard(set(tokenize(old.title)), set(tokenize(new.title))),
        "title_cos": title_cos,
        "text_cos": text_cos,
        "year_gap": min(gap, 30) / 10,
        "same_year": float(gap == 0),
        "version_cue": float(bool(_VERSION.search(new.title))),
        "keyword_jaccard": _jaccard(
            {k.lower() for k in old.keywords}, {k.lower() for k in new.keywords}
        ),
        "inv_rank": 1.0 / rank,
        "explicit_cue": float(explicit_cue(new, old)),
        "mentions_old": float(mentions(new.abstract, old.doc_id)),
    }
