"""Hide explicit supersession cues so lineage has to be inferred from content.

A third of RFC replacements say so in the abstract ("This document obsoletes
RFC 2833"). Three settings, from easiest to hardest:

* ``declared`` keeps those cues, which is realistic for corpora like FAA
  circulars ("cancels AC 20-115C").
* ``masked`` replaces every RFC number in titles and abstracts with a neutral
  token. The sentence "This document obsoletes [REF]" still tells a reader that
  *something* is replaced, and an LLM agent does use that.
* ``strict`` also removes every sentence with a supersession verb, so a method
  can only use content, dates and titles. This is the setting for agents.
"""

from __future__ import annotations

import re
from dataclasses import replace

from supersede.corpus import Doc

SETTINGS = ("declared", "masked", "strict")
_RFC_NUMBER = re.compile(r"\b(RFCs?)\s*-?\s*\d+(?:\s*(?:,|and|or)\s*(?:RFC\s*)?\d+)*", re.I)


_SUPERSESSION_VERB = re.compile(r"\b(obsolet\w*|supersed\w*|replac\w*)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.;])\s+")


def mask_text(text: str) -> str:
    return _RFC_NUMBER.sub("[REF]", text)


def strip_supersession(text: str) -> str:
    """Drop every sentence that mentions obsoleting, superseding or replacing."""
    kept = [s for s in _SENTENCE.split(text) if not _SUPERSESSION_VERB.search(s)]
    return " ".join(kept)


def apply(docs: list[Doc], setting: str) -> list[Doc]:
    if setting == "declared":
        return docs
    if setting == "masked":
        return [replace(d, title=mask_text(d.title), abstract=mask_text(d.abstract)) for d in docs]
    if setting == "strict":
        return [
            replace(
                d,
                title=mask_text(d.title),
                abstract=mask_text(strip_supersession(d.abstract)),
            )
            for d in docs
        ]
    raise ValueError(f"unknown setting {setting!r}; choose from {SETTINGS}")
