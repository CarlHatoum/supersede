"""Hide explicit supersession cues so lineage has to be inferred from content.

A third of RFC replacements say so in the abstract ("This document obsoletes
RFC 2833"). The ``declared`` setting keeps those cues, which is realistic for
corpora like FAA circulars ("cancels AC 20-115C"). The ``masked`` setting
replaces every RFC number in titles and abstracts with a neutral token, so a
method can only use content, dates and titles.
"""

from __future__ import annotations

import re
from dataclasses import replace

from supersede.corpus import Doc

SETTINGS = ("declared", "masked")
_RFC_NUMBER = re.compile(r"\b(RFCs?)\s*-?\s*\d+(?:\s*(?:,|and|or)\s*(?:RFC\s*)?\d+)*", re.I)


def mask_text(text: str) -> str:
    return _RFC_NUMBER.sub("[REF]", text)


def apply(docs: list[Doc], setting: str) -> list[Doc]:
    if setting == "declared":
        return docs
    if setting == "masked":
        return [replace(d, title=mask_text(d.title), abstract=mask_text(d.abstract)) for d in docs]
    raise ValueError(f"unknown setting {setting!r}; choose from {SETTINGS}")
