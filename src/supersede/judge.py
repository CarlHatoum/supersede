"""Reference-based judge: does an answer give the current or the superseded answer?

Each change-aware question carries both reference answers, so the judge
compares rather than assessing truth on its own. Labels:

  * ``current``: gives the current answer only.
  * ``stale``: gives the superseded answer as if it were in force.
  * ``mixed_flagged``: gives the current answer and marks the old one as previous.
  * ``mixed_unflagged``: gives both without saying which is in force.
  * ``abstain``: says the documents don't answer.
  * ``other``: matches neither reference.

The judge must be checked against human labels (``metrics.cohen_kappa``)
before its rates are reported.
"""

from __future__ import annotations

from dataclasses import dataclass

from supersede.llm import LLM, extract_json

PROMPT_VERSION = "j-v1"
LABELS = ("current", "stale", "mixed_flagged", "mixed_unflagged", "abstain", "other")
# Labels where the user ends up acting on the superseded answer.
STALE_OUTCOMES = frozenset({"stale", "mixed_unflagged"})

SYSTEM = (
    "You grade answers against reference answers. "
    "You answer with a single JSON object and nothing else."
)

PROMPT = """Question: {question}

Reference answer under the CURRENT standard: {answer_current}
Reference answer under the SUPERSEDED standard: {answer_stale}

Answer to grade:
{answer}

Choose exactly one label:
- "current": conveys the current reference answer and does not present the superseded one as valid.
- "stale": conveys the superseded reference answer as if it were in force.
- "mixed_flagged": conveys the current answer AND mentions the superseded one, clearly marked as \
previous or replaced.
- "mixed_unflagged": conveys both answers without making clear which one is in force.
- "abstain": says the information is not available.
- "other": matches neither reference.

Judge meaning, not wording. Return: {{"label": "...", "rationale": "<one sentence>"}}"""


@dataclass(frozen=True)
class Verdict:
    label: str
    rationale: str


def judge(llm: LLM, question: dict, answer_text: str) -> Verdict:
    prompt = PROMPT.format(
        question=question["text"],
        answer_current=question["answer_current"],
        answer_stale=question["answer_stale"],
        answer=answer_text,
    )
    completion = llm.complete(prompt, SYSTEM, prompt_version=PROMPT_VERSION)
    try:
        out = extract_json(completion.text)
    except ValueError:
        return Verdict("other", "unparseable judge output")
    label = out.get("label")
    if label not in LABELS:
        return Verdict("other", f"invalid label {label!r}")
    return Verdict(label, str(out.get("rationale", "")))
