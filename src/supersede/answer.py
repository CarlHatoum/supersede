"""Answer generation over retrieved documents, under three answer policies.

  * ``naive``: the retriever's top-k documents as they come, with no lineage.
  * ``latest_only``: promote current versions, then drop superseded documents.
  * ``show_conflict``: promote current versions and keep superseded ones, marked
    as superseded, so the answer gives the current value and can mention the
    previous one.

The lineage used here is the oracle graph. An inferred graph (from the
librarian) plugs in through the same ``LineageGraph`` type.
"""

from __future__ import annotations

from dataclasses import dataclass

from supersede.corpus import Doc
from supersede.lineage import LineageGraph
from supersede.llm import LLM, extract_json
from supersede.rerank import filter_superseded, promote_current

PROMPT_VERSION = "a-v1"
POLICIES = ("naive", "latest_only", "show_conflict")

SYSTEM = (
    "You answer engineering questions using only the documents provided. "
    "You answer with a single JSON object and nothing else."
)

INSTRUCTIONS = {
    "naive": "Answer from the documents.",
    "latest_only": "Answer from the documents.",
    "show_conflict": (
        "Some documents are marked SUPERSEDED. Answer according to the documents that are in "
        "force. If a superseded document says something different, add one sentence starting "
        '"Previously," that states the old answer.'
    ),
}

PROMPT = """Question: {question}

Documents:
{context}

{instruction}
Cite the IDs of the documents you used. If the documents do not answer the question, say so.

Return: {{"answer": "...", "citations": ["<doc id>", ...]}}"""


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[str, ...]
    context_ids: tuple[str, ...]
    policy: str


def context_ids(ranking: list[str], graph: LineageGraph, policy: str, k: int) -> list[str]:
    if policy == "naive":
        return ranking[:k]
    promoted = promote_current(ranking, graph)
    if policy == "latest_only":
        return filter_superseded(promoted, graph)[:k]
    if policy == "show_conflict":
        return promoted[:k]
    raise ValueError(f"unknown policy {policy!r}")


def labels_for(ids: list[str], blind: bool) -> dict[str, str]:
    """Context label per document. Blind mode uses D1..Dk, because RFC numbers reveal date order."""
    return {d: (f"D{i}" if blind else d) for i, d in enumerate(ids, 1)}


def render_context(
    ids: list[str], docs: dict[str, Doc], graph: LineageGraph, mark: bool, blind: bool = False
) -> str:
    """Document blocks for the prompt.

    ``blind`` removes every version cue the model could use by itself: no
    publication year, RFC numbers masked in titles and abstracts, and neutral
    document labels. The ``SUPERSEDED`` marks of ``show_conflict`` remain,
    because they are the lineage signal being tested.
    """
    from supersede.librarian.masking import mask_text

    labels = labels_for(ids, blind)
    blocks = []
    for doc_id in ids:
        d = docs[doc_id]
        title, abstract = d.title, d.abstract
        if blind:
            title, abstract = mask_text(title), mask_text(abstract)
        header = f"[{labels[doc_id]}] {title}" + (f" ({d.year})" if d.year and not blind else "")
        if mark and graph.is_superseded(doc_id):
            heads = sorted(graph.current_versions(doc_id))
            shown = [labels[h] for h in heads if h in labels] if blind else heads
            header += " — SUPERSEDED" + (" by " + ", ".join(shown) if shown else "")
        blocks.append(f"{header}\n{abstract or '(no abstract)'}")
    return "\n\n".join(blocks)


def answer(
    llm: LLM,
    question: str,
    ranking: list[str],
    docs: dict[str, Doc],
    graph: LineageGraph,
    policy: str,
    k: int = 5,
    blind: bool = False,
) -> Answer:
    ids = context_ids(ranking, graph, policy, k)
    prompt = PROMPT.format(
        question=question,
        context=render_context(ids, docs, graph, mark=policy == "show_conflict", blind=blind),
        instruction=INSTRUCTIONS[policy],
    )
    completion = llm.complete(prompt, SYSTEM, prompt_version=PROMPT_VERSION)
    try:
        out = extract_json(completion.text)
        text = str(out.get("answer", ""))
        back = {label: d for d, label in labels_for(ids, blind).items()}
        cites = tuple(back[str(c)] for c in out.get("citations", []) if str(c) in back)
    except ValueError:
        text, cites = completion.text.strip(), ()
    return Answer(text=text, citations=cites, context_ids=tuple(ids), policy=policy)
