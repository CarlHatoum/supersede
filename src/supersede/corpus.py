"""Document model shared by every corpus builder."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    abstract: str = ""
    keywords: tuple[str, ...] = ()
    year: int | None = None
    status: str = ""
    # Supersession ("this document replaces that one entirely").
    obsoletes: tuple[str, ...] = ()
    obsoleted_by: tuple[str, ...] = ()
    # Partial amendments: the old document stays valid but is no longer complete on its own.
    updates: tuple[str, ...] = ()
    updated_by: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def text(self) -> str:
        parts = [self.title, self.abstract, " ".join(self.keywords)]
        return "\n".join(p for p in parts if p)

    @classmethod
    def from_dict(cls, d: dict) -> Doc:
        tuple_fields = ("keywords", "obsoletes", "obsoleted_by", "updates", "updated_by")
        return cls(**{k: tuple(v) if k in tuple_fields else v for k, v in d.items()})


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_docs(path: Path, docs: Iterable[Doc]) -> int:
    return write_jsonl(path, (asdict(d) for d in docs))


def load_docs(path: Path) -> list[Doc]:
    return [Doc.from_dict(d) for d in read_jsonl(path)]
