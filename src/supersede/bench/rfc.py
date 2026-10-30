"""Build the RFC slice of the benchmark from the RFC Editor's ``rfc-index.xml``.

Ground truth comes from the index's ``obsoletes`` / ``obsoleted-by`` fields,
so no labelling is needed.

Usage:
    python -m supersede.bench.rfc --out data/rfc            # download, then build
    python -m supersede.bench.rfc --out data/rfc --index path/to/rfc-index.xml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from supersede.corpus import Doc, save_docs, write_jsonl
from supersede.lineage import LineageGraph

INDEX_URL = "https://www.rfc-editor.org/rfc-index.xml"
NS = {"r": "https://www.rfc-editor.org/rfc-index"}
TEST_FRACTION = 0.7


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path, url: str = INDEX_URL) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as f:
        f.write(resp.read())
    return dest


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _ids(entry: ET.Element, tag: str) -> tuple[str, ...]:
    return tuple(
        _text(d) for d in entry.findall(f"r:{tag}/r:doc-id", NS) if _text(d).startswith("RFC")
    )


def parse_index(path: Path) -> list[Doc]:
    docs = []
    for entry in ET.parse(path).getroot().findall("r:rfc-entry", NS):
        year = _text(entry.find("r:date/r:year", NS))
        docs.append(
            Doc(
                doc_id=_text(entry.find("r:doc-id", NS)),
                title=_text(entry.find("r:title", NS)),
                abstract=" ".join(_text(p) for p in entry.findall("r:abstract/r:p", NS)),
                keywords=tuple(_text(k) for k in entry.findall("r:keywords/r:kw", NS)),
                year=int(year) if year.isdigit() else None,
                status=_text(entry.find("r:current-status", NS)),
                obsoletes=_ids(entry, "obsoletes"),
                obsoleted_by=_ids(entry, "obsoleted-by"),
                updates=_ids(entry, "updates"),
                updated_by=_ids(entry, "updated-by"),
            )
        )
    return docs


def split_of(component: frozenset[str]) -> str:
    """Deterministic split by lineage component, so no topic leaks from dev to test."""
    key = min(component).encode()
    bucket = int(hashlib.sha256(key).hexdigest(), 16) % 1000
    return "test" if bucket < TEST_FRACTION * 1000 else "dev"


def build_queries(docs: list[Doc], graph: LineageGraph) -> list[dict]:
    """One query per supersession edge and naming variant.

    v0 queries are built from titles, with two variants:
      * ``old_title``: the topic named as users know it from the old document.
      * ``new_title``: the topic named with the replacement's wording.
    The gold set is always the current heads of the old document's lineage.
    LLM-written natural-language questions come in a later version.
    """
    by_id = {d.doc_id: d for d in docs}
    queries, seen = [], set()
    for old, new in graph.edges:
        comp = graph.component(old)
        current = sorted(graph.current_versions(old))
        stale = sorted(d for d in comp if graph.is_superseded(d))
        for variant, source in (("old_title", old), ("new_title", new)):
            text = by_id[source].title
            key = (variant, text.lower(), tuple(current))
            if not text or key in seen:
                continue
            seen.add(key)
            queries.append(
                {
                    "qid": f"rfc-{variant}-{old}-{new}",
                    "text": text,
                    "variant": variant,
                    "source_doc": source,
                    "edge": [old, new],
                    "current": current,
                    "stale": stale,
                    "component": min(comp),
                    "split": split_of(comp),
                }
            )
    return queries


def build(index_path: Path, out: Path) -> dict:
    docs = parse_index(index_path)
    graph = LineageGraph(docs)
    queries = build_queries(docs, graph)
    comps = graph.components()

    save_docs(out / "corpus.jsonl", docs)
    write_jsonl(out / "lineage.jsonl", ({"old": o, "new": n} for o, n in graph.edges))
    write_jsonl(out / "queries.jsonl", queries)

    chain_depths = [len(c) for c in comps]
    manifest = {
        "source": INDEX_URL,
        "index_sha256": sha256(index_path),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_docs": len(docs),
        "n_docs_with_abstract": sum(bool(d.abstract) for d in docs),
        "n_supersession_edges": len(graph.edges),
        "n_superseded_docs": sum(graph.is_superseded(d.doc_id) for d in docs),
        "n_update_edges": sum(len(d.updates) for d in docs),
        "n_components": len(comps),
        "max_component_size": max(chain_depths, default=0),
        "asymmetric_edges": graph.asymmetric_edges,
        "dangling_edges": graph.dangling_edges,
        "n_queries": len(queries),
        "n_queries_by_split": {s: sum(q["split"] == s for q in queries) for s in ("dev", "test")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/rfc"))
    ap.add_argument("--index", type=Path, help="local rfc-index.xml; downloaded if omitted")
    ap.add_argument("--expect-sha256", help="fail if the index snapshot differs")
    args = ap.parse_args()

    index = args.index or args.out / "raw" / "rfc-index.xml"
    if not index.exists():
        print(f"downloading {INDEX_URL} -> {index}")
        download(index)
    if args.expect_sha256 and sha256(index) != args.expect_sha256:
        raise SystemExit(f"snapshot mismatch: {sha256(index)} != {args.expect_sha256}")

    print(json.dumps(build(index, args.out), indent=2))


if __name__ == "__main__":
    main()
