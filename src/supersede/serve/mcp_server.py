"""MCP server exposing the librarian's corpus tools.

Run over stdio (what agents and IDEs launch):
    python -m supersede.serve.mcp_server

Configuration via environment variables:
    SUPERSEDE_DATA     corpus directory (default: data/rfc)
    SUPERSEDE_SETTING  "masked" (default) or "declared"

Try it from Claude Code:
    claude mcp add supersede -- .venv/bin/python -m supersede.serve.mcp_server
"""

from __future__ import annotations

import os
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from supersede.librarian.tools import CorpusTools, load_corpus_tools

mcp = FastMCP("supersede-librarian", log_level="WARNING")


@lru_cache(maxsize=1)
def corpus() -> CorpusTools:
    return load_corpus_tools(
        os.environ.get("SUPERSEDE_DATA", "data/rfc"),
        setting=os.environ.get("SUPERSEDE_SETTING", "masked"),
    )


@mcp.tool()
def search(query: str, k: int = 10) -> list[dict]:
    """Search the document collection. Returns the most relevant documents (ID, title, year)."""
    return corpus().search(query, k)


@mcp.tool()
def read_doc(doc_id: str) -> dict:
    """Read one document: title, year, keywords and abstract."""
    return corpus().read_doc(doc_id)


@mcp.tool()
def compare(doc_a: str, doc_b: str) -> dict:
    """Compare two documents side by side, with word-overlap signals and which one is older."""
    return corpus().compare(doc_a, doc_b)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
