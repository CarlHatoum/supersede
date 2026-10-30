"""LLM backends behind one interface, plus a response cache.

Backends:
  * ``ClaudeCLI``: Claude through the Claude Code CLI in headless mode
    (``claude -p``), billed to the user's Claude plan. No API key needed. The
    CLI can't set temperature or a seed, so reproducibility comes from the cache.
  * ``Ollama``: a local open-weight model over Ollama's HTTP API, with
    temperature and seed.
  * ``MockLLM``: deterministic responses for tests.

``CachedLLM`` wraps any backend in a SQLite cache keyed by
``(backend name, system, prompt, params)``. With ``offline=True`` a cache miss
raises instead of calling the model, which proves a result can be rebuilt from
the published cache alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


class LLM(Protocol):
    name: str

    def complete(self, prompt: str, system: str = "", **params) -> Completion: ...


class CacheMiss(RuntimeError):
    pass


class ClaudeCLI:
    """``claude -p`` with no tools, no MCP servers, no settings and no saved session.

    It runs in an empty temporary directory so that no project ``CLAUDE.md`` or
    settings leak into the prompt. ``cost_usd`` is the CLI's notional API price;
    on a subscription plan it measures usage, it is not billed.
    """

    def __init__(
        self, model: str = "claude-sonnet-5-5", binary: str = "claude", timeout: int = 300
    ):
        self.model, self.binary, self.timeout = model, binary, timeout
        self.name = f"claude-cli:{model}"

    def complete(self, prompt: str, system: str = "", **params) -> Completion:
        if os.environ.get("CLAUDECODE"):
            raise RuntimeError(
                "ClaudeCLI can't run inside a Claude Code session. Run this command from a "
                "regular terminal, or use --offline to read cached responses only."
            )
        if shutil.which(self.binary) is None:
            raise RuntimeError(f"{self.binary!r} not found on PATH; install Claude Code first")
        cmd = [
            self.binary, "-p",
            "--model", self.model,
            "--output-format", "json",
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--setting-sources", "",
        ]  # fmt: skip
        if system:
            cmd += ["--system-prompt", system]
        if params.get("effort"):
            cmd += ["--effort", params["effort"]]
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout, cwd=cwd
            )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
        return self.parse(proc.stdout, self.model)

    @staticmethod
    def parse(stdout: str, model: str) -> Completion:
        out = json.loads(stdout)
        if out.get("is_error") or out.get("subtype", "success") != "success":
            raise RuntimeError(f"claude -p returned an error: {str(out.get('result'))[:500]}")
        usage = out.get("usage") or {}
        models = list((out.get("modelUsage") or {}).keys())
        return Completion(
            text=out.get("result", ""),
            model=models[0] if len(models) == 1 else model,
            input_tokens=int(usage.get("input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cost_usd=float(out.get("total_cost_usd") or 0.0),
        )


class Ollama:
    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model, self.host = model, host.rstrip("/")
        self.name = f"ollama:{model}"

    def complete(self, prompt: str, system: str = "", **params) -> Completion:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": params.get("temperature", 0.0),
                "seed": params.get("seed", 0),
            },
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        return Completion(
            text=out["message"]["content"],
            model=self.model,
            input_tokens=out.get("prompt_eval_count", 0),
            output_tokens=out.get("eval_count", 0),
        )


class MockLLM:
    def __init__(self, respond: Callable[[str, str], str], name: str = "mock"):
        self.respond, self.name = respond, name
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt: str, system: str = "", **params) -> Completion:
        self.calls.append((system, prompt))
        return Completion(text=self.respond(system, prompt), model=self.name)


@dataclass
class UsageStats:
    calls: int = 0
    hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    models: set[str] = field(default_factory=set)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["models"] = sorted(self.models)
        return d


class CachedLLM:
    def __init__(self, inner: LLM, path: Path, offline: bool = False):
        self.inner, self.offline = inner, offline
        self.name = inner.name
        self.stats = UsageStats()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, backend TEXT, system TEXT, prompt TEXT,"
            " params TEXT, completion TEXT, created_at TEXT)"
        )
        self._db.commit()

    @staticmethod
    def key(backend: str, system: str, prompt: str, params: dict) -> str:
        blob = json.dumps([backend, system, prompt, params], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, prompt: str, system: str = "", **params) -> Completion:
        k = self.key(self.inner.name, system, prompt, params)
        with self._lock:
            row = self._db.execute(
                "SELECT completion FROM responses WHERE key = ?", (k,)
            ).fetchone()
        if row is not None:
            c = Completion(**{**json.loads(row[0]), "cached": True})
            self._account(c, hit=True)
            return c
        if self.offline:
            raise CacheMiss(f"no cached response for key {k[:12]} ({self.inner.name})")
        c = self.inner.complete(prompt, system, **params)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    k,
                    self.inner.name,
                    system,
                    prompt,
                    json.dumps(params, sort_keys=True),
                    json.dumps(asdict(c)),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            self._db.commit()
        self._account(c, hit=False)
        return c

    def _account(self, c: Completion, hit: bool) -> None:
        with self._lock:
            s = self.stats
            s.calls += 1
            s.hits += hit
            s.input_tokens += c.input_tokens
            s.output_tokens += c.output_tokens
            s.cost_usd += c.cost_usd
            s.models.add(c.model)


def make_llm(backend: str, model: str | None, cache: Path, offline: bool = False) -> CachedLLM:
    if backend == "claude-cli":
        inner: LLM = ClaudeCLI(model or "claude-sonnet-5-5")
    elif backend == "ollama":
        inner = Ollama(model or "qwen2.5:7b")
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return CachedLLM(inner, cache, offline=offline)


def extract_json(text: str) -> dict:
    """The first JSON object in ``text``, tolerating code fences and surrounding prose."""
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object in response: {text[:200]!r}")
