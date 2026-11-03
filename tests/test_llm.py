import json

import pytest

from supersede.llm import CachedLLM, CacheMiss, ClaudeCLI, LLMError, MockLLM, extract_json


def test_cache_hits_and_offline(tmp_path):
    mock = MockLLM(lambda system, prompt: prompt.upper())
    llm = CachedLLM(mock, tmp_path / "c.sqlite")
    assert llm.complete("hi", "sys").text == "HI"
    again = llm.complete("hi", "sys")
    assert again.cached and len(mock.calls) == 1
    llm.complete("hi", "sys", sample=1)  # different params, different key
    assert len(mock.calls) == 2

    offline = CachedLLM(MockLLM(lambda s, p: "never"), tmp_path / "c.sqlite", offline=True)
    assert offline.complete("hi", "sys").text == "HI"
    with pytest.raises(CacheMiss):
        offline.complete("unseen", "sys")
    assert offline.stats.hits == 1


def test_cache_is_keyed_on_backend_name(tmp_path):
    a = CachedLLM(MockLLM(lambda s, p: "a", name="m1"), tmp_path / "c.sqlite")
    b = CachedLLM(MockLLM(lambda s, p: "b", name="m2"), tmp_path / "c.sqlite")
    assert a.complete("x").text == "a"
    assert b.complete("x").text == "b"


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    assert extract_json('{broken {"ok": true}') == {"ok": True}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_claude_cli_parse():
    stdout = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": '{"ok": true}',
            "total_cost_usd": 0.0123,
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 7},
            "modelUsage": {"claude-sonnet-5-5": {}},
        }
    )
    c = ClaudeCLI.parse(stdout, "fallback")
    assert (c.text, c.model, c.input_tokens, c.output_tokens) == (
        '{"ok": true}',
        "claude-sonnet-5-5",
        15,
        7,
    )
    with pytest.raises(LLMError) as e:
        ClaudeCLI.parse(json.dumps({"is_error": True, "result": "API Error: rate limit"}), "m", 1)
    assert e.value.transient
    version = json.dumps({"is_error": True, "result": "400 invalid_request_error: version too old"})
    with pytest.raises(LLMError, match="version too old") as e:
        ClaudeCLI.parse(version, "m", 1)
    assert not e.value.transient
    with pytest.raises(LLMError, match="boom"):
        ClaudeCLI.parse("", "m", 1, stderr="boom")


def test_claude_cli_refuses_inside_claude_code(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(RuntimeError, match="regular terminal"):
        ClaudeCLI().complete("hi")
