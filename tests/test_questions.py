import json
from pathlib import Path

from supersede.bench.questions import build, validate
from supersede.bench.rfc import build as build_rfc
from supersede.corpus import Doc
from supersede.llm import CachedLLM, MockLLM

FIXTURE = Path(__file__).parent / "fixtures" / "rfc-index-mini.xml"
OLD = Doc("RFC1", "Widget Transfer Protocol", abstract="Widgets use port 70 over UDP.")
NEW = Doc("RFC2", "Widget Transfer Protocol v2", abstract="Widgets now use port 7070 over TCP.")
GOOD = {
    "change_type": "mechanism",
    "question": "Which transport does the widget transfer protocol use?",
    "answer_current": "TCP",
    "answer_stale": "UDP",
    "evidence_current": "port 7070 over  TCP",
    "evidence_stale": "Widgets use port 70 over UDP.",
}


def test_validate_accepts_grounded_item():
    assert validate(GOOD, OLD, NEW) is None


def test_validate_rejections():
    assert validate({"skip": "no change"}, OLD, NEW) == "skipped_by_model"
    assert validate({**GOOD, "answer_stale": ""}, OLD, NEW) == "missing_field"
    bad = {**GOOD, "evidence_current": "port 8080"}
    assert validate(bad, OLD, NEW) == "evidence_current_not_verbatim"
    assert validate({**GOOD, "evidence_stale": "TCP"}, OLD, NEW) == "evidence_stale_not_verbatim"
    assert validate({**GOOD, "answer_stale": " tcp"}, OLD, NEW) == "answers_identical"
    leak = {**GOOD, "question": "Which transport does RFC 1 use?"}
    assert validate(leak, OLD, NEW) == "question_leaks_reference"
    assert validate({**GOOD, "change_type": "editorial"}, OLD, NEW) == "not_substantive"
    reveal = {**GOOD, "question": "Does the widget transfer protocol use TCP or UDP?"}
    assert validate(reveal, OLD, NEW) == "question_reveals_answer"
    silent = {**GOOD, "answer_stale": "None; the abstract does not say."}
    assert validate(silent, OLD, NEW) == "answer_from_silence"


def test_build_end_to_end_with_mock(tmp_path):
    build_rfc(FIXTURE, tmp_path)

    def respond(system, prompt):
        # Only the RFC1 -> RFC2 pair has abstracts on both sides.
        assert "Widget Transfer Protocol" in prompt
        return (
            "```json\n"
            + json.dumps(
                {
                    "change_type": "scope",
                    "question": "How many framing changes does the widget transfer protocol make?",
                    "answer_current": "It adds framing.",
                    "answer_stale": "No framing.",
                    "evidence_current": "Adds framing.",
                    "evidence_stale": "Defines the widget transfer protocol.",
                }
            )
            + "\n```"
        )

    llm = CachedLLM(MockLLM(respond), tmp_path / "cache.sqlite")
    questions, stats = build(tmp_path, llm, limit=None, workers=2)
    assert stats["n_candidate_edges"] == 1
    assert stats["n_accepted"] == 1 and stats["acceptance_rate"] == 1.0
    q = questions[0]
    assert q["current"] == ["RFC3"] and q["stale"] == ["RFC1", "RFC2"]
    assert q["variant"] == "change_aware" and q["provenance"]["prompt_version"] == "q-v3"

    offline = CachedLLM(MockLLM(lambda s, p: "unused"), tmp_path / "cache.sqlite", offline=True)
    again, stats2 = build(tmp_path, offline, limit=None)
    assert again == questions and stats2["usage"]["hits"] == 1
