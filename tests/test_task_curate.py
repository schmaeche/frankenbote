"""Closed-loop tests for the curator task: inputs → prompt, tool input → decisions."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from frankenbote.llm import ItemNote
from frankenbote.llm.tasks import curator_task
from frankenbote.models import CuratorDecision, CuratorResponse, Priority
from tests.conftest import make_article, make_curator_config

PROMPTS = Path(__file__).parent / "fixtures" / "prompts"


def _decision(idx: int, section: str | None = "politik_verwaltung") -> dict:
    return {
        "article_index": idx,
        "section": section,
        "priority": "P1",
        "relevance_score": 7.0,
        "rationale": "Test rationale.",
    }


# ── identity and schema ──────────────────────────────────────────────────────

class TestIdentity:
    def test_identity(self):
        task = curator_task(make_curator_config())
        assert task.name == "curator"
        assert task.label == "Curator"
        assert task.tool_name == "submit_decisions"
        assert "submit_decisions" in task.system_prompt
        assert "UNTRUSTED INPUT" in task.system_prompt

    def test_schema_restricts_section_to_configured_ids(self):
        config = make_curator_config(
            sections=[
                {"id": "politik", "display_name": "P", "description": "d"},
                {"id": "sport", "display_name": "S", "description": "d"},
            ]
        )
        schema = curator_task(config).tool["input_schema"]
        section = schema["$defs"]["CuratorDecision"]["properties"]["section"]
        assert {"enum": ["politik", "sport"], "type": "string"} in section["anyOf"]
        assert {"type": "null"} in section["anyOf"]

    def test_schema_shape(self):
        schema = curator_task(make_curator_config()).tool["input_schema"]
        assert schema["required"] == ["decisions"]
        assert schema["additionalProperties"] is False
        decision = schema["$defs"]["CuratorDecision"]
        assert decision["required"] == [
            "article_index", "section", "priority", "relevance_score", "rationale"
        ]
        assert decision["additionalProperties"] is False
        assert decision["properties"]["rationale"] == {"maxLength": 300, "type": "string"}
        assert schema["$defs"]["Priority"]["enum"] == ["P1", "P2", "P3", "P4"]
        assert "title" not in decision

    def test_max_tokens_formula(self):
        task = curator_task(make_curator_config())
        assert task.max_tokens_for(1) == 650
        assert task.max_tokens_for(10) == 2000
        assert task.max_tokens_for(1_000_000) == 48000

    def test_requires_at_least_one_section(self):
        with pytest.raises(ValueError, match="at least one configured section"):
            curator_task(make_curator_config(sections=[]))


# ── parse ────────────────────────────────────────────────────────────────────

class TestParse:
    def test_accepts_known_section_and_null(self):
        task = curator_task(make_curator_config())
        resp = task.parse({"decisions": [
            _decision(0, "politik_verwaltung"),
            _decision(1, None),
        ]})
        assert isinstance(resp, CuratorResponse)
        assert all(isinstance(d, CuratorDecision) for d in resp.decisions)
        assert resp.decisions[0].section == "politik_verwaltung"
        assert resp.decisions[1].section is None

    def test_rejects_unknown_section(self):
        with pytest.raises(ValidationError):
            curator_task(make_curator_config()).parse(
                {"decisions": [_decision(0, "not_configured")]}
            )

    def test_normalises_json_string(self):
        resp = curator_task(make_curator_config()).parse(
            {"decisions": json.dumps([_decision(0)])}
        )
        assert resp.decisions[0].priority is Priority.P1


# ── render: inputs → prompt ──────────────────────────────────────────────────

class TestRender:
    def test_matches_the_golden_prompt(self):
        """The rendered prompt is byte-identical to the recorded one.

        Update the fixture deliberately — a diff here means the model sees
        something different.
        """
        candidates = [
            make_article(
                source_name="Nordbayern",
                title="Stadtrat beschließt neuen Haushalt",
                summary="Der Nürnberger Stadtrat hat den Haushalt für 2027 verabschiedet.",
                link="https://example.com/a",
            ),
            make_article(
                source_name="BR24",
                title="Kein Vorspann vorhanden",
                summary="",
                link="https://example.com/b",
            ),
        ]
        rendered = curator_task(make_curator_config()).render(candidates)
        assert rendered == (PROMPTS / "curate.txt").read_text(encoding="utf-8")

    def test_empty_feed_summary_gets_a_placeholder(self):
        rendered = curator_task(make_curator_config()).render(
            [make_article(summary="")]
        )
        assert "<summary>(no summary)</summary>" in rendered

    def test_untrusted_framing_is_present(self):
        rendered = curator_task(make_curator_config()).render([make_article()])
        assert "untrusted data" in rendered


# ── interpret: tool input → aligned decisions ────────────────────────────────

class TestInterpret:
    def _interpret(self, n_articles: int, decisions: list[dict]):
        task = curator_task(make_curator_config())
        articles = [
            make_article(link=f"https://example.com/{i}") for i in range(n_articles)
        ]
        return task.interpret(task.parse({"decisions": decisions}), articles)

    def test_decision_matched_by_index(self):
        outcome = self._interpret(1, [_decision(0, "wirtschaft")])
        assert outcome.values[0].section == "wirtschaft"
        assert outcome.notes == []

    def test_index_order_independent(self):
        outcome = self._interpret(
            2, [_decision(1, "kultur"), _decision(0, "wirtschaft")]
        )
        assert [d.section for d in outcome.values] == ["wirtschaft", "kultur"]

    def test_output_length_equals_input_length(self):
        outcome = self._interpret(5, [_decision(i) for i in range(3)])
        assert len(outcome.values) == 5

    def test_missing_decision_gets_the_sentinel(self):
        outcome = self._interpret(1, [])
        [decision] = outcome.values
        assert decision.section is None
        assert decision.priority is Priority.P4
        assert decision.relevance_score == 0.0
        assert "no decision" in decision.rationale.lower()

    def test_missing_decision_is_noted(self):
        outcome = self._interpret(2, [_decision(0)])
        assert outcome.notes == [ItemNote(1, "no decision returned")]

    def test_section_none_decision_preserved(self):
        outcome = self._interpret(1, [_decision(0, None)])
        assert outcome.values[0].section is None
        assert outcome.notes == []
