"""Tests for frankenbote.curator — pure helpers only (no API calls)."""

import json
from types import SimpleNamespace

import pytest

from frankenbote.curator import (
    _build_curator_tool,
    _build_user_prompt,
    _extract_curator_result,
    _merge_decisions,
    _normalize_tool_input,
)
from frankenbote.models import CuratorDecision, Priority
from tests.conftest import make_article, make_curator_config


# ── helpers for building mock batch results ──────────────────────────────────

def _make_succeeded_result(custom_id: str, tool_input: dict) -> SimpleNamespace:
    """Minimal mock of MessageBatchIndividualResponse with a succeeded tool_use."""
    tool_block = SimpleNamespace(type="tool_use", name="submit_decisions", input=tool_input)
    message = SimpleNamespace(content=[tool_block])
    result = SimpleNamespace(type="succeeded", message=message)
    return SimpleNamespace(custom_id=custom_id, result=result)


def _make_failed_result(custom_id: str, result_type: str) -> SimpleNamespace:
    """Minimal mock for errored / expired / canceled batch results."""
    result = SimpleNamespace(type=result_type)
    return SimpleNamespace(custom_id=custom_id, result=result)


# ── _normalize_tool_input ────────────────────────────────────────────────────

class TestNormalizeToolInput:
    def test_list_passthrough(self):
        decisions_list = [{"article_index": 0, "section": "politik", "priority": "P1",
                           "relevance_score": 7.0, "rationale": "ok"}]
        tool_input = {"decisions": decisions_list}
        result = _normalize_tool_input(tool_input)
        assert result["decisions"] is decisions_list

    def test_json_string_is_parsed_to_list(self):
        decisions_list = [{"article_index": 0, "section": "politik", "priority": "P1",
                           "relevance_score": 7.0, "rationale": "ok"}]
        tool_input = {"decisions": json.dumps(decisions_list)}
        result = _normalize_tool_input(tool_input)
        assert isinstance(result["decisions"], list)
        assert result["decisions"][0]["article_index"] == 0

    def test_invalid_json_string_raises_value_error(self):
        tool_input = {"decisions": "not valid json {{{"}
        with pytest.raises(ValueError, match="not valid JSON"):
            _normalize_tool_input(tool_input)

    def test_json_string_that_is_not_list_raises(self):
        tool_input = {"decisions": json.dumps({"not": "a list"})}
        with pytest.raises(ValueError):
            _normalize_tool_input(tool_input)

    def test_other_keys_preserved(self):
        tool_input = {"decisions": [], "extra_key": "value"}
        result = _normalize_tool_input(tool_input)
        assert result["extra_key"] == "value"


# ── _merge_decisions ─────────────────────────────────────────────────────────

class TestMergeDecisions:
    def _make_decision(self, idx: int, section: str | None = "politik") -> CuratorDecision:
        return CuratorDecision(
            article_index=idx,
            section=section,
            priority=Priority.P1,
            relevance_score=7.0,
            rationale="Test rationale.",
        )

    def test_article_matched_by_index(self):
        articles = [make_article(link="https://example.com/0")]
        decisions = [self._make_decision(0, section="wirtschaft")]
        merged = _merge_decisions(articles, decisions)
        assert merged[0].section == "wirtschaft"

    def test_missing_decision_gets_sentinel(self):
        articles = [make_article()]
        merged = _merge_decisions(articles, [])  # no decisions returned
        assert merged[0].section is None
        assert merged[0].priority == Priority.P4
        assert merged[0].relevance_score == 0.0

    def test_missing_decision_rationale_indicates_missing(self):
        articles = [make_article()]
        merged = _merge_decisions(articles, [])
        assert "no decision" in merged[0].rationale.lower()

    def test_index_order_independent(self):
        # Decision for index 1 comes before decision for index 0
        articles = [
            make_article(link="https://example.com/0"),
            make_article(link="https://example.com/1"),
        ]
        decisions = [
            self._make_decision(1, section="kultur"),
            self._make_decision(0, section="wirtschaft"),
        ]
        merged = _merge_decisions(articles, decisions)
        assert merged[0].section == "wirtschaft"
        assert merged[1].section == "kultur"

    def test_output_length_equals_input_length(self):
        articles = [make_article(link=f"https://example.com/{i}") for i in range(5)]
        decisions = [self._make_decision(i) for i in range(3)]  # partial decisions
        merged = _merge_decisions(articles, decisions)
        assert len(merged) == 5

    def test_section_none_decision_preserved(self):
        articles = [make_article()]
        decisions = [self._make_decision(0, section=None)]
        merged = _merge_decisions(articles, decisions)
        assert merged[0].section is None


# ── _build_user_prompt ───────────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_prompt_contains_article_index(self):
        articles = [make_article(title="Test Article")]
        config = make_curator_config()
        prompt = _build_user_prompt(articles, config)
        assert 'index="0"' in prompt

    def test_prompt_contains_expected_count(self):
        articles = [make_article(link=f"https://example.com/{i}") for i in range(3)]
        config = make_curator_config()
        prompt = _build_user_prompt(articles, config)
        assert "3 decisions expected" in prompt

    def test_prompt_contains_section_ids(self):
        articles = [make_article()]
        config = make_curator_config()
        prompt = _build_user_prompt(articles, config)
        assert "politik_verwaltung" in prompt
        assert "wirtschaft" in prompt

    def test_prompt_contains_article_title(self):
        articles = [make_article(title="Unique Title XYZ")]
        config = make_curator_config()
        prompt = _build_user_prompt(articles, config)
        assert "Unique Title XYZ" in prompt


# ── _build_curator_tool ──────────────────────────────────────────────────────

class TestBuildCuratorTool:
    def test_tool_name_is_submit_decisions(self):
        tool = _build_curator_tool(["politik", "wirtschaft"])
        assert tool["name"] == "submit_decisions"

    def test_section_ids_appear_in_enum(self):
        tool = _build_curator_tool(["politik", "wirtschaft"])
        section_enum = (
            tool["input_schema"]["properties"]["decisions"]["items"]
            ["properties"]["section"]["enum"]
        )
        assert "politik" in section_enum
        assert "wirtschaft" in section_enum

    def test_null_in_section_enum(self):
        tool = _build_curator_tool(["politik"])
        section_enum = (
            tool["input_schema"]["properties"]["decisions"]["items"]
            ["properties"]["section"]["enum"]
        )
        assert None in section_enum


# ── _extract_curator_result ──────────────────────────────────────────────────

class TestExtractCuratorResult:
    _TOOL_INPUT = {"decisions": [{"article_index": 0, "section": "politik",
                                   "priority": "P1", "relevance_score": 8.0,
                                   "rationale": "Local story."}]}

    def test_succeeded_returns_tool_input_and_tool_use(self):
        results = [_make_succeeded_result("curator", self._TOOL_INPUT)]
        tool_input, stop_reason = _extract_curator_result(results)
        assert stop_reason == "tool_use"
        assert tool_input == self._TOOL_INPUT

    def test_errored_returns_none_and_errored(self):
        results = [_make_failed_result("curator", "errored")]
        tool_input, stop_reason = _extract_curator_result(results)
        assert tool_input is None
        assert stop_reason == "errored"

    def test_expired_returns_none_and_expired(self):
        results = [_make_failed_result("curator", "expired")]
        tool_input, stop_reason = _extract_curator_result(results)
        assert tool_input is None
        assert stop_reason == "expired"

    def test_canceled_returns_none_and_canceled(self):
        results = [_make_failed_result("curator", "canceled")]
        tool_input, stop_reason = _extract_curator_result(results)
        assert tool_input is None
        assert stop_reason == "canceled"

    def test_missing_custom_id_returns_no_result(self):
        results = [_make_succeeded_result("something-else", self._TOOL_INPUT)]
        tool_input, stop_reason = _extract_curator_result(results)
        assert tool_input is None
        assert stop_reason == "no_result"

    def test_empty_iterator_returns_no_result(self):
        tool_input, stop_reason = _extract_curator_result([])
        assert tool_input is None
        assert stop_reason == "no_result"

    def test_succeeded_but_no_matching_tool_block(self):
        wrong_block = SimpleNamespace(type="tool_use", name="other_tool", input={})
        message = SimpleNamespace(content=[wrong_block])
        result = SimpleNamespace(type="succeeded", message=message)
        item = SimpleNamespace(custom_id="curator", result=result)
        tool_input, stop_reason = _extract_curator_result([item])
        assert tool_input is None
        assert stop_reason == "no_tool_use_block"

    def test_other_custom_ids_before_curator_are_skipped(self):
        results = [
            _make_failed_result("unrelated", "errored"),
            _make_succeeded_result("curator", self._TOOL_INPUT),
        ]
        tool_input, stop_reason = _extract_curator_result(results)
        assert stop_reason == "tool_use"
        assert tool_input == self._TOOL_INPUT
