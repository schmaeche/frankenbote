"""Tests for frankenbote.curator — pure helpers plus curate() against a scripted LLM client."""

import json

import pytest

import textwrap

from frankenbote.curator import (
    _build_curator_tool,
    _build_user_prompt,
    _merge_decisions,
    _normalize_tool_input,
    curate,
    load_curator_config,
)
from frankenbote.llm import LLMTransientError
from frankenbote.models import CuratorDecision, Priority
from tests.conftest import (
    ScriptedLLMClient,
    make_article,
    make_curator_config,
    tool_result,
)


# ── load_curator_config ───────────────────────────────────────────────────────

class TestLoadCuratorConfig:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Sections config not found"):
            load_curator_config(tmp_path / "missing.yaml")

    def test_yaml_not_a_dict_raises(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a top-level 'curator:' key"):
            load_curator_config(cfg)

    def test_missing_curator_key_raises(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("other_key: something\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a top-level 'curator:' key"):
            load_curator_config(cfg)

    def test_valid_yaml_returns_curator_config(self, tmp_path):
        cfg = tmp_path / "sections.yaml"
        cfg.write_text(
            textwrap.dedent("""\
                curator:
                  model: claude-sonnet-4-6
                  guidance: Test guidance.
                  priorities:
                    - id: P1
                      label: Lokal
                      description: Local news
                  sections:
                    - id: kultur
                      display_name: Kultur
                      description: Culture events
            """),
            encoding="utf-8",
        )
        result = load_curator_config(cfg)
        assert result.model == "claude-sonnet-4-6"
        assert result.guidance == "Test guidance."
        assert len(result.priorities) == 1
        assert len(result.sections) == 1


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


# ── curate() ─────────────────────────────────────────────────────────────────

def _decision(idx: int, section="politik", priority="P1", score=8.0, rationale="Local."):
    return {
        "article_index": idx, "section": section, "priority": priority,
        "relevance_score": score, "rationale": rationale,
    }


class TestCurate:
    def test_empty_candidates_skip_the_client(self):
        assert curate([], make_curator_config(), client=ScriptedLLMClient()) == []

    def test_builds_request_and_merges_decisions(self):
        config = make_curator_config(model="claude-test")
        candidates = [make_article(title="A"), make_article(title="B", link="https://example.com/b")]
        client = ScriptedLLMClient([
            [tool_result("curator", {"decisions": [_decision(0), _decision(1, section=None)]})]
        ])

        curated = curate(candidates, config, client=client)

        assert [c.article.title for c in curated] == ["A", "B"]
        assert curated[0].section == "politik"
        assert curated[1].section is None
        # Request shape handed to the client.
        name, requests = client.calls[0]
        assert name == "submit_batch"
        [request] = requests
        assert request["custom_id"] == "curator"
        params = request["params"]
        assert params["model"] == "claude-test"
        assert params["tool"]["name"] == "submit_decisions"
        assert params["max_tokens"] == 500 + 150 * 2
        assert "<article index=\"0\"" in params["user_prompt"]
        assert "UNTRUSTED INPUT" in params["system"]

    def test_batch_off_uses_sync_call(self):
        client = ScriptedLLMClient([tool_result("curator", {"decisions": [_decision(0)]})])
        curate([make_article()], make_curator_config(), use_batch=False, client=client)
        assert [n for n, _ in client.calls] == ["call_tool"]

    def test_decisions_as_json_string_are_normalised(self):
        client = ScriptedLLMClient([
            tool_result("curator", {"decisions": json.dumps([_decision(0)])})
        ])
        curated = curate([make_article()], make_curator_config(), use_batch=False, client=client)
        assert curated[0].section == "politik"

    def test_retries_once_then_succeeds(self):
        client = ScriptedLLMClient([
            LLMTransientError("net"),
            tool_result("curator", {"decisions": [_decision(0)]}),
        ])
        curated = curate([make_article()], make_curator_config(), use_batch=False, client=client)
        assert len(curated) == 1
        assert len(client.calls) == 2

    def test_persistent_failure_raises_runtime_error(self, monkeypatch):
        import frankenbote.llm.base as base_module
        monkeypatch.setattr(base_module, "save_failure", lambda *a: "debug.txt")
        bad = tool_result("curator", None, "max_tokens")
        client = ScriptedLLMClient([bad, bad])
        with pytest.raises(RuntimeError, match="Curator failed twice"):
            curate([make_article()], make_curator_config(), use_batch=False, client=client)

    def test_missing_api_key_without_client_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            curate([make_article()], make_curator_config())
