"""Tests for frankenbote.curator — pure helpers plus curate() against a scripted LLM client."""

import json
import textwrap

import pytest

from frankenbote.curator import (
    _build_user_prompt,
    _merge_decisions,
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

_VALID_SECTIONS = textwrap.dedent("""\
    curator:
      guidance: Test guidance.
      priorities:
        - id: P1
          label: Lokal
          description: Local news
      sections:
        - id: kultur
          display_name: Kultur
          description: Culture events
""")


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
        cfg.write_text(_VALID_SECTIONS, encoding="utf-8")
        result = load_curator_config(cfg)
        assert result.guidance == "Test guidance."
        assert len(result.priorities) == 1
        assert len(result.sections) == 1
        assert not hasattr(result, "model")

    def test_stale_curator_model_key_raises(self, tmp_path):
        cfg = tmp_path / "sections.yaml"
        cfg.write_text(
            _VALID_SECTIONS.replace(
                "curator:\n", "curator:\n  model: claude-sonnet-4-6\n"
            ),
            encoding="utf-8",
        )
        with pytest.raises(
            ValueError, match="'curator.model' has moved to config/config.yaml"
        ):
            load_curator_config(cfg)

    def test_stale_summarizer_block_raises(self, tmp_path):
        cfg = tmp_path / "sections.yaml"
        cfg.write_text(
            _VALID_SECTIONS + "summarizer:\n  model: claude-haiku-4-5\n",
            encoding="utf-8",
        )
        with pytest.raises(
            ValueError, match="'summarizer:' block has moved to config/config.yaml"
        ):
            load_curator_config(cfg)


# ── _merge_decisions ─────────────────────────────────────────────────────────


class TestMergeDecisions:
    def _make_decision(
        self, idx: int, section: str | None = "politik"
    ) -> CuratorDecision:
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


# ── curate() ─────────────────────────────────────────────────────────────────


def _decision(
    idx: int, section="politik_verwaltung", priority="P1", score=8.0, rationale="Local."
):
    return {
        "article_index": idx,
        "section": section,
        "priority": priority,
        "relevance_score": score,
        "rationale": rationale,
    }


class TestCurate:
    def test_empty_candidates_skip_the_client(self):
        client = ScriptedLLMClient()
        assert curate([], make_curator_config(), client) == []
        assert client.calls == []

    def test_builds_request_and_merges_decisions(self):
        config = make_curator_config()
        candidates = [
            make_article(title="A"),
            make_article(title="B", link="https://example.com/b"),
        ]
        client = ScriptedLLMClient(
            [
                [
                    tool_result(
                        "curator",
                        {"decisions": [_decision(0), _decision(1, section=None)]},
                    )
                ]
            ]
        )

        curated = curate(candidates, config, client)

        assert [c.article.title for c in curated] == ["A", "B"]
        assert curated[0].section == "politik_verwaltung"
        assert curated[1].section is None
        # Request shape handed to the client: task-addressed, no model.
        name, requests = client.calls[0]
        assert name == "submit_batch"
        [request] = requests
        assert request["task"] == "curator"
        assert request["custom_id"] == "curator"
        params = request["params"]
        assert "model" not in params
        assert params["tool"]["name"] == "submit_decisions"
        assert params["max_tokens"] == 500 + 150 * 2
        assert '<article index="0"' in params["user_prompt"]
        assert "UNTRUSTED INPUT" in params["system"]

    def test_tool_schema_enumerates_configured_sections(self):
        client = ScriptedLLMClient(
            [[tool_result("curator", {"decisions": [_decision(0)]})]]
        )
        curate([make_article()], make_curator_config(), client)
        [request] = client.calls[0][1]
        schema = request["params"]["tool"]["input_schema"]
        section = schema["$defs"]["CuratorDecision"]["properties"]["section"]
        assert {
            "enum": ["politik_verwaltung", "wirtschaft", "kultur"],
            "type": "string",
        } in section["anyOf"]

    def test_client_batch_off_uses_sync_call(self):
        client = ScriptedLLMClient(
            [tool_result("curator", {"decisions": [_decision(0)]})], use_batch=False
        )
        curate([make_article()], make_curator_config(), client)
        assert [n for n, _ in client.calls] == ["call_tool"]

    def test_decisions_as_json_string_are_normalised(self):
        client = ScriptedLLMClient(
            [tool_result("curator", {"decisions": json.dumps([_decision(0)])})],
            use_batch=False,
        )
        curated = curate([make_article()], make_curator_config(), client)
        assert curated[0].section == "politik_verwaltung"

    def test_unknown_section_is_a_validation_failure(self, monkeypatch):
        import frankenbote.llm.base as base_module

        monkeypatch.setattr(base_module, "save_failure", lambda *a: "debug.txt")
        bad = tool_result(
            "curator", {"decisions": [_decision(0, section="not_configured")]}
        )
        client = ScriptedLLMClient([bad, bad], use_batch=False)
        with pytest.raises(
            RuntimeError, match="Curator tool output invalid after retry"
        ):
            curate([make_article()], make_curator_config(), client)

    def test_retries_once_then_succeeds(self):
        client = ScriptedLLMClient(
            [
                LLMTransientError("net"),
                tool_result("curator", {"decisions": [_decision(0)]}),
            ],
            use_batch=False,
        )
        curated = curate([make_article()], make_curator_config(), client)
        assert len(curated) == 1
        assert len(client.calls) == 2

    def test_persistent_failure_raises_runtime_error(self, monkeypatch):
        import frankenbote.llm.base as base_module

        monkeypatch.setattr(base_module, "save_failure", lambda *a: "debug.txt")
        bad = tool_result("curator", None, "max_tokens")
        client = ScriptedLLMClient([bad, bad], use_batch=False)
        with pytest.raises(RuntimeError, match="Curator failed twice"):
            curate([make_article()], make_curator_config(), client)
