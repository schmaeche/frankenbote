"""Tests for frankenbote.summarizer — pure helpers only (no API calls)."""

import json

import pytest

from frankenbote.summarizer import (
    SummarizerConfig,
    _build_user_prompt,
    _normalize_tool_input,
    load_summarizer_config,
)
from tests.conftest import make_curated


# ── _normalize_tool_input ────────────────────────────────────────────────────

class TestNormalizeToolInput:
    def test_list_passthrough(self):
        summaries_list = [{"article_index": 0, "summary": "A summary."}]
        tool_input = {"summaries": summaries_list}
        result = _normalize_tool_input(tool_input)
        assert result["summaries"] is summaries_list

    def test_json_string_is_parsed_to_list(self):
        summaries_list = [{"article_index": 0, "summary": "A summary."}]
        tool_input = {"summaries": json.dumps(summaries_list)}
        result = _normalize_tool_input(tool_input)
        assert isinstance(result["summaries"], list)
        assert result["summaries"][0]["article_index"] == 0

    def test_invalid_json_string_raises_value_error(self):
        tool_input = {"summaries": "not valid json {{{"}
        with pytest.raises(ValueError, match="not valid JSON"):
            _normalize_tool_input(tool_input)

    def test_json_string_that_is_not_list_raises(self):
        tool_input = {"summaries": json.dumps({"not": "a list"})}
        with pytest.raises(ValueError):
            _normalize_tool_input(tool_input)

    def test_other_keys_preserved(self):
        tool_input = {"summaries": [], "extra": "data"}
        result = _normalize_tool_input(tool_input)
        assert result["extra"] == "data"


# ── _build_user_prompt ───────────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_prompt_contains_article_count(self):
        articles = [
            make_curated(article=__import__('tests.conftest', fromlist=['make_article']).make_article(
                link=f"https://example.com/{i}"
            ))
            for i in range(4)
        ]
        prompt = _build_user_prompt(articles)
        assert "4 Einträge erwartet" in prompt

    def test_prompt_contains_index_zero(self):
        articles = [make_curated()]
        prompt = _build_user_prompt(articles)
        assert 'index="0"' in prompt

    def test_lead_attribute_appears(self):
        lead_article = make_curated(is_lead=True)
        prompt = _build_user_prompt([lead_article])
        assert 'is_lead="true"' in prompt

    def test_non_lead_attribute_appears(self):
        non_lead = make_curated(is_lead=False)
        prompt = _build_user_prompt([non_lead])
        assert 'is_lead="false"' in prompt

    def test_article_title_in_prompt(self):
        from tests.conftest import make_article
        article = make_curated(article=make_article(title="Unique Headline ABC"))
        prompt = _build_user_prompt([article])
        assert "Unique Headline ABC" in prompt

    def test_count_in_closing_line_matches_input(self):
        from tests.conftest import make_article
        articles = [
            make_curated(article=make_article(link=f"https://example.com/{i}"))
            for i in range(7)
        ]
        prompt = _build_user_prompt(articles)
        assert "7 Einträge erwartet" in prompt


# ── SummarizerConfig ─────────────────────────────────────────────────────────

class TestSummarizerConfig:
    def test_valid_model_string(self):
        cfg = SummarizerConfig(model="claude-sonnet-4-6")
        assert cfg.model == "claude-sonnet-4-6"

    def test_missing_model_raises(self):
        with pytest.raises(Exception):
            SummarizerConfig()


# ── load_summarizer_config ───────────────────────────────────────────────────

class TestLoadSummarizerConfig:
    def test_loads_model_from_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "sections.yaml"
        cfg_file.write_text("summarizer:\n  model: claude-haiku-4-5\n", encoding="utf-8")
        cfg = load_summarizer_config(cfg_file)
        assert cfg.model == "claude-haiku-4-5"

    def test_missing_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_summarizer_config(tmp_path / "nonexistent.yaml")

    def test_yaml_without_summarizer_key_raises(self, tmp_path):
        cfg_file = tmp_path / "sections.yaml"
        cfg_file.write_text("curator:\n  model: claude-sonnet-4-6\n", encoding="utf-8")
        with pytest.raises(ValueError, match="summarizer"):
            load_summarizer_config(cfg_file)

    def test_non_dict_yaml_raises(self, tmp_path):
        cfg_file = tmp_path / "sections.yaml"
        cfg_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="summarizer"):
            load_summarizer_config(cfg_file)

    def test_accepts_path_as_string(self, tmp_path):
        cfg_file = tmp_path / "sections.yaml"
        cfg_file.write_text("summarizer:\n  model: claude-opus-4-7\n", encoding="utf-8")
        cfg = load_summarizer_config(str(cfg_file))
        assert cfg.model == "claude-opus-4-7"
