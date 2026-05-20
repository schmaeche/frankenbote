"""Tests for frankenbote.summarizer — pure helpers only (no API calls)."""

import json

import pytest

from frankenbote.summarizer import (
    SummarizerConfig,
    _WrapUpResponse,
    _build_user_prompt,
    _build_wrap_up_prompt,
    _normalize_tool_input,
    _select_body,
    load_summarizer_config,
)
from tests.conftest import make_article, make_curated


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

    def test_wrap_up_model_defaults_to_none(self):
        cfg = SummarizerConfig(model="claude-haiku-4-5")
        assert cfg.wrap_up_model is None

    def test_wrap_up_model_accepted(self):
        cfg = SummarizerConfig(
            model="claude-haiku-4-5", wrap_up_model="claude-sonnet-4-6"
        )
        assert cfg.wrap_up_model == "claude-sonnet-4-6"


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

    def test_loads_wrap_up_model_when_present(self, tmp_path):
        cfg_file = tmp_path / "sections.yaml"
        cfg_file.write_text(
            "summarizer:\n  model: claude-haiku-4-5\n"
            "  wrap_up_model: claude-sonnet-4-6\n",
            encoding="utf-8",
        )
        cfg = load_summarizer_config(cfg_file)
        assert cfg.wrap_up_model == "claude-sonnet-4-6"


# ── _select_body ─────────────────────────────────────────────────────────────

class TestSelectBody:
    def test_prefers_fetched_body(self):
        art = make_curated(article=make_article(summary="Feed snippet."))
        assert _select_body(art, "Full fetched article text.") == "Full fetched article text."

    def test_falls_back_to_feed_snippet_when_fetch_none(self):
        art = make_curated(article=make_article(summary="Feed snippet text."))
        assert _select_body(art, None) == "Feed snippet text."

    def test_falls_back_when_fetched_is_empty(self):
        art = make_curated(article=make_article(summary="Feed snippet text."))
        assert _select_body(art, "") == "Feed snippet text."

    def test_falls_back_when_fetched_is_whitespace(self):
        art = make_curated(article=make_article(summary="Feed snippet text."))
        assert _select_body(art, "   \n  ") == "Feed snippet text."

    def test_returns_none_when_both_empty(self):
        art = make_curated(article=make_article(summary=""))
        assert _select_body(art, None) is None

    def test_returns_none_when_both_whitespace(self):
        art = make_curated(article=make_article(summary="   "))
        assert _select_body(art, "  ") is None


# ── _build_wrap_up_prompt ────────────────────────────────────────────────────

class TestBuildWrapUpPrompt:
    def test_contains_title(self):
        art = make_curated(article=make_article(title="Unique Headline XYZ"))
        prompt = _build_wrap_up_prompt(art, "Body text here.")
        assert "Unique Headline XYZ" in prompt

    def test_contains_body(self):
        prompt = _build_wrap_up_prompt(make_curated(), "Distinctive body content 12345.")
        assert "Distinctive body content 12345." in prompt

    def test_contains_source_name(self):
        art = make_curated(article=make_article(source_name="Frankenpost"))
        prompt = _build_wrap_up_prompt(art, "Body.")
        assert "Frankenpost" in prompt

    def test_mentions_the_tool(self):
        prompt = _build_wrap_up_prompt(make_curated(), "Body.")
        assert "submit_wrap_up" in prompt


# ── _WrapUpResponse ──────────────────────────────────────────────────────────

class TestWrapUpResponse:
    def test_accepts_string(self):
        assert _WrapUpResponse(wrap_up="Some text").wrap_up == "Some text"

    def test_accepts_null(self):
        assert _WrapUpResponse(wrap_up=None).wrap_up is None

    def test_missing_field_raises(self):
        with pytest.raises(Exception):
            _WrapUpResponse()
