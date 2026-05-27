"""Tests for frankenbote.summarizer — pure helpers only (no API calls)."""

import json
from types import SimpleNamespace

import pytest

from frankenbote.summarizer import (
    SummarizerConfig,
    _WrapUpResponse,
    _build_user_prompt,
    _build_wrap_up_batch_requests,
    _build_wrap_up_prompt,
    _extract_summarizer_result,
    _extract_wrap_up_results,
    _normalize_tool_input,
    _select_body,
    load_summarizer_config,
)
from tests.conftest import make_article, make_curated


# ── helpers for building mock batch results ──────────────────────────────────

def _make_succeeded_result(custom_id: str, tool_name: str, tool_input: dict) -> SimpleNamespace:
    tool_block = SimpleNamespace(type="tool_use", name=tool_name, input=tool_input)
    message = SimpleNamespace(content=[tool_block])
    result = SimpleNamespace(type="succeeded", message=message)
    return SimpleNamespace(custom_id=custom_id, result=result)


def _make_failed_result(custom_id: str, result_type: str) -> SimpleNamespace:
    result = SimpleNamespace(type=result_type)
    return SimpleNamespace(custom_id=custom_id, result=result)


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


# ── _extract_summarizer_result ───────────────────────────────────────────────

class TestExtractSummarizerResult:
    _TOOL_INPUT = {"summaries": [{"article_index": 0, "summary": "Ein kurzer Text."}]}

    def test_succeeded_returns_tool_input_and_tool_use(self):
        results = [_make_succeeded_result("summarizer", "submit_summaries", self._TOOL_INPUT)]
        tool_input, stop_reason = _extract_summarizer_result(results)
        assert stop_reason == "tool_use"
        assert tool_input == self._TOOL_INPUT

    def test_errored_returns_none_and_errored(self):
        results = [_make_failed_result("summarizer", "errored")]
        tool_input, stop_reason = _extract_summarizer_result(results)
        assert tool_input is None
        assert stop_reason == "errored"

    def test_expired_returns_none_and_expired(self):
        results = [_make_failed_result("summarizer", "expired")]
        tool_input, stop_reason = _extract_summarizer_result(results)
        assert tool_input is None
        assert stop_reason == "expired"

    def test_missing_custom_id_returns_no_result(self):
        results = [_make_succeeded_result("something-else", "submit_summaries", self._TOOL_INPUT)]
        tool_input, stop_reason = _extract_summarizer_result(results)
        assert tool_input is None
        assert stop_reason == "no_result"

    def test_empty_iterator_returns_no_result(self):
        tool_input, stop_reason = _extract_summarizer_result([])
        assert tool_input is None
        assert stop_reason == "no_result"

    def test_succeeded_but_wrong_tool_block(self):
        results = [_make_succeeded_result("summarizer", "other_tool", self._TOOL_INPUT)]
        tool_input, stop_reason = _extract_summarizer_result(results)
        assert tool_input is None
        assert stop_reason == "no_tool_use_block"


# ── _extract_wrap_up_results ─────────────────────────────────────────────────

class TestExtractWrapUpResults:
    def test_succeeded_items_mapped(self):
        results = [
            _make_succeeded_result("wrapup-0-0", "submit_wrap_up", {"wrap_up": "Text A."}),
            _make_succeeded_result("wrapup-1-2", "submit_wrap_up", {"wrap_up": "Text B."}),
        ]
        mapping = _extract_wrap_up_results(results)
        assert mapping[(0, 0)] == "Text A."
        assert mapping[(1, 2)] == "Text B."

    def test_llm_null_wrap_up_stored_as_none(self):
        results = [_make_succeeded_result("wrapup-0-0", "submit_wrap_up", {"wrap_up": None})]
        mapping = _extract_wrap_up_results(results)
        assert mapping[(0, 0)] is None

    def test_errored_item_mapped_to_none(self):
        results = [_make_failed_result("wrapup-0-1", "errored")]
        mapping = _extract_wrap_up_results(results)
        assert mapping[(0, 1)] is None

    def test_expired_item_mapped_to_none(self):
        results = [_make_failed_result("wrapup-2-0", "expired")]
        mapping = _extract_wrap_up_results(results)
        assert mapping[(2, 0)] is None

    def test_malformed_custom_id_skipped(self):
        results = [_make_succeeded_result("wrapup-notanint-x", "submit_wrap_up", {"wrap_up": "x"})]
        mapping = _extract_wrap_up_results(results)
        assert mapping == {}

    def test_non_wrapup_custom_id_ignored(self):
        results = [_make_succeeded_result("summarizer", "submit_summaries", {})]
        mapping = _extract_wrap_up_results(results)
        assert mapping == {}

    def test_validation_error_stored_as_none(self):
        bad_block = SimpleNamespace(type="tool_use", name="submit_wrap_up", input={"bad_field": "x"})
        message = SimpleNamespace(content=[bad_block])
        result_obj = SimpleNamespace(type="succeeded", message=message)
        item = SimpleNamespace(custom_id="wrapup-0-0", result=result_obj)
        mapping = _extract_wrap_up_results([item])
        assert mapping[(0, 0)] is None

    def test_no_tool_use_block_in_succeeded_result(self):
        text_block = SimpleNamespace(type="text", text="some text")
        message = SimpleNamespace(content=[text_block])
        result_obj = SimpleNamespace(type="succeeded", message=message)
        item = SimpleNamespace(custom_id="wrapup-0-0", result=result_obj)
        mapping = _extract_wrap_up_results([item])
        assert mapping[(0, 0)] is None


# ── _build_wrap_up_batch_requests ────────────────────────────────────────────

class TestBuildWrapUpBatchRequests:
    def _make_selected(self, n: int):
        return [
            (s, a, make_curated(article=make_article(
                link=f"https://example.com/{s}-{a}",
                title=f"Article {s}-{a}",
                summary="A feed snippet.",
            )))
            for s, a in [(0, 0), (0, 1), (1, 0)][:n]
        ]

    def test_custom_id_format(self):
        selected = self._make_selected(2)
        bodies = {item.article.link: "Body text." for _, _, item in selected}
        requests = _build_wrap_up_batch_requests(selected, bodies, "claude-haiku-4-5", 1200)
        custom_ids = [r["custom_id"] for r in requests]
        assert custom_ids == ["wrapup-0-0", "wrapup-0-1"]

    def test_request_count_matches_articles_with_bodies(self):
        selected = self._make_selected(3)
        bodies = {item.article.link: "Body." for _, _, item in selected}
        requests = _build_wrap_up_batch_requests(selected, bodies, "claude-haiku-4-5", 1200)
        assert len(requests) == 3

    def test_articles_without_body_excluded(self):
        # _select_body returns None only when both fetched body AND feed summary are absent.
        # Give the second and third articles empty summaries so they are excluded.
        first = (0, 0, make_curated(article=make_article(
            link="https://example.com/0-0", summary="Feed snippet."
        )))
        second = (0, 1, make_curated(article=make_article(
            link="https://example.com/0-1", summary=""
        )))
        third = (1, 0, make_curated(article=make_article(
            link="https://example.com/1-0", summary=""
        )))
        bodies = {first[2].article.link: "Body text."}
        requests = _build_wrap_up_batch_requests([first, second, third], bodies, "claude-haiku-4-5", 1200)
        assert len(requests) == 1
        assert requests[0]["custom_id"] == "wrapup-0-0"

    def test_all_no_body_returns_empty(self):
        # Articles with empty summaries and no fetched bodies → _select_body returns None.
        selected = [
            (0, 0, make_curated(article=make_article(link="https://example.com/0", summary=""))),
            (0, 1, make_curated(article=make_article(link="https://example.com/1", summary=""))),
        ]
        requests = _build_wrap_up_batch_requests(selected, {}, "claude-haiku-4-5", 1200)
        assert requests == []

    def test_params_contain_model(self):
        selected = self._make_selected(1)
        bodies = {selected[0][2].article.link: "Body text."}
        requests = _build_wrap_up_batch_requests(selected, bodies, "claude-test-model", 1200)
        assert requests[0]["params"]["model"] == "claude-test-model"
