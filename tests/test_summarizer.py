"""Tests for frankenbote.summarizer — pure helpers plus the public entry points
against a scripted LLM client (no API calls)."""

from datetime import datetime, timezone

import pytest

from frankenbote.llm import LLMError, LLMTransientError
from frankenbote.models import Edition, EditionSection, EditionStats
from frankenbote.summarizer import (
    _build_user_prompt,
    _build_wrap_up_batch_items,
    _build_wrap_up_prompt,
    _generate_one_wrap_up,
    _map_wrap_up_results,
    _select_body,
    summarize_edition,
)
from tests.conftest import ScriptedLLMClient, make_article, make_curated, tool_result


# ── _build_user_prompt ───────────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_prompt_contains_article_count(self):
        articles = [
            make_curated(article=make_article(link=f"https://example.com/{i}"))
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
        article = make_curated(article=make_article(title="Unique Headline ABC"))
        prompt = _build_user_prompt([article])
        assert "Unique Headline ABC" in prompt

    def test_count_in_closing_line_matches_input(self):
        articles = [
            make_curated(article=make_article(link=f"https://example.com/{i}"))
            for i in range(7)
        ]
        prompt = _build_user_prompt(articles)
        assert "7 Einträge erwartet" in prompt


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


# ── _map_wrap_up_results ─────────────────────────────────────────────────────

class TestMapWrapUpResults:
    def test_succeeded_items_mapped(self):
        results = [
            tool_result("wrapup-0-0", {"wrap_up": "Text A."}),
            tool_result("wrapup-1-2", {"wrap_up": "Text B."}),
        ]
        mapping = _map_wrap_up_results(results)
        assert mapping[(0, 0)] == "Text A."
        assert mapping[(1, 2)] == "Text B."

    def test_llm_null_wrap_up_stored_as_none(self):
        mapping = _map_wrap_up_results([tool_result("wrapup-0-0", {"wrap_up": None})])
        assert mapping[(0, 0)] is None

    def test_errored_item_mapped_to_none(self):
        mapping = _map_wrap_up_results([tool_result("wrapup-0-1", None, "errored")])
        assert mapping[(0, 1)] is None

    def test_expired_item_mapped_to_none(self):
        mapping = _map_wrap_up_results([tool_result("wrapup-2-0", None, "expired")])
        assert mapping[(2, 0)] is None

    def test_malformed_custom_id_skipped(self):
        mapping = _map_wrap_up_results([tool_result("wrapup-notanint-x", {"wrap_up": "x"})])
        assert mapping == {}

    def test_non_wrapup_custom_id_ignored(self):
        mapping = _map_wrap_up_results([tool_result("summarizer", {})])
        assert mapping == {}

    def test_validation_error_stored_as_none(self):
        mapping = _map_wrap_up_results([tool_result("wrapup-0-0", {"bad_field": "x"})])
        assert mapping[(0, 0)] is None

    def test_no_tool_use_block_in_succeeded_result(self):
        mapping = _map_wrap_up_results([tool_result("wrapup-0-0", None, "no_tool_use_block")])
        assert mapping[(0, 0)] is None


# ── _build_wrap_up_batch_items ───────────────────────────────────────────────

class TestBuildWrapUpBatchItems:
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
        items = _build_wrap_up_batch_items(selected, bodies)
        assert [custom_id for custom_id, _ in items] == ["wrapup-0-0", "wrapup-0-1"]

    def test_item_count_matches_articles_with_bodies(self):
        selected = self._make_selected(3)
        bodies = {item.article.link: "Body." for _, _, item in selected}
        assert len(_build_wrap_up_batch_items(selected, bodies)) == 3

    def test_prompt_uses_fetched_body(self):
        selected = self._make_selected(1)
        bodies = {selected[0][2].article.link: "Distinctive fetched body."}
        [(_, prompt)] = _build_wrap_up_batch_items(selected, bodies)
        assert "Distinctive fetched body." in prompt
        assert "Article 0-0" in prompt

    def test_prompt_falls_back_to_feed_snippet(self):
        selected = self._make_selected(1)
        [(_, prompt)] = _build_wrap_up_batch_items(selected, {})
        assert "A feed snippet." in prompt

    def test_articles_without_body_excluded(self):
        # _select_body returns None only when both fetched body AND feed summary are absent.
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
        items = _build_wrap_up_batch_items([first, second, third], bodies)
        assert len(items) == 1
        assert items[0][0] == "wrapup-0-0"

    def test_all_no_body_returns_empty(self):
        selected = [
            (0, 0, make_curated(article=make_article(link="https://example.com/0", summary=""))),
            (0, 1, make_curated(article=make_article(link="https://example.com/1", summary=""))),
        ]
        assert _build_wrap_up_batch_items(selected, {}) == []


# ── summarize_edition() ──────────────────────────────────────────────────────

def _make_edition(*section_articles: list) -> Edition:
    now = datetime(2026, 5, 6, tzinfo=timezone.utc)
    sections = [
        EditionSection(id=f"sec{i}", display_name=f"Section {i}", articles=arts)
        for i, arts in enumerate(section_articles)
    ]
    n = sum(len(a) for a in section_articles)
    return Edition(
        edition_date="2026-05-09", window_start=now, window_end=now, sections=sections,
        stats=EditionStats(
            candidates_in=n, curated_kept=n, curated_dropped=0, selected=n,
            by_priority={}, by_section={},
        ),
    )


class TestSummarizeEdition:
    def test_empty_edition_skips_the_client(self):
        edition = _make_edition([])
        client = ScriptedLLMClient()
        assert summarize_edition(edition, client) is edition
        assert client.calls == []

    def test_populates_ai_summary_by_flat_index(self):
        edition = _make_edition(
            [make_curated(article=make_article(link="https://e.com/1"))],
            [make_curated(article=make_article(link="https://e.com/2")),
             make_curated(article=make_article(link="https://e.com/3"))],
        )
        client = ScriptedLLMClient([[tool_result("summarizer", {"summaries": [
            {"article_index": 0, "summary": "Eins."},
            {"article_index": 1, "summary": None},
            {"article_index": 2, "summary": "Drei."},
        ]})]])

        out = summarize_edition(edition, client)

        assert out.sections[0].articles[0].ai_summary == "Eins."
        assert out.sections[1].articles[0].ai_summary is None
        assert out.sections[1].articles[1].ai_summary == "Drei."
        # Input edition untouched.
        assert edition.sections[0].articles[0].ai_summary is None
        [request] = client.calls[0][1]
        assert request["task"] == "summarizer"
        assert request["custom_id"] == "summarizer"
        assert "model" not in request["params"]
        assert request["params"]["tool"]["name"] == "submit_summaries"
        assert request["params"]["max_tokens"] == 200 + 120 * 3

    def test_client_batch_off_uses_sync_call(self):
        edition = _make_edition([make_curated()])
        client = ScriptedLLMClient([tool_result("summarizer", {"summaries": [
            {"article_index": 0, "summary": "S."}]})], use_batch=False)
        summarize_edition(edition, client)
        assert [n for n, _ in client.calls] == ["call_tool"]

    def test_network_error_is_retried(self):
        edition = _make_edition([make_curated()])
        client = ScriptedLLMClient([
            LLMTransientError("net"),
            tool_result("summarizer", {"summaries": [{"article_index": 0, "summary": "S."}]}),
        ], use_batch=False)
        out = summarize_edition(edition, client)
        assert out.sections[0].articles[0].ai_summary == "S."

    def test_summaries_as_json_string_are_normalised(self):
        edition = _make_edition([make_curated()])
        client = ScriptedLLMClient([tool_result(
            "summarizer", {"summaries": '[{"article_index": 0, "summary": "S."}]'}
        )], use_batch=False)
        assert summarize_edition(edition, client).sections[0].articles[0].ai_summary == "S."

    def test_persistent_validation_failure_raises(self, monkeypatch):
        import frankenbote.llm.base as base_module
        monkeypatch.setattr(base_module, "save_failure", lambda *a: "debug.txt")
        bad = tool_result("summarizer", {"summaries": [{"article_index": "x", "summary": 1}]})
        client = ScriptedLLMClient([bad, bad], use_batch=False)
        with pytest.raises(RuntimeError, match="Summarizer"):
            summarize_edition(_make_edition([make_curated()]), client)


# ── _generate_one_wrap_up() ──────────────────────────────────────────────────

class TestGenerateOneWrapUp:
    def test_returns_wrap_up_text(self):
        client = ScriptedLLMClient([tool_result("wrap_up", {"wrap_up": "Langer Text."})])
        out = _generate_one_wrap_up(client, make_curated(), "Body.")
        assert out == "Langer Text."
        [(name, request)] = client.calls
        assert name == "call_tool"  # always synchronous, even with batch on
        assert request["task"] == "wrap_up"
        assert request["params"]["tool"]["name"] == "submit_wrap_up"
        assert request["params"]["max_tokens"] == 1200
        assert "Body." in request["params"]["user_prompt"]

    def test_null_wrap_up_returns_none(self):
        client = ScriptedLLMClient([tool_result("wrap_up", {"wrap_up": None})])
        assert _generate_one_wrap_up(client, make_curated(), "Body.") is None

    def test_network_error_retried_then_ok(self):
        client = ScriptedLLMClient([LLMTransientError("net"), tool_result("wrap_up", {"wrap_up": "T."})])
        assert _generate_one_wrap_up(client, make_curated(), "Body.") == "T."

    def test_persistent_failure_returns_none_without_raising(self, monkeypatch):
        import frankenbote.llm.base as base_module
        dumps = []
        monkeypatch.setattr(base_module, "save_failure", lambda *a: dumps.append(a) or "x")
        bad = tool_result("wrap_up", None, "max_tokens")
        client = ScriptedLLMClient([bad, bad])
        assert _generate_one_wrap_up(client, make_curated(), "Body.") is None
        assert dumps == []  # per-article wrap-ups never write debug dumps

    def test_non_transient_api_error_returns_none(self):
        client = ScriptedLLMClient([LLMError("400 bad request")])
        assert _generate_one_wrap_up(client, make_curated(), "Body.") is None
        assert len(client.calls) == 1
