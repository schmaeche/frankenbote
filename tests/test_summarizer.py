"""Tests for frankenbote.summarizer — body selection and the orchestration of
summarize_edition() against a scripted LLM client (no API calls).

Prompt rendering and the reading of the model's answers belong to the
summarizer and wrap-up tasks (test_task_summarize.py, test_task_wrap_up.py).
"""

from datetime import datetime, timezone

import pytest

from frankenbote.llm import LLMTransientError
from frankenbote.models import Edition, EditionSection, EditionStats
from frankenbote.summarizer import (
    _select_body,
    _wrap_up_inputs,
    summarize_edition,
)
from tests.conftest import ScriptedLLMClient, make_article, make_curated, tool_result


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


# ── _wrap_up_inputs ──────────────────────────────────────────────────────────

class TestWrapUpInputs:
    def _make_selected(self, n: int):
        return [
            (s, a, make_curated(article=make_article(
                link=f"https://example.com/{s}-{a}",
                title=f"Article {s}-{a}",
                summary="A feed snippet.",
            )))
            for s, a in [(0, 0), (0, 1), (1, 0)][:n]
        ]

    def test_positions_and_items_stay_parallel(self):
        selected = self._make_selected(3)
        bodies = {item.article.link: "Body." for _, _, item in selected}
        positions, items = _wrap_up_inputs(selected, bodies)
        assert positions == [(0, 0), (0, 1), (1, 0)]
        assert [article.article.title for article, _ in items] == [
            "Article 0-0", "Article 0-1", "Article 1-0"
        ]

    def test_prefers_the_fetched_body(self):
        selected = self._make_selected(1)
        bodies = {selected[0][2].article.link: "Distinctive fetched body."}
        _, items = _wrap_up_inputs(selected, bodies)
        assert items[0][1] == "Distinctive fetched body."

    def test_falls_back_to_the_feed_snippet(self):
        _, items = _wrap_up_inputs(self._make_selected(1), {})
        assert items[0][1] == "A feed snippet."

    def test_articles_without_usable_text_are_dropped(self):
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
        positions, items = _wrap_up_inputs([first, second, third], bodies)
        assert positions == [(0, 0)]
        assert len(items) == 1

    def test_all_without_text_returns_nothing(self):
        selected = [
            (0, 0, make_curated(article=make_article(link="https://example.com/0", summary=""))),
            (0, 1, make_curated(article=make_article(link="https://example.com/1", summary=""))),
        ]
        assert _wrap_up_inputs(selected, {}) == ([], [])


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
