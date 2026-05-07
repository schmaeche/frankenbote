"""Tests for frankenbote.filter — the pure filtering pipeline."""

from datetime import datetime, timedelta, timezone

import pytest

from frankenbote.filter import FilterConfig, compute_window, filter_articles
from tests.conftest import FIXED_NOW, IN_WINDOW_DATE, make_article


# ── compute_window ──────────────────────────────────────────────────────────

class TestComputeWindowRolling7d:
    def test_start_is_exactly_7_days_before_now(self):
        config = FilterConfig(window={"anchor": "rolling_7d", "timezone": "UTC"})
        start, end = compute_window(config, now=FIXED_NOW)
        assert start == FIXED_NOW - timedelta(days=7)
        assert end == FIXED_NOW

    def test_naive_now_gets_timezone_attached(self):
        config = FilterConfig(window={"anchor": "rolling_7d", "timezone": "UTC"})
        naive_now = datetime(2026, 5, 6, 12, 0, 0)  # no tzinfo
        start, end = compute_window(config, now=naive_now)
        assert end.tzinfo is not None
        assert start == end - timedelta(days=7)


class TestComputeWindowPreviousSaturday:
    """Parametrize over all weekdays to lock in the Saturday-edge-case logic."""

    @pytest.mark.parametrize("weekday_offset, expected_days_back", [
        # offset from the reference Wednesday (weekday=2)
        (0, 4),   # Wednesday → back 4 days to Saturday
        (1, 5),   # Thursday  → back 5 days to Saturday
        (2, 6),   # Friday    → back 6 days to Saturday
        (3, 7),   # Saturday  → back 7 days (NOT 0!) to previous Saturday
        (4, 1),   # Sunday    → back 1 day to Saturday
        (5, 2),   # Monday    → back 2 days to Saturday
        (6, 3),   # Tuesday   → back 3 days to Saturday
    ])
    def test_previous_saturday(self, weekday_offset, expected_days_back):
        # FIXED_NOW is Wednesday 2026-05-06
        now = FIXED_NOW + timedelta(days=weekday_offset)
        config = FilterConfig(window={"anchor": "previous_saturday", "timezone": "UTC"})
        start, end = compute_window(config, now=now)
        expected_saturday = (now - timedelta(days=expected_days_back)).date()
        assert start.date() == expected_saturday
        assert start.hour == 0 and start.minute == 0 and start.second == 0

    def test_start_is_always_midnight(self):
        config = FilterConfig(window={"anchor": "previous_saturday", "timezone": "UTC"})
        start, _ = compute_window(config, now=FIXED_NOW)
        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0

    def test_end_is_now(self):
        config = FilterConfig(window={"anchor": "previous_saturday", "timezone": "UTC"})
        _, end = compute_window(config, now=FIXED_NOW)
        assert end == FIXED_NOW


# ── filter_articles ──────────────────────────────────────────────────────────

def _default_config() -> FilterConfig:
    return FilterConfig(window={"anchor": "rolling_7d", "timezone": "UTC"})


class TestFilterArticlesDateWindow:
    def test_article_inside_window_is_kept(self):
        article = make_article(published=IN_WINDOW_DATE)
        result = filter_articles([article], _default_config(), now=FIXED_NOW)
        assert len(result.articles) == 1
        assert result.stats.output_count == 1
        assert result.stats.dropped_outside_window == 0

    def test_article_outside_window_is_dropped(self):
        old_date = FIXED_NOW - timedelta(days=30)
        article = make_article(published=old_date)
        result = filter_articles([article], _default_config(), now=FIXED_NOW)
        assert len(result.articles) == 0
        assert result.stats.dropped_outside_window == 1

    def test_article_at_window_boundary_is_kept(self):
        # Exactly at the start boundary (rolling_7d: now - 7 days)
        boundary = FIXED_NOW - timedelta(days=7)
        article = make_article(published=boundary)
        result = filter_articles([article], _default_config(), now=FIXED_NOW)
        assert len(result.articles) == 1

    def test_mixed_window_articles(self):
        inside = make_article(link="https://example.com/1", published=IN_WINDOW_DATE)
        outside = make_article(link="https://example.com/2", published=FIXED_NOW - timedelta(days=30))
        result = filter_articles([inside, outside], _default_config(), now=FIXED_NOW)
        assert len(result.articles) == 1
        assert result.stats.dropped_outside_window == 1
        assert result.stats.output_count == 1


class TestFilterArticlesFallbackDate:
    def test_no_published_date_uses_fetched_at(self):
        # fetched_at is FIXED_NOW which is inside the 7d window
        article = make_article(published=None, fetched_at=IN_WINDOW_DATE)
        result = filter_articles([article], _default_config(), now=FIXED_NOW)
        assert len(result.articles) == 1

    def test_no_published_date_increments_counter(self):
        article = make_article(published=None, fetched_at=IN_WINDOW_DATE)
        result = filter_articles([article], _default_config(), now=FIXED_NOW)
        assert result.stats.dropped_no_date_kept == 1

    def test_article_with_published_does_not_increment_counter(self):
        article = make_article(published=IN_WINDOW_DATE)
        result = filter_articles([article], _default_config(), now=FIXED_NOW)
        assert result.stats.dropped_no_date_kept == 0


class TestFilterArticlesBlocklist:
    def _config_with_blocklist(self, terms: list[str]) -> FilterConfig:
        return FilterConfig(
            window={"anchor": "rolling_7d", "timezone": "UTC"},
            drop_if_title_contains=terms,
        )

    def test_exact_match_drops_article(self):
        article = make_article(title="Breaking: Disaster strikes")
        config = self._config_with_blocklist(["Disaster"])
        result = filter_articles([article], config, now=FIXED_NOW)
        assert len(result.articles) == 0
        assert result.stats.dropped_blocked_title == 1

    def test_case_insensitive_match(self):
        article = make_article(title="DISASTER in the city")
        config = self._config_with_blocklist(["disaster"])
        result = filter_articles([article], config, now=FIXED_NOW)
        assert len(result.articles) == 0

    def test_substring_match(self):
        article = make_article(title="Anzeige: Special offer")
        config = self._config_with_blocklist(["Anzeige"])
        result = filter_articles([article], config, now=FIXED_NOW)
        assert len(result.articles) == 0

    def test_no_match_keeps_article(self):
        article = make_article(title="City council meets today")
        config = self._config_with_blocklist(["Anzeige", "Gewinnspiel"])
        result = filter_articles([article], config, now=FIXED_NOW)
        assert len(result.articles) == 1
        assert result.stats.dropped_blocked_title == 0

    def test_empty_blocklist_keeps_all(self):
        articles = [
            make_article(link="https://example.com/1", title="Article one"),
            make_article(link="https://example.com/2", title="Article two"),
        ]
        config = self._config_with_blocklist([])
        result = filter_articles(articles, config, now=FIXED_NOW)
        assert len(result.articles) == 2


class TestFilterArticlesDeduplication:
    def test_duplicate_link_first_wins(self):
        a1 = make_article(link="https://example.com/same", title="First version")
        a2 = make_article(link="https://example.com/same", title="Second version")
        result = filter_articles([a1, a2], _default_config(), now=FIXED_NOW)
        assert len(result.articles) == 1
        assert result.articles[0].title == "First version"
        assert result.stats.dropped_duplicates == 1

    def test_unique_links_all_kept(self):
        articles = [
            make_article(link="https://example.com/1"),
            make_article(link="https://example.com/2"),
            make_article(link="https://example.com/3"),
        ]
        result = filter_articles(articles, _default_config(), now=FIXED_NOW)
        assert len(result.articles) == 3
        assert result.stats.dropped_duplicates == 0


class TestFilterArticlesSorting:
    def test_output_is_newest_first(self):
        older = make_article(link="https://example.com/1", published=IN_WINDOW_DATE)
        newer = make_article(
            link="https://example.com/2",
            published=IN_WINDOW_DATE + timedelta(hours=3),
        )
        result = filter_articles([older, newer], _default_config(), now=FIXED_NOW)
        assert result.articles[0].link == newer.link
        assert result.articles[1].link == older.link


class TestFilterStats:
    def test_input_count_reflects_all_articles(self):
        articles = [make_article(link=f"https://example.com/{i}") for i in range(5)]
        result = filter_articles(articles, _default_config(), now=FIXED_NOW)
        assert result.stats.input_count == 5

    def test_window_returned_in_result(self):
        result = filter_articles([], _default_config(), now=FIXED_NOW)
        assert result.window_start < result.window_end
        assert result.window_end == FIXED_NOW
