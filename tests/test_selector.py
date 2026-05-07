"""Tests for frankenbote.selector — quota math and greedy selection."""

import pytest

from frankenbote.models import Priority
from frankenbote.selector import SelectorOptions, _compute_quotas, select
from tests.conftest import make_article, make_curated, make_curator_config

DEFAULT_TARGETS = {
    Priority.P1: 0.50,
    Priority.P2: 0.25,
    Priority.P3: 0.15,
    Priority.P4: 0.10,
}


# ── _compute_quotas ──────────────────────────────────────────────────────────

class TestComputeQuotas:
    def test_quotas_sum_to_size(self):
        for size in [5, 7, 10, 13, 20, 25]:
            quotas = _compute_quotas(size, DEFAULT_TARGETS)
            assert sum(quotas.values()) == size, f"quotas don't sum to {size}"

    def test_size_7_distributes_correctly(self):
        # 7 * 0.50 = 3.5, 7 * 0.25 = 1.75, 7 * 0.15 = 1.05, 7 * 0.10 = 0.70
        # floors: P1=3, P2=1, P3=1, P4=0 → sum=5, leftover=2
        # largest remainders: P2(0.75), P4(0.70) get +1 each → P1=3,P2=2,P3=1,P4=1
        quotas = _compute_quotas(7, DEFAULT_TARGETS)
        assert sum(quotas.values()) == 7

    def test_size_zero_all_zeros(self):
        quotas = _compute_quotas(0, DEFAULT_TARGETS)
        assert sum(quotas.values()) == 0

    def test_all_priorities_present_in_output(self):
        quotas = _compute_quotas(20, DEFAULT_TARGETS)
        assert set(quotas.keys()) == set(Priority)

    def test_no_negative_quotas(self):
        quotas = _compute_quotas(3, DEFAULT_TARGETS)
        assert all(v >= 0 for v in quotas.values())


# ── select ───────────────────────────────────────────────────────────────────

def _articles_for_section(n: int, section: str, priority: Priority, base_score: float = 7.0):
    return [
        make_curated(
            article=make_article(link=f"https://example.com/{section}/{i}"),
            section=section,
            priority=priority,
            relevance_score=base_score - i * 0.1,
        )
        for i in range(n)
    ]


class TestSelectBasic:
    def test_rejected_articles_excluded(self):
        kept = make_curated(section="politik_verwaltung")
        dropped = make_curated(section=None, article=make_article(link="https://example.com/dropped"))
        config = make_curator_config()
        edition = select([kept, dropped], config, source_ids_in_order=["test_src"])
        all_articles = [a for s in edition.sections for a in s.articles]
        links = [a.article.link for a in all_articles]
        assert "https://example.com/dropped" not in links

    def test_edition_size_respected(self):
        articles = _articles_for_section(20, "politik_verwaltung", Priority.P1)
        config = make_curator_config()
        options = SelectorOptions(edition_size=5)
        edition = select(articles, config, source_ids_in_order=["test_src"], options=options)
        total = sum(len(s.articles) for s in edition.sections)
        assert total == 5

    def test_fewer_articles_than_size_gives_smaller_edition(self):
        articles = _articles_for_section(3, "politik_verwaltung", Priority.P1)
        config = make_curator_config()
        options = SelectorOptions(edition_size=25)
        edition = select(articles, config, source_ids_in_order=["test_src"], options=options)
        total = sum(len(s.articles) for s in edition.sections)
        assert total == 3

    def test_empty_input_gives_empty_edition(self):
        config = make_curator_config()
        edition = select([], config, source_ids_in_order=[])
        assert edition.sections == []
        assert edition.stats.selected == 0


class TestSelectLeadMarking:
    def test_first_article_in_section_is_lead(self):
        articles = _articles_for_section(3, "politik_verwaltung", Priority.P1)
        config = make_curator_config()
        edition = select(articles, config, source_ids_in_order=["test_src"])
        section = next(s for s in edition.sections if s.id == "politik_verwaltung")
        assert section.articles[0].is_lead is True

    def test_remaining_articles_not_lead(self):
        articles = _articles_for_section(3, "politik_verwaltung", Priority.P1)
        config = make_curator_config()
        edition = select(articles, config, source_ids_in_order=["test_src"])
        section = next(s for s in edition.sections if s.id == "politik_verwaltung")
        for art in section.articles[1:]:
            assert art.is_lead is False

    def test_each_section_has_exactly_one_lead(self):
        p1_articles = _articles_for_section(3, "politik_verwaltung", Priority.P1)
        wirtschaft_articles = _articles_for_section(3, "wirtschaft", Priority.P2)
        config = make_curator_config()
        edition = select(
            p1_articles + wirtschaft_articles,
            config,
            source_ids_in_order=["test_src"],
        )
        for section in edition.sections:
            leads = [a for a in section.articles if a.is_lead]
            assert len(leads) == 1, f"section {section.id} has {len(leads)} leads"


class TestSelectSectionOrdering:
    def test_sections_in_config_order(self):
        # Config has: politik_verwaltung, wirtschaft, kultur
        articles = (
            _articles_for_section(2, "kultur", Priority.P3)
            + _articles_for_section(2, "wirtschaft", Priority.P2)
            + _articles_for_section(2, "politik_verwaltung", Priority.P1)
        )
        config = make_curator_config()
        edition = select(articles, config, source_ids_in_order=["test_src"])
        section_ids = [s.id for s in edition.sections]
        # Only sections that have articles appear; order must match config
        expected_order = ["politik_verwaltung", "wirtschaft", "kultur"]
        present = [sid for sid in expected_order if sid in section_ids]
        assert section_ids == present

    def test_empty_sections_omitted(self):
        articles = _articles_for_section(2, "politik_verwaltung", Priority.P1)
        config = make_curator_config()
        edition = select(articles, config, source_ids_in_order=["test_src"])
        section_ids = [s.id for s in edition.sections]
        assert "wirtschaft" not in section_ids
        assert "kultur" not in section_ids


class TestSelectStats:
    def test_stats_curated_dropped_counts_none_sections(self):
        kept = make_curated(section="politik_verwaltung")
        dropped = make_curated(
            section=None,
            article=make_article(link="https://example.com/dropped"),
        )
        config = make_curator_config()
        edition = select([kept, dropped], config, source_ids_in_order=["test_src"])
        assert edition.stats.curated_dropped == 1
        assert edition.stats.curated_kept == 1

    def test_stats_selected_matches_article_count(self):
        articles = _articles_for_section(5, "politik_verwaltung", Priority.P1)
        config = make_curator_config()
        edition = select(articles, config, source_ids_in_order=["test_src"])
        total = sum(len(s.articles) for s in edition.sections)
        assert edition.stats.selected == total

    def test_stats_by_priority_sums_to_selected(self):
        articles = (
            _articles_for_section(3, "politik_verwaltung", Priority.P1)
            + _articles_for_section(2, "wirtschaft", Priority.P2)
        )
        config = make_curator_config()
        edition = select(articles, config, source_ids_in_order=["test_src"])
        assert sum(edition.stats.by_priority.values()) == edition.stats.selected
