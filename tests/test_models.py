"""Tests for frankenbote.models — Pydantic validation edge cases."""

import pytest
from pydantic import ValidationError

from frankenbote.models import (
    Article,
    Category,
    CuratedArticle,
    CuratorDecision,
    CuratorResponse,
    Priority,
    Source,
)
from tests.conftest import IN_WINDOW_DATE, FIXED_NOW


# ── Source validation ────────────────────────────────────────────────────────

class TestSourceValidation:
    def _valid_kwargs(self, **overrides):
        base = dict(
            id="valid_id",
            name="Test Source",
            url="https://example.com/feed",
            category=Category.LOCAL,
        )
        base.update(overrides)
        return base

    def test_valid_source_accepted(self):
        s = Source(**self._valid_kwargs())
        assert s.id == "valid_id"

    def test_id_with_spaces_rejected(self):
        with pytest.raises(ValidationError):
            Source(**self._valid_kwargs(id="has space"))

    def test_id_with_uppercase_rejected(self):
        with pytest.raises(ValidationError):
            Source(**self._valid_kwargs(id="HasUpper"))

    def test_id_with_hyphen_rejected(self):
        with pytest.raises(ValidationError):
            Source(**self._valid_kwargs(id="has-hyphen"))

    def test_id_with_underscore_and_numbers_accepted(self):
        s = Source(**self._valid_kwargs(id="nue_nn_01"))
        assert s.id == "nue_nn_01"

    def test_max_articles_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            Source(**self._valid_kwargs(max_articles=0))

    def test_max_articles_above_maximum_rejected(self):
        with pytest.raises(ValidationError):
            Source(**self._valid_kwargs(max_articles=501))

    def test_max_articles_at_bounds_accepted(self):
        Source(**self._valid_kwargs(max_articles=1))
        Source(**self._valid_kwargs(max_articles=500))

    def test_non_http_url_rejected(self):
        with pytest.raises(ValidationError):
            Source(**self._valid_kwargs(url="ftp://example.com/feed"))

    def test_http_url_accepted(self):
        s = Source(**self._valid_kwargs(url="http://example.com/feed"))
        assert str(s.url).startswith("http://")

    def test_enabled_defaults_to_true(self):
        s = Source(**self._valid_kwargs())
        assert s.enabled is True

    def test_allow_http_defaults_to_false(self):
        s = Source(**self._valid_kwargs())
        assert s.allow_http is False


# ── CuratedArticle validation ─────────────────────────────────────────────────

class TestCuratedArticleValidation:
    def _valid_article(self):
        return Article(
            source_id="test_src",
            source_name="Test",
            title="Title",
            link="https://example.com/1",
            published=IN_WINDOW_DATE,
            fetched_at=FIXED_NOW,
        )

    def _valid_kwargs(self, **overrides):
        base = dict(
            article=self._valid_article(),
            section="politik",
            priority=Priority.P1,
            relevance_score=7.0,
            rationale="A valid rationale.",
        )
        base.update(overrides)
        return base

    def test_valid_curated_article_accepted(self):
        ca = CuratedArticle(**self._valid_kwargs())
        assert ca.relevance_score == 7.0

    def test_relevance_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            CuratedArticle(**self._valid_kwargs(relevance_score=-0.1))

    def test_relevance_score_above_ten_rejected(self):
        with pytest.raises(ValidationError):
            CuratedArticle(**self._valid_kwargs(relevance_score=10.1))

    def test_relevance_score_at_bounds_accepted(self):
        CuratedArticle(**self._valid_kwargs(relevance_score=0.0))
        CuratedArticle(**self._valid_kwargs(relevance_score=10.0))

    def test_section_none_accepted(self):
        ca = CuratedArticle(**self._valid_kwargs(section=None))
        assert ca.section is None

    def test_is_lead_defaults_to_false(self):
        ca = CuratedArticle(**self._valid_kwargs())
        assert ca.is_lead is False


# ── CuratorDecision validation ───────────────────────────────────────────────

class TestCuratorDecisionValidation:
    def _valid_kwargs(self, **overrides):
        base = dict(
            article_index=0,
            section="politik",
            priority=Priority.P2,
            relevance_score=5.0,
            rationale="Short rationale.",
        )
        base.update(overrides)
        return base

    def test_rationale_at_max_length_accepted(self):
        long_rationale = "x" * 300
        d = CuratorDecision(**self._valid_kwargs(rationale=long_rationale))
        assert len(d.rationale) == 300

    def test_rationale_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            CuratorDecision(**self._valid_kwargs(rationale="x" * 301))

    def test_negative_article_index_rejected(self):
        with pytest.raises(ValidationError):
            CuratorDecision(**self._valid_kwargs(article_index=-1))


# ── CuratorResponse round-trip ───────────────────────────────────────────────

class TestCuratorResponseRoundTrip:
    def test_model_dump_and_reconstruct(self):
        decision = CuratorDecision(
            article_index=0,
            section="wirtschaft",
            priority=Priority.P1,
            relevance_score=8.5,
            rationale="Economy story.",
        )
        response = CuratorResponse(decisions=[decision])
        dumped = response.model_dump()
        reconstructed = CuratorResponse(**dumped)
        assert reconstructed.decisions[0].article_index == 0
        assert reconstructed.decisions[0].relevance_score == 8.5
