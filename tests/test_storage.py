"""Tests for frankenbote.storage — round-trip JSON with tmp_path."""

from datetime import datetime, timezone

import pytest

import frankenbote.storage as storage_module
from frankenbote.models import Edition, EditionSection, EditionStats
from tests.conftest import FIXED_NOW, make_article, make_curated


EDITION_DATE = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def _make_edition() -> Edition:
    curated = make_curated()
    curated = curated.model_copy(update={"is_lead": True})
    section = EditionSection(
        id="politik_verwaltung",
        display_name="Politik & Verwaltung",
        articles=[curated],
    )
    return Edition(
        edition_date=EDITION_DATE.date().isoformat(),
        window_start=FIXED_NOW,
        window_end=FIXED_NOW,
        sections=[section],
        stats=EditionStats(
            candidates_in=10,
            curated_kept=5,
            curated_dropped=5,
            selected=1,
            by_priority={"P1": 1},
            by_section={"politik_verwaltung": 1},
        ),
    )


@pytest.fixture(autouse=True)
def redirect_editions_dir(tmp_path, monkeypatch):
    """Redirect all storage operations to a temporary directory."""
    monkeypatch.setattr(storage_module, "EDITIONS_DIR", tmp_path)


# ── candidates ───────────────────────────────────────────────────────────────

class TestCandidatesRoundTrip:
    def test_save_and_load_returns_same_articles(self):
        articles = [
            make_article(link="https://example.com/1"),
            make_article(link="https://example.com/2"),
        ]
        storage_module.save_candidates(articles, EDITION_DATE, FIXED_NOW, FIXED_NOW)
        loaded = storage_module.load_candidates(EDITION_DATE)
        assert len(loaded) == 2
        assert loaded[0].link == articles[0].link
        assert loaded[1].link == articles[1].link

    def test_save_returns_path_that_exists(self, tmp_path):
        articles = [make_article()]
        path = storage_module.save_candidates(articles, EDITION_DATE, FIXED_NOW, FIXED_NOW)
        assert path.exists()

    def test_candidates_path_naming(self, tmp_path):
        articles = [make_article()]
        path = storage_module.save_candidates(articles, EDITION_DATE, FIXED_NOW, FIXED_NOW)
        assert path.name == f"{EDITION_DATE.date().isoformat()}-candidates.json"

    def test_load_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            storage_module.load_candidates(EDITION_DATE)


# ── curated raw ──────────────────────────────────────────────────────────────

class TestCuratedRawRoundTrip:
    def test_save_and_load_returns_same_items(self):
        curated = [
            make_curated(article=make_article(link="https://example.com/1")),
            make_curated(
                article=make_article(link="https://example.com/2"),
                section=None,
            ),
        ]
        storage_module.save_curated_raw(curated, EDITION_DATE)
        loaded = storage_module.load_curated_raw(EDITION_DATE)
        assert len(loaded) == 2
        assert loaded[0].article.link == "https://example.com/1"
        assert loaded[1].section is None

    def test_load_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            storage_module.load_curated_raw(EDITION_DATE)


# ── edition (final) ──────────────────────────────────────────────────────────

class TestEditionRoundTrip:
    def test_save_and_load_returns_equal_edition(self):
        edition = _make_edition()
        storage_module.save_edition(edition, EDITION_DATE)
        loaded = storage_module.load_edition(EDITION_DATE)
        assert loaded.edition_date == edition.edition_date
        assert loaded.stats.selected == edition.stats.selected
        assert len(loaded.sections) == 1
        assert loaded.sections[0].articles[0].is_lead is True

    def test_save_returns_path_that_exists(self):
        edition = _make_edition()
        path = storage_module.save_edition(edition, EDITION_DATE)
        assert path.exists()

    def test_edition_path_naming(self):
        edition = _make_edition()
        path = storage_module.save_edition(edition, EDITION_DATE)
        assert path.name == f"{EDITION_DATE.date().isoformat()}.json"

    def test_load_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            storage_module.load_edition(EDITION_DATE)
