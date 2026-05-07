"""Tests for frankenbote.renderer — HTML output with tmp Jinja2 templates."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from frankenbote.renderer import _copy_assets, _index_entry, _prune_old_html
from frankenbote.models import Edition, EditionSection, EditionStats
from tests.conftest import FIXED_NOW, make_curated, make_article


def _make_edition(edition_date: str = "2026-05-03", selected: int = 5) -> Edition:
    curated = make_curated()
    curated = curated.model_copy(update={"is_lead": True})
    section = EditionSection(
        id="politik_verwaltung",
        display_name="Politik & Verwaltung",
        articles=[curated],
    )
    return Edition(
        edition_date=edition_date,
        window_start=FIXED_NOW,
        window_end=FIXED_NOW,
        sections=[section],
        stats=EditionStats(
            candidates_in=10,
            curated_kept=5,
            curated_dropped=5,
            selected=selected,
            by_priority={"P1": selected},
            by_section={"politik_verwaltung": selected},
        ),
    )


# ── _index_entry ─────────────────────────────────────────────────────────────

class TestIndexEntry:
    def test_date_label_format(self):
        edition = _make_edition(edition_date="2026-05-03")
        entry = _index_entry(edition)
        assert entry["date_label"] == "03.05.2026"

    def test_filename_contains_iso_date(self):
        edition = _make_edition(edition_date="2026-05-03")
        entry = _index_entry(edition)
        assert "2026-05-03" in entry["filename"]

    def test_article_count_matches_stats_selected(self):
        edition = _make_edition(selected=12)
        entry = _index_entry(edition)
        assert entry["article_count"] == 12


# ── _prune_old_html ───────────────────────────────────────────────────────────

class TestPruneOldHtml:
    def test_removes_html_not_in_kept_dates(self, tmp_path):
        old_file = tmp_path / "2026-04-01.html"
        old_file.write_text("<html/>", encoding="utf-8")
        count = _prune_old_html(tmp_path, kept_dates={"2026-05-03"})
        assert count == 1
        assert not old_file.exists()

    def test_keeps_html_in_kept_dates(self, tmp_path):
        keep_file = tmp_path / "2026-05-03.html"
        keep_file.write_text("<html/>", encoding="utf-8")
        count = _prune_old_html(tmp_path, kept_dates={"2026-05-03"})
        assert count == 0
        assert keep_file.exists()

    def test_mixed_keep_and_prune(self, tmp_path):
        (tmp_path / "2026-05-03.html").write_text("<html/>", encoding="utf-8")
        (tmp_path / "2026-04-01.html").write_text("<html/>", encoding="utf-8")
        (tmp_path / "2026-03-15.html").write_text("<html/>", encoding="utf-8")
        count = _prune_old_html(tmp_path, kept_dates={"2026-05-03"})
        assert count == 2
        assert (tmp_path / "2026-05-03.html").exists()

    def test_empty_dir_returns_zero(self, tmp_path):
        count = _prune_old_html(tmp_path, kept_dates={"2026-05-03"})
        assert count == 0

    def test_non_html_files_not_touched(self, tmp_path):
        css_file = tmp_path / "style.css"
        css_file.write_text("body {}", encoding="utf-8")
        count = _prune_old_html(tmp_path, kept_dates=set())
        assert count == 0
        assert css_file.exists()


# ── _copy_assets ─────────────────────────────────────────────────────────────

class TestCopyAssets:
    def test_copies_files_and_returns_count(self, tmp_path):
        src = tmp_path / "assets"
        dst = tmp_path / "output" / "assets"
        src.mkdir()
        dst.mkdir(parents=True)
        (src / "style.css").write_text("body {}", encoding="utf-8")
        (src / "icon.svg").write_text("<svg/>", encoding="utf-8")
        count = _copy_assets(src, dst)
        assert count == 2
        assert (dst / "style.css").exists()
        assert (dst / "icon.svg").exists()

    def test_missing_src_returns_zero(self, tmp_path):
        dst = tmp_path / "output" / "assets"
        dst.mkdir(parents=True)
        count = _copy_assets(tmp_path / "nonexistent", dst)
        assert count == 0

    def test_file_content_is_preserved(self, tmp_path):
        src = tmp_path / "assets"
        dst = tmp_path / "output" / "assets"
        src.mkdir()
        dst.mkdir(parents=True)
        (src / "style.css").write_text("body { color: red; }", encoding="utf-8")
        _copy_assets(src, dst)
        assert (dst / "style.css").read_text() == "body { color: red; }"
