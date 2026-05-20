"""Tests for frankenbote.renderer — HTML output with tmp Jinja2 templates."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from frankenbote.renderer import (
    RenderConfig,
    _copy_assets,
    _index_entry,
    _list_recent_editions,
    _prune_old_html,
    _split_paragraphs,
    render_all,
)
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


# ── _split_paragraphs ────────────────────────────────────────────────────────

class TestSplitParagraphs:
    def test_splits_on_blank_line(self):
        assert _split_paragraphs("Para one.\n\nPara two.") == ["Para one.", "Para two."]

    def test_single_paragraph(self):
        assert _split_paragraphs("Just one paragraph.") == ["Just one paragraph."]

    def test_collapses_multiple_blank_lines(self):
        assert _split_paragraphs("A.\n\n\n\nB.") == ["A.", "B."]

    def test_handles_windows_newlines(self):
        assert _split_paragraphs("A.\r\n\r\nB.") == ["A.", "B."]

    def test_strips_surrounding_whitespace(self):
        assert _split_paragraphs("  \n\nMiddle.\n\n  ") == ["Middle."]

    def test_empty_string_returns_empty_list(self):
        assert _split_paragraphs("") == []


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


# ── _list_recent_editions ─────────────────────────────────────────────────────

def _write_edition_file(editions_dir: Path, date_str: str) -> None:
    edition = _make_edition(edition_date=date_str)
    (editions_dir / f"{date_str}.json").write_text(
        edition.model_dump_json(), encoding="utf-8"
    )


class TestListRecentEditions:
    def test_returns_empty_when_dir_missing(self, monkeypatch):
        monkeypatch.setattr("frankenbote.renderer.EDITIONS_DIR", Path("/nonexistent/__test__"))
        assert _list_recent_editions(5) == []

    def test_returns_empty_for_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("frankenbote.renderer.EDITIONS_DIR", tmp_path)
        assert _list_recent_editions(5) == []

    def test_returns_editions_newest_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr("frankenbote.renderer.EDITIONS_DIR", tmp_path)
        monkeypatch.setattr("frankenbote.storage.EDITIONS_DIR", tmp_path)
        for d in ["2026-04-01", "2026-05-01", "2026-03-01"]:
            _write_edition_file(tmp_path, d)
        result = _list_recent_editions(5)
        assert [e.edition_date for e in result] == ["2026-05-01", "2026-04-01", "2026-03-01"]

    def test_retention_limits_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr("frankenbote.renderer.EDITIONS_DIR", tmp_path)
        monkeypatch.setattr("frankenbote.storage.EDITIONS_DIR", tmp_path)
        for d in ["2026-03-01", "2026-04-01", "2026-05-01"]:
            _write_edition_file(tmp_path, d)
        result = _list_recent_editions(2)
        assert len(result) == 2
        assert result[0].edition_date == "2026-05-01"
        assert result[1].edition_date == "2026-04-01"

    def test_intermediate_files_excluded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("frankenbote.renderer.EDITIONS_DIR", tmp_path)
        monkeypatch.setattr("frankenbote.storage.EDITIONS_DIR", tmp_path)
        _write_edition_file(tmp_path, "2026-05-01")
        (tmp_path / "2026-05-01-candidates.json").write_text("{}", encoding="utf-8")
        (tmp_path / "2026-05-01-curated-raw.json").write_text("{}", encoding="utf-8")
        result = _list_recent_editions(5)
        assert len(result) == 1
        assert result[0].edition_date == "2026-05-01"

    def test_invalid_filename_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("frankenbote.renderer.EDITIONS_DIR", tmp_path)
        monkeypatch.setattr("frankenbote.storage.EDITIONS_DIR", tmp_path)
        (tmp_path / "not-a-date.json").write_text("{}", encoding="utf-8")
        _write_edition_file(tmp_path, "2026-05-01")
        result = _list_recent_editions(5)
        assert len(result) == 1
        assert result[0].edition_date == "2026-05-01"


# ── render_all ────────────────────────────────────────────────────────────────

def _minimal_templates(parent: Path) -> Path:
    templates_dir = parent / "templates"
    templates_dir.mkdir()
    (templates_dir / "edition.html.j2").write_text(
        "<html><body>{{ edition.edition_date }}</body></html>", encoding="utf-8"
    )
    (templates_dir / "index.html.j2").write_text(
        "<html><body>index</body></html>", encoding="utf-8"
    )
    return templates_dir


def _render_config(tmp_path: Path, templates_dir: Path, **overrides) -> RenderConfig:
    defaults = dict(
        templates_dir=templates_dir,
        assets_dir=tmp_path / "assets",
        output_dir=tmp_path / "output",
        sections_config=Path("/nonexistent/sections.yaml"),
    )
    defaults.update(overrides)
    return RenderConfig(**defaults)


class TestRenderAll:
    def test_returns_expected_stat_keys(self, tmp_path, monkeypatch):
        templates_dir = _minimal_templates(tmp_path)
        monkeypatch.setattr("frankenbote.renderer._list_recent_editions", lambda n: [])
        stats = render_all(_render_config(tmp_path, templates_dir))
        assert set(stats.keys()) == {"editions_rendered", "editions_pruned", "assets_copied"}

    def test_creates_index_html(self, tmp_path, monkeypatch):
        templates_dir = _minimal_templates(tmp_path)
        monkeypatch.setattr("frankenbote.renderer._list_recent_editions", lambda n: [])
        render_all(_render_config(tmp_path, templates_dir))
        assert (tmp_path / "output" / "index.html").exists()

    def test_renders_one_html_per_edition(self, tmp_path, monkeypatch):
        templates_dir = _minimal_templates(tmp_path)
        editions = [_make_edition("2026-05-01"), _make_edition("2026-04-01")]
        monkeypatch.setattr("frankenbote.renderer._list_recent_editions", lambda n: editions)
        render_all(_render_config(tmp_path, templates_dir))
        out = tmp_path / "output" / "editions"
        assert (out / "2026-05-01.html").exists()
        assert (out / "2026-04-01.html").exists()

    def test_editions_rendered_count_in_stats(self, tmp_path, monkeypatch):
        templates_dir = _minimal_templates(tmp_path)
        editions = [_make_edition("2026-05-01"), _make_edition("2026-04-01")]
        monkeypatch.setattr("frankenbote.renderer._list_recent_editions", lambda n: editions)
        stats = render_all(_render_config(tmp_path, templates_dir))
        assert stats["editions_rendered"] == 2

    def test_prunes_stale_html(self, tmp_path, monkeypatch):
        templates_dir = _minimal_templates(tmp_path)
        editions = [_make_edition("2026-05-01")]
        monkeypatch.setattr("frankenbote.renderer._list_recent_editions", lambda n: editions)
        stale = tmp_path / "output" / "editions"
        stale.mkdir(parents=True)
        (stale / "2026-01-01.html").write_text("<html/>", encoding="utf-8")
        stats = render_all(_render_config(tmp_path, templates_dir))
        assert stats["editions_pruned"] == 1
        assert not (stale / "2026-01-01.html").exists()

    def test_copies_assets(self, tmp_path, monkeypatch):
        templates_dir = _minimal_templates(tmp_path)
        monkeypatch.setattr("frankenbote.renderer._list_recent_editions", lambda n: [])
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (assets_dir / "style.css").write_text("body {}", encoding="utf-8")
        stats = render_all(_render_config(tmp_path, templates_dir, assets_dir=assets_dir))
        assert stats["assets_copied"] == 1
        assert (tmp_path / "output" / "assets" / "style.css").exists()

    def test_zero_stats_when_no_editions_and_no_assets(self, tmp_path, monkeypatch):
        templates_dir = _minimal_templates(tmp_path)
        monkeypatch.setattr("frankenbote.renderer._list_recent_editions", lambda n: [])
        stats = render_all(_render_config(tmp_path, templates_dir))
        assert stats["editions_rendered"] == 0
        assert stats["editions_pruned"] == 0
        assert stats["assets_copied"] == 0
