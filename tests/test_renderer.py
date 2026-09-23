"""Tests for frankenbote.renderer — HTML output with tmp Jinja2 templates."""

import re
from pathlib import Path

from frankenbote.renderer import (
    DEFAULT_THEME,
    ERROR_PAGES,
    THEME_STORAGE_KEY,
    THEMES,
    RenderConfig,
    _copy_assets,
    _favicon_url,
    _index_entry,
    _list_recent_editions,
    _make_jinja_env,
    _prune_old_html,
    _render_edition,
    _render_index,
    _split_paragraphs,
    render_all,
)
from frankenbote.models import Edition, EditionSection, EditionStats
from tests.conftest import FIXED_NOW, make_curated, make_article

REAL_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


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


# ── Article images in the real edition template ─────────────────────────────

def _edition_with_articles(curated_articles: list) -> Edition:
    section = EditionSection(
        id="politik_verwaltung",
        display_name="Politik & Verwaltung",
        articles=curated_articles,
    )
    return Edition(
        edition_date="2026-05-03",
        window_start=FIXED_NOW,
        window_end=FIXED_NOW,
        sections=[section],
        stats=EditionStats(
            candidates_in=10, curated_kept=5, curated_dropped=5,
            selected=len(curated_articles),
            by_priority={"P1": len(curated_articles)},
            by_section={"politik_verwaltung": len(curated_articles)},
        ),
    )


def _render_real_template(edition: Edition, priority_labels: dict | None = None) -> str:
    env = _make_jinja_env(REAL_TEMPLATES_DIR)
    return _render_edition(env, edition, priority_labels=priority_labels or {})


class TestArticleImages:
    def test_lead_article_with_image_renders_hero(self):
        lead = make_curated(
            is_lead=True,
            article=make_article(image_url="https://img.example.com/full.jpg"),
        )
        html = _render_real_template(_edition_with_articles([lead]))
        assert "article-image--hero" in html
        assert 'src="https://img.example.com/full.jpg"' in html

    def test_regular_article_with_image_renders_thumb(self):
        lead = make_curated(is_lead=True)
        regular = make_curated(
            article=make_article(image_url="https://img.example.com/thumb.jpg"),
        )
        html = _render_real_template(_edition_with_articles([lead, regular]))
        assert "article-image--thumb" in html
        assert "article-image--hero" not in html

    def test_article_without_image_renders_no_article_image(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        assert "article-image" not in html

    def test_image_is_lazy_and_decorative(self):
        lead = make_curated(
            is_lead=True,
            article=make_article(image_url="https://img.example.com/full.jpg"),
        )
        html = _render_real_template(_edition_with_articles([lead]))
        img_start = html.rindex("<img", 0, html.index("full.jpg"))
        img_tag = html[img_start:html.index(">", img_start) + 1]
        assert 'alt=""' in img_tag
        assert 'loading="lazy"' in img_tag

    def test_image_is_not_an_outbound_link(self):
        lead = make_curated(
            is_lead=True,
            article=make_article(image_url="https://img.example.com/full.jpg"),
        )
        html = _render_real_template(_edition_with_articles([lead]))
        img_start = html.rindex("<img", 0, html.index("full.jpg"))
        head_start = html.rindex("<article", 0, img_start)
        # "<a " with the space: "<article" would match a bare "<a".
        assert "<a " not in html[head_start:img_start]

    def test_image_sits_inside_the_disclosure_so_it_expands(self):
        lead = make_curated(
            is_lead=True,
            wrap_up="Der Absatz.",
            article=make_article(image_url="https://img.example.com/full.jpg"),
        )
        html = _render_real_template(_edition_with_articles([lead]))
        summary = _disclosure_summary(html)
        assert "full.jpg" in summary


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
    (templates_dir / "error.html.j2").write_text(
        "<html><body>{{ status_code }}{{ title }}</body></html>", encoding="utf-8"
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
        assert set(stats.keys()) == {
            "editions_rendered",
            "editions_pruned",
            "error_pages_rendered",
            "assets_copied",
        }

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


# ── Headline disclosure and source favicon (ticket #42) ─────────────────────

def _disclosure_summary(html: str) -> str:
    """The inner HTML of the article card's <summary> toggle."""
    start = html.index('<summary class="article__disclosure">')
    return html[start:html.index("</summary>", start)]


class TestFaviconUrl:
    def test_builds_favicon_url_from_host(self):
        assert _favicon_url("https://www.nordbayern.de/artikel/1") == (
            "https://www.nordbayern.de/favicon.ico"
        )

    def test_upgrades_http_source_to_https(self):
        assert _favicon_url("http://example.com/a") == "https://example.com/favicon.ico"

    def test_drops_userinfo_from_host(self):
        assert _favicon_url("https://user:pw@example.com/a") == (
            "https://example.com/favicon.ico"
        )

    def test_keeps_explicit_port(self):
        assert _favicon_url("https://example.com:8443/a") == (
            "https://example.com:8443/favicon.ico"
        )

    def test_lowercases_host(self):
        assert _favicon_url("https://EXAMPLE.com/a") == "https://example.com/favicon.ico"

    def test_non_http_scheme_returns_none(self):
        assert _favicon_url("mailto:redaktion@example.com") is None

    def test_missing_host_returns_none(self):
        assert _favicon_url("/relative/path") is None

    def test_empty_link_returns_none(self):
        assert _favicon_url("") is None

    def test_unparsable_link_returns_none(self):
        # urlparse raises ValueError on a malformed IPv6 host.
        assert _favicon_url("https://[::1/artikel") is None


class TestHeadlineDisclosure:
    def test_headline_is_not_linked_to_the_source(self):
        lead = make_curated(is_lead=True, wrap_up="Der Absatz.")
        html = _render_real_template(_edition_with_articles([lead]))
        title_start = html.index('<h3 class="article__title">')
        assert "<a" not in html[title_start:html.index("</h3>", title_start)]

    def test_headline_is_the_summary_of_the_wrap_up_details(self):
        lead = make_curated(is_lead=True, wrap_up="Erster Absatz.\n\nZweiter Absatz.")
        html = _render_real_template(_edition_with_articles([lead]))
        assert '<details class="article__wrapup-details"' in html
        assert "Test Article" in _disclosure_summary(html)

    def test_wrap_up_paragraphs_follow_the_summary(self):
        lead = make_curated(is_lead=True, wrap_up="Erster Absatz.\n\nZweiter Absatz.")
        html = _render_real_template(_edition_with_articles([lead]))
        body = html[html.index("</summary>"):]
        assert '<p class="article__wrapup">Erster Absatz.</p>' in body
        assert '<p class="article__wrapup">Zweiter Absatz.</p>' in body

    def test_teaser_is_inside_the_summary_so_it_expands(self):
        lead = make_curated(
            is_lead=True, wrap_up="Der Absatz.", ai_summary="Die Kurzfassung."
        )
        html = _render_real_template(_edition_with_articles([lead]))
        assert "Die Kurzfassung." in _disclosure_summary(html)

    def test_expand_cue_is_the_last_element_of_the_summary(self):
        lead = make_curated(
            is_lead=True, wrap_up="Der Absatz.", ai_summary="Die Kurzfassung."
        )
        html = _render_real_template(_edition_with_articles([lead]))
        summary = _disclosure_summary(html)
        assert summary.index("article__wrapup-cue") > summary.index("Die Kurzfassung.")

    def test_article_without_wrap_up_renders_no_disclosure(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        assert "article__wrapup-details" not in html
        assert '<div class="article__head">' in html
        assert "Test Article" in html


def _favicon_tag(html: str) -> str:
    start = html.index('<img class="favicon"')
    return html[start:html.index(">", start) + 1]


def _source_link(html: str) -> str:
    """The full <a class="source-link"> element of the first article card."""
    start = html.index('<a class="source-link"')
    return html[start:html.index("</a>", start)]


class TestSourceFavicon:
    def test_favicon_is_hotlinked_from_the_source_domain(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        assert 'src="https://example.com/favicon.ico"' in _favicon_tag(html)

    def test_favicon_falls_back_to_the_local_frankenrechen(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        assert "this.src='../assets/frankenrechen.svg';" in _favicon_tag(html)

    def test_fallback_clears_its_own_handler_to_avoid_a_loop(self):
        # Without this a failing fallback would re-fire onerror forever.
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        tag = _favicon_tag(html)
        assert tag.index("this.onerror=null") < tag.index("this.src=")

    def test_favicon_is_decorative_and_fixed_size(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        tag = _favicon_tag(html)
        assert 'alt=""' in tag
        assert 'width="16" height="16"' in tag

    def test_source_link_opens_the_article_in_a_safe_new_tab(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        start = html.index('<a class="source-link"')
        link = html[start:html.index(">", start)]
        assert 'href="https://example.com/article/1"' in link
        assert 'target="_blank"' in link
        assert 'rel="noopener noreferrer"' in link

    def test_source_link_is_labelled_for_screen_readers(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        assert 'aria-label="Originalartikel bei Test Source \u00f6ffnen"' in html

    def test_source_link_sits_in_the_meta_line(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        meta_start = html.index('<div class="article__meta">')
        meta = html[meta_start:html.index("</div>", meta_start)]
        assert "Test Source" in meta
        assert 'class="source-link"' in meta

    def test_link_covers_the_publisher_name(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        assert '<span class="article__source">Test Source</span>' in _source_link(html)

    def test_link_covers_the_published_date(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        assert '<span class="article__date">04.05. 10:00</span>' in _source_link(html)

    def test_priority_badge_stays_outside_the_link(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(
            _edition_with_articles([lead]), priority_labels={"P1": "Lokal"}
        )
        assert "Lokal" not in _source_link(html)

    def test_article_without_a_date_still_links(self):
        lead = make_curated(is_lead=True, article=make_article(published=None))
        html = _render_real_template(_edition_with_articles([lead]))
        link = _source_link(html)
        assert "article__date" not in link
        assert "Test Source" in link

    def test_link_without_usable_host_renders_the_fallback_directly(self):
        lead = make_curated(
            is_lead=True, article=make_article(link="mailto:redaktion@example.com")
        )
        html = _render_real_template(_edition_with_articles([lead]))
        tag = _favicon_tag(html)
        assert 'src="../assets/frankenrechen.svg"' in tag
        # No point hotlinking, so no handler either.
        assert "onerror" not in tag


# ── Error pages ──────────────────────────────────────────────────────────────

def _render_error_page(filename: str) -> str:
    """Render one ERROR_PAGES entry with the real template."""
    env = _make_jinja_env(REAL_TEMPLATES_DIR)
    return env.get_template("error.html.j2").render(**ERROR_PAGES[filename])


class TestErrorPages:
    def test_render_all_writes_every_error_page(self, tmp_path, monkeypatch):
        templates_dir = _minimal_templates(tmp_path)
        monkeypatch.setattr("frankenbote.renderer._list_recent_editions", lambda n: [])
        stats = render_all(_render_config(tmp_path, templates_dir))
        assert stats["error_pages_rendered"] == len(ERROR_PAGES)
        for filename in ERROR_PAGES:
            assert (tmp_path / "output" / filename).exists()

    def test_error_pages_are_not_pruned_as_stale_editions(self, tmp_path, monkeypatch):
        templates_dir = _minimal_templates(tmp_path)
        monkeypatch.setattr("frankenbote.renderer._list_recent_editions", lambda n: [])
        stats = render_all(_render_config(tmp_path, templates_dir))
        assert stats["editions_pruned"] == 0
        assert (tmp_path / "output" / "error.html").exists()

    def test_401_page_names_its_status_code(self):
        html = _render_error_page("error-401.html")
        assert '<p class="error__code">401</p>' in html

    def test_generic_page_renders_no_status_code(self):
        html = _render_error_page("error.html")
        assert "error__code" not in html

    def test_every_page_links_to_the_index_to_retrigger_auth(self):
        for filename in ERROR_PAGES:
            html = _render_error_page(filename)
            assert '<a class="error__action" href="/index.html">' in html

    def test_asset_paths_are_absolute(self):
        # An ErrorDocument is served at the URL that failed, which may sit at
        # any depth — relative asset paths would resolve against that depth.
        for filename in ERROR_PAGES:
            html = _render_error_page(filename)
            assert 'href="/assets/style.css"' in html
            assert 'src="/assets/frankenrechen.svg"' in html
            assert "../assets/" not in html

    def test_pages_are_in_english(self):
        for filename in ERROR_PAGES:
            assert '<html lang="en">' in _render_error_page(filename)

    def test_pages_stay_out_of_search_indexes(self):
        for filename in ERROR_PAGES:
            html = _render_error_page(filename)
            assert '<meta name="robots" content="noindex, nofollow">' in html


# ── Themes ───────────────────────────────────────────────────────────────────

REAL_ASSETS_DIR = Path(__file__).parent.parent / "assets"
NON_DEFAULT_THEMES = [t for t in THEMES if t != DEFAULT_THEME]


def _render_real_index() -> str:
    return _render_index(_make_jinja_env(REAL_TEMPLATES_DIR), [])


def _head(html: str) -> str:
    return html[html.index("<head>"):html.index("</head>")]


class TestThemes:
    def test_default_theme_is_classic(self):
        assert DEFAULT_THEME == "classic"
        assert DEFAULT_THEME in THEMES

    def test_every_non_default_theme_ships_a_stylesheet(self):
        for theme in NON_DEFAULT_THEMES:
            assert (REAL_ASSETS_DIR / f"theme-{theme}.css").is_file()

    def test_theme_stylesheets_are_copied_to_output(self, tmp_path):
        dst = tmp_path / "out"
        dst.mkdir()
        _copy_assets(REAL_ASSETS_DIR, dst)
        for theme in NON_DEFAULT_THEMES:
            assert (dst / f"theme-{theme}.css").is_file()

    def test_theme_stylesheets_are_inert_until_selected(self):
        # Every theme stylesheet is linked on every page, so every rule in it
        # must be scoped to its own data-theme — or it leaks into the default.
        for theme in NON_DEFAULT_THEMES:
            css = (REAL_ASSETS_DIR / f"theme-{theme}.css").read_text(encoding="utf-8")
            css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
            selectors = re.findall(r"([^{}]+)\{", css)
            scope = f'[data-theme="{theme}"]'
            for selector in selectors:
                selector = selector.strip()
                if selector.startswith("@"):
                    continue
                for part in selector.split(","):
                    assert part.strip().startswith(scope), part

    def test_pages_render_the_default_theme_attribute(self):
        edition_html = _render_real_template(_edition_with_articles([make_curated(is_lead=True)]))
        for html in (edition_html, _render_real_index()):
            assert f'<html lang="de" data-theme="{DEFAULT_THEME}">' in html

    def test_edition_links_theme_stylesheets_one_level_up(self):
        html = _render_real_template(_edition_with_articles([make_curated(is_lead=True)]))
        for theme in NON_DEFAULT_THEMES:
            assert f'href="../assets/theme-{theme}.css"' in _head(html)

    def test_index_links_theme_stylesheets_at_root_level(self):
        html = _render_real_index()
        for theme in NON_DEFAULT_THEMES:
            assert f'href="assets/theme-{theme}.css"' in _head(html)
            assert "../assets/" not in html

    def test_theme_stylesheets_load_after_the_base_stylesheet(self):
        head = _head(_render_real_index())
        for theme in NON_DEFAULT_THEMES:
            assert head.index("assets/style.css") < head.index(f"assets/theme-{theme}.css")

    def test_stored_theme_is_applied_in_head_before_first_paint(self):
        for html in (
            _render_real_template(_edition_with_articles([make_curated(is_lead=True)])),
            _render_real_index(),
        ):
            head = _head(html)
            assert "<script>" in head
            assert f'localStorage.getItem("{THEME_STORAGE_KEY}")' in head
            assert "document.documentElement.dataset.theme" in head

    def test_switcher_sits_right_below_the_masthead_on_both_pages(self):
        for html in (
            _render_real_template(_edition_with_articles([make_curated(is_lead=True)])),
            _render_real_index(),
        ):
            masthead = html[html.index('<header class="masthead">'):html.index("</header>")]
            assert "theme-switcher" not in masthead
            after_masthead = html[html.index("</header>") + len("</header>"):]
            assert after_masthead.lstrip().startswith('<div class="theme-switcher" hidden>')

    def test_switcher_offers_every_theme_with_its_label(self):
        html = _render_real_index()
        for theme, label in THEMES.items():
            assert f'<option value="{theme}">{label}</option>' in html

    def test_switcher_is_hidden_until_its_script_reveals_it(self):
        # Without JavaScript the dropdown could not do anything, so it ships
        # hidden and the script right after it un-hides it.
        html = _render_real_index()
        start = html.index('<div class="theme-switcher" hidden>')
        switcher_end = html.index("</div>", start)
        assert html[switcher_end + len("</div>"):].lstrip().startswith("<script>")

    def test_switcher_persists_the_choice(self):
        html = _render_real_index()
        assert f'var key = "{THEME_STORAGE_KEY}";' in html
        assert "localStorage.setItem(key, select.value)" in html

    def test_imageless_article_is_marked_for_the_theme_fallback(self):
        lead = make_curated(is_lead=True)
        html = _render_real_template(_edition_with_articles([lead]))
        assert '<article class="article article--noimage">' in html

    def test_article_with_image_is_not_marked_imageless(self):
        lead = make_curated(
            is_lead=True,
            article=make_article(image_url="https://img.example.com/full.jpg"),
        )
        html = _render_real_template(_edition_with_articles([lead]))
        assert '<article class="article">' in html
        assert "article--noimage" not in html

    def test_error_pages_have_no_theme_switcher(self):
        # Error pages must work without CSS and use root-absolute paths; they
        # stay in the default look on purpose.
        for filename in ERROR_PAGES:
            html = _render_error_page(filename)
            assert "theme-switcher" not in html
            assert "theme-" not in _head(html)
