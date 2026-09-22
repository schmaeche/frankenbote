"""Tests for frankenbote.scraper and the scrape path through fetch_all."""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError

from frankenbote.fetcher import USER_AGENT, fetch_all
from frankenbote.models import ScrapeConfig
from frankenbote.scraper import ScrapeSession, parse, parse_date
from tests.conftest import make_source

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PAGE_URL = "https://www.nuernberg.de/internet/stadtportal/index.html"
ROBOTS_URL = "https://www.nuernberg.de/robots.txt"
NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _scrape_source(**overrides):
    defaults = dict(id="nbg", name="Stadt Nürnberg", url=PAGE_URL, type="scrape")
    defaults.update(overrides)
    return make_source(**defaults)


def _html(*articles: str) -> bytes:
    return f"<html><body>{''.join(articles)}</body></html>".encode()


def _fixture() -> bytes:
    return (FIXTURES_DIR / "scrape_nuernberg.html").read_bytes()


# ── Source config ────────────────────────────────────────────────────────────

class TestSourceConfig:
    def test_rss_is_default_type(self):
        source = make_source()
        assert source.type == "rss"
        assert source.scrape is None

    def test_scrape_type_gets_default_selectors(self):
        assert _scrape_source().scrape == ScrapeConfig()

    def test_scrape_block_on_rss_source_rejected(self):
        with pytest.raises(ValidationError, match="only valid with 'type: scrape'"):
            make_source(scrape={"article_selector": "article"})

    def test_unknown_scrape_key_rejected(self):
        with pytest.raises(ValidationError):
            _scrape_source(scrape={"artcle_selector": "article"})

    def test_link_attr_must_be_attribute_name(self):
        with pytest.raises(ValidationError):
            _scrape_source(scrape={"link_attr": "a[href]"})


# ── parse() against the nuernberg.de snapshot ────────────────────────────────

class TestParseNuernberg:
    def test_default_selector_keeps_undated_navigation_box(self):
        articles = parse(_scrape_source(), _fixture(), now=NOW)
        assert len(articles) == 4
        nav = next(a for a in articles if a.title == "Veranstaltungen")
        assert nav.published is None

    def test_data_publish_selector_keeps_only_news_cards(self):
        source = _scrape_source(scrape={"article_selector": "article[data-publish]"})
        articles = parse(source, _fixture(), now=NOW)
        assert [a.title for a in articles] == [
            "Verkaufsoffener Sonntag",
            "Nürnberg zeigt Interesse an den Finals",
            "Bürgerversammlung im Nürnberger Süden",
        ]

    def test_card_fields(self):
        article = parse(_scrape_source(), _fixture(), now=NOW)[0]
        assert article.source_id == "nbg"
        assert article.source_name == "Stadt Nürnberg"
        assert article.link == "https://www.nuernberg.de/internet/stadtportal/verkaufsoffene_sonntage_nuernberg.html"
        assert article.summary.startswith("Freuen Sie sich auf den 27. September.")
        assert article.published == datetime(2026, 9, 21, 15, 0, tzinfo=UTC)  # data-publish, ms
        assert article.fetched_at == NOW

    def test_relative_link_made_absolute(self):
        articles = parse(_scrape_source(), _fixture(), now=NOW)
        finals = next(a for a in articles if a.title.startswith("Nürnberg zeigt"))
        assert finals.link == "https://www.nuernberg.de/internet/stadtportal/aktuell_100121.html"

    def test_relative_image_made_absolute(self):
        article = parse(_scrape_source(), _fixture(), now=NOW)[0]
        assert article.image_url is not None
        assert article.image_url.startswith("https://www.nuernberg.de/imperia/md/innenstadt/bilder/")

    def test_old_card_keeps_its_date(self):
        articles = parse(_scrape_source(), _fixture(), now=NOW)
        old = next(a for a in articles if a.title.startswith("Bürgerversammlung"))
        assert old.published is not None and old.published.year == 2024


# ── parse() — field extraction and validation ────────────────────────────────

class TestParse:
    def test_minimal_article(self):
        raw = _html('<article><h3>Titel</h3><p>Text.</p><a href="/a.html">mehr</a></article>')
        [article] = parse(_scrape_source(), raw, now=NOW)
        assert article.title == "Titel"
        assert article.summary == "Text."
        assert article.link == "https://www.nuernberg.de/a.html"
        assert article.published is None
        assert article.image_url is None

    @pytest.mark.parametrize(
        ("markup", "missing"),
        [
            ("<article><h2>Titel</h2><p>Text.</p></article>", "url"),
            ('<article><p>Text.</p><a href="/a">x</a></article>', "title"),
            ('<article><h2>Titel</h2><a href="/a">x</a></article>', "summary"),
            ('<article><h2> </h2><p>  </p><a href="/a">x</a></article>', "title, summary"),
        ],
    )
    def test_skips_item_missing_required_field(self, markup, missing, caplog):
        with caplog.at_level(logging.WARNING, logger="frankenbote.scraper"):
            assert parse(_scrape_source(), _html(markup), now=NOW) == []
        assert f"nbg item 1: skipped, missing {missing}" in caplog.text

    def test_bad_item_does_not_stop_the_rest(self):
        raw = _html(
            "<article><h2>Ohne Link</h2><p>Text.</p></article>",
            '<article><h2>Gut</h2><p>Text.</p><a href="/gut">x</a></article>',
        )
        assert [a.title for a in parse(_scrape_source(), raw, now=NOW)] == ["Gut"]

    def test_heading_link_preferred(self):
        raw = _html(
            '<article><a href="/bild"><img src="/i.jpg"></a>'
            '<h2><a href="/artikel">Titel</a></h2><p>Text.</p></article>'
        )
        [article] = parse(_scrape_source(), raw, now=NOW)
        assert article.link == "https://www.nuernberg.de/artikel"

    def test_svg_use_href_is_not_a_link(self):
        raw = _html('<article><h2>Titel</h2><p>Text.</p><svg><use href="/icons.svg#x"></use></svg></article>')
        assert parse(_scrape_source(), raw, now=NOW) == []

    @pytest.mark.parametrize("href", ["#top", "javascript:void(0)", "mailto:a@b.de"])
    def test_non_article_links_ignored(self, href):
        raw = _html(f'<article><h2>Titel</h2><p>Text.</p><a href="{href}">x</a></article>')
        assert parse(_scrape_source(), raw, now=NOW) == []

    def test_custom_link_attr(self):
        source = _scrape_source(scrape={"link_attr": "data-url"})
        raw = _html('<article data-url="/artikel"><h2>Titel</h2><p>Text.</p></article>')
        [article] = parse(source, raw, now=NOW)
        assert article.link == "https://www.nuernberg.de/artikel"

    def test_custom_selectors(self):
        source = _scrape_source(
            scrape={"article_selector": "li", "title_selector": "strong", "summary_selector": "span"}
        )
        raw = _html('<ul><li><strong>Titel</strong><span>Text.</span><a href="/a">x</a></li></ul>')
        [article] = parse(source, raw, now=NOW)
        assert (article.title, article.summary) == ("Titel", "Text.")

    def test_summary_is_not_the_title_again(self):
        source = _scrape_source(scrape={"title_selector": "p", "summary_selector": "p"})
        raw = _html('<article><p>Titel</p><p>Text.</p><a href="/a">x</a></article>')
        [article] = parse(source, raw, now=NOW)
        assert (article.title, article.summary) == ("Titel", "Text.")

    def test_whitespace_collapsed(self):
        raw = _html('<article><h2>  Viel\n   Platz </h2><p>Ein <b>fetter</b>\n Text.</p><a href="/a">x</a></article>')
        [article] = parse(_scrape_source(), raw, now=NOW)
        assert article.title == "Viel Platz"
        assert article.summary == "Ein fetter Text."

    def test_max_articles_cap(self):
        item = '<article><h2>T</h2><p>S</p><a href="/a">x</a></article>'
        assert len(parse(_scrape_source(max_articles=2), _html(item * 5), now=NOW)) == 2

    def test_no_matches_returns_empty_list(self):
        assert parse(_scrape_source(), _html("<div>nichts</div>"), now=NOW) == []

    @pytest.mark.parametrize(
        "raw",
        [
            b"",
            b"<article><h2>Titel<p>Text.<a href='/a'>x",  # nothing closed
            b"<<<>>> </article> <article <h2>",
            b"\xff\xfe\x00garbage\x00",
        ],
    )
    def test_malformed_html_does_not_crash(self, raw):
        assert isinstance(parse(_scrape_source(), raw, now=NOW), list)

    def test_unclosed_tags_still_extracted(self):
        raw = b"<article><h2>Titel</h2><p>Text.<a href='/a'>x</article>"
        [article] = parse(_scrape_source(), raw, now=NOW)
        assert article.title == "Titel"

    def test_html_parser_fallback(self, monkeypatch):
        import frankenbote.scraper as scraper_module
        from bs4 import BeautifulSoup, FeatureNotFound

        def fake_soup(raw, features):
            if features == "lxml":
                raise FeatureNotFound
            return BeautifulSoup(raw, features)

        monkeypatch.setattr(scraper_module, "BeautifulSoup", fake_soup)
        raw = _html('<article><h2>Titel</h2><p>Text.</p><a href="/a">x</a></article>')
        assert len(parse(_scrape_source(), raw, now=NOW)) == 1


# ── Dates ────────────────────────────────────────────────────────────────────

class TestPublishedExtraction:
    def _published(self, article_markup: str):
        [article] = parse(_scrape_source(), _html(article_markup), now=NOW)
        return article.published

    def test_time_datetime_attribute(self):
        markup = '<article><time datetime="2026-09-20T08:30:00+02:00">20.09.</time><h2>T</h2><p>S</p><a href="/a">x</a></article>'
        assert self._published(markup) == datetime(2026, 9, 20, 6, 30, tzinfo=UTC)

    def test_time_text_when_no_attribute(self):
        markup = '<article><h2>T</h2><p>S</p><time>20. September 2026</time><a href="/a">x</a></article>'
        assert self._published(markup) == datetime(2026, 9, 19, 22, 0, tzinfo=UTC)

    def test_itemprop_date_published(self):
        markup = '<article><meta itemprop="datePublished" content="2026-09-20"><h2>T</h2><p>S</p><a href="/a">x</a></article>'
        assert self._published(markup) == datetime(2026, 9, 19, 22, 0, tzinfo=UTC)

    def test_publish_attribute_preferred_over_other_date_attributes(self):
        markup = '<article data-date-modified="1789000000" data-publish="1789711200000"><h2>T</h2><p>S</p><a href="/a">x</a></article>'
        assert self._published(markup) == datetime(2026, 9, 18, 6, 0, tzinfo=UTC)

    def test_unrelated_timestamp_attribute_ignored(self):
        markup = '<article data-sort-timestamp="1789711200000"><h2>T</h2><p>S</p><a href="/a">x</a></article>'
        assert self._published(markup) is None

    def test_unparseable_date_logged_and_none(self, caplog):
        markup = '<article><time>irgendwann</time><h2>T</h2><p>S</p><a href="/a">x</a></article>'
        with caplog.at_level(logging.WARNING, logger="frankenbote.scraper"):
            assert self._published(markup) is None
        assert "unparseable date" in caplog.text
        assert "irgendwann" in caplog.text

    def test_later_candidate_used_when_first_unparseable(self):
        markup = '<article data-publish="1789711200000"><time>bald</time><h2>T</h2><p>S</p><a href="/a">x</a></article>'
        assert self._published(markup) == datetime(2026, 9, 18, 6, 0, tzinfo=UTC)


class TestParseDate:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # ISO 8601
            ("2026-05-07T10:00:00Z", datetime(2026, 5, 7, 10, 0, tzinfo=UTC)),
            ("2026-05-07T10:00:00+02:00", datetime(2026, 5, 7, 8, 0, tzinfo=UTC)),
            ("2026-05-07", datetime(2026, 5, 6, 22, 0, tzinfo=UTC)),  # naive → Berlin
            # Unix timestamps
            ("1789711200000", datetime(2026, 9, 18, 6, 0, tzinfo=UTC)),
            ("1789711200", datetime(2026, 9, 18, 6, 0, tzinfo=UTC)),
            # Common news formats
            ("May 7, 2026", datetime(2026, 5, 6, 22, 0, tzinfo=UTC)),
            ("7. Mai 2026", datetime(2026, 5, 6, 22, 0, tzinfo=UTC)),
            ("Donnerstag, 7. Mai 2026", datetime(2026, 5, 6, 22, 0, tzinfo=UTC)),
            ("07.05.2026", datetime(2026, 5, 6, 22, 0, tzinfo=UTC)),  # day-first
            ("07.05.2026, 14:25 Uhr", datetime(2026, 5, 7, 12, 25, tzinfo=UTC)),
            ("am 3. März 2026 um 9:00 Uhr", datetime(2026, 3, 3, 8, 0, tzinfo=UTC)),  # CET
            ("Do, 1. Okt. 2026", datetime(2026, 9, 30, 22, 0, tzinfo=UTC)),
            ("Stand: 24. Dez 2026", datetime(2026, 12, 23, 23, 0, tzinfo=UTC)),
            # Relative
            ("vor 2 Tagen", datetime(2026, 9, 20, 12, 0, tzinfo=UTC)),
            ("vor einer Stunde", datetime(2026, 9, 22, 11, 0, tzinfo=UTC)),
            ("vor 30 Min.", datetime(2026, 9, 22, 11, 30, tzinfo=UTC)),
            ("5 days ago", datetime(2026, 9, 17, 12, 0, tzinfo=UTC)),
            ("a week ago", datetime(2026, 9, 15, 12, 0, tzinfo=UTC)),
            ("heute", NOW),
            ("Gestern, 14:30 Uhr", datetime(2026, 9, 21, 12, 30, tzinfo=UTC)),
        ],
    )
    def test_supported_formats(self, text, expected):
        assert parse_date(text, NOW) == expected

    @pytest.mark.parametrize("text", ["", "   ", "irgendwann", "12345", "true", "vor langer Zeit"])
    def test_unparseable_returns_none(self, text):
        assert parse_date(text, NOW) is None

    def test_result_is_timezone_aware_utc(self):
        result = parse_date("7. Mai 2026", NOW)
        assert result is not None and result.tzinfo == UTC


# ── ScrapeSession — robots.txt and politeness ────────────────────────────────

ROBOTS_TXT = """User-agent: *
Disallow: /intranet/

User-agent: FrankenboteBot
Disallow: /verboten/
"""


class TestRobots:
    async def _allowed(self, url: str, **robots_response) -> None:
        respx.get(ROBOTS_URL).mock(return_value=httpx.Response(**robots_response))
        async with httpx.AsyncClient() as client:
            await ScrapeSession(client, USER_AGENT, min_interval=0).ensure_allowed(url)

    @respx.mock
    async def test_allowed_path_passes(self):
        await self._allowed(PAGE_URL, status_code=200, text=ROBOTS_TXT)

    @respx.mock
    async def test_disallow_for_all_agents_honored(self):
        robots = "User-agent: *\nDisallow: /intranet/\n"
        with pytest.raises(PermissionError, match="robots.txt disallows"):
            await self._allowed("https://www.nuernberg.de/intranet/x.html", status_code=200, text=robots)

    @respx.mock
    async def test_own_group_replaces_wildcard_group(self):
        # Per RFC 9309 only the most specific matching group applies.
        await self._allowed("https://www.nuernberg.de/intranet/x.html", status_code=200, text=ROBOTS_TXT)

    @respx.mock
    async def test_disallow_for_our_agent_honored(self):
        with pytest.raises(PermissionError):
            await self._allowed("https://www.nuernberg.de/verboten/x.html", status_code=200, text=ROBOTS_TXT)

    @respx.mock
    async def test_missing_robots_txt_allows_everything(self):
        await self._allowed(PAGE_URL, status_code=404)

    @respx.mock
    async def test_forbidden_robots_txt_disallows_everything(self):
        with pytest.raises(PermissionError):
            await self._allowed(PAGE_URL, status_code=403)

    @respx.mock
    async def test_server_error_raises_like_a_failed_download(self):
        with pytest.raises(httpx.HTTPStatusError):
            await self._allowed(PAGE_URL, status_code=503)

    @respx.mock
    async def test_robots_txt_fetched_once_per_origin(self):
        route = respx.get(ROBOTS_URL).mock(return_value=httpx.Response(200, text=ROBOTS_TXT))
        async with httpx.AsyncClient() as client:
            session = ScrapeSession(client, USER_AGENT, min_interval=0)
            await session.ensure_allowed(PAGE_URL)
            await session.ensure_allowed("https://www.nuernberg.de/other.html")
        assert route.call_count == 1


class TestPoliteness:
    async def test_same_host_requests_are_spaced(self):
        async with httpx.AsyncClient() as client:
            session = ScrapeSession(client, USER_AGENT, min_interval=0.1)
            start = time.monotonic()
            await session.wait_turn("https://a.example/1")
            await session.wait_turn("https://a.example/2")
            assert time.monotonic() - start >= 0.09

    async def test_different_hosts_are_not_delayed(self):
        async with httpx.AsyncClient() as client:
            session = ScrapeSession(client, USER_AGENT, min_interval=10)
            start = time.monotonic()
            await session.wait_turn("https://a.example/1")
            await session.wait_turn("https://b.example/1")
            assert time.monotonic() - start < 1


# ── fetch_all() — scrape path ────────────────────────────────────────────────

class TestFetchAllScrape:
    @respx.mock
    async def test_scraped_source_returns_articles(self):
        respx.get(ROBOTS_URL).mock(return_value=httpx.Response(404))
        respx.get(PAGE_URL).mock(return_value=httpx.Response(200, content=_fixture()))
        [result] = await fetch_all([_scrape_source()], min_host_interval=0)
        assert result.ok
        assert len(result.articles) == 4

    @respx.mock
    async def test_robots_disallow_is_an_error_and_page_not_fetched(self):
        respx.get(ROBOTS_URL).mock(return_value=httpx.Response(200, text="User-agent: *\nDisallow: /\n"))
        page = respx.get(PAGE_URL).mock(return_value=httpx.Response(200, content=_fixture()))
        [result] = await fetch_all([_scrape_source()], min_host_interval=0)
        assert not result.ok
        assert result.error is not None and "robots.txt disallows" in result.error
        assert not page.called

    @respx.mock
    async def test_unreachable_robots_is_an_error(self):
        respx.get(ROBOTS_URL).mock(side_effect=httpx.ConnectTimeout("timeout"))
        [result] = await fetch_all([_scrape_source()], min_host_interval=0)
        assert not result.ok
        assert result.error is not None and "ConnectTimeout" in result.error

    @respx.mock
    async def test_zero_articles_is_an_error(self):
        respx.get(ROBOTS_URL).mock(return_value=httpx.Response(404))
        respx.get(PAGE_URL).mock(return_value=httpx.Response(200, content=_html("<div>neu</div>")))
        [result] = await fetch_all([_scrape_source()], min_host_interval=0)
        assert not result.ok
        assert result.error is not None and "no articles found" in result.error

    @respx.mock
    async def test_page_http_error_is_an_error(self):
        respx.get(ROBOTS_URL).mock(return_value=httpx.Response(404))
        respx.get(PAGE_URL).mock(return_value=httpx.Response(500))
        [result] = await fetch_all([_scrape_source()], min_host_interval=0)
        assert not result.ok

    async def test_plain_http_rejected_before_any_request(self):
        # No respx route: any request would raise, so this also proves none is made.
        with respx.mock(assert_all_called=False) as router:
            [result] = await fetch_all([_scrape_source(url="http://www.nuernberg.de/")], min_host_interval=0)
            assert not router.calls
        assert result.error is not None and "plain HTTP not allowed" in result.error

    @respx.mock
    async def test_rss_and_scrape_share_one_candidate_pool(self):
        respx.get(ROBOTS_URL).mock(return_value=httpx.Response(404))
        respx.get(PAGE_URL).mock(return_value=httpx.Response(200, content=_fixture()))
        respx.get("https://example.com/feed").mock(
            return_value=httpx.Response(200, content=(FIXTURES_DIR / "sample_feed.xml").read_bytes())
        )
        rss = make_source(id="feed_src")
        results = await fetch_all([rss, _scrape_source()], min_host_interval=0)
        assert [r.ok for r in results] == [True, True]
        assert {r.source.id for r in results} == {"feed_src", "nbg"}
        # The RSS path is unchanged: no robots.txt lookup for the feed's host.
        assert not any(str(call.request.url) == "https://example.com/robots.txt" for call in respx.calls)
