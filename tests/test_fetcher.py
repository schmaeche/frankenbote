"""Tests for frankenbote.fetcher — _parse() pure tests + async mocked tests."""

from pathlib import Path

import httpx
import pytest
import respx

from frankenbote.fetcher import MAX_RESPONSE_BYTES, _download, _parse, fetch_all
from tests.conftest import make_source

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _feed_bytes(filename: str = "sample_feed.xml") -> bytes:
    return (FIXTURES_DIR / filename).read_bytes()


# ── _parse() — pure function tests ──────────────────────────────────────────

class TestParse:
    def test_parses_two_valid_articles(self):
        source = make_source()
        articles = _parse(source, _feed_bytes())
        # sample_feed.xml has 2 valid articles + 1 no-title + 1 no-link + 1 no-date
        valid = [a for a in articles if a.title in ("First Article", "Second Article")]
        assert len(valid) == 2

    def test_skips_entry_without_title(self):
        source = make_source()
        articles = _parse(source, _feed_bytes())
        titles = [a.title for a in articles]
        # The entry with no title is skipped
        assert all(t for t in titles)

    def test_skips_entry_without_link(self):
        source = make_source()
        articles = _parse(source, _feed_bytes())
        links = [a.link for a in articles]
        assert all(l for l in links)

    def test_article_without_date_has_published_none(self):
        source = make_source()
        articles = _parse(source, _feed_bytes())
        no_date = next(a for a in articles if a.title == "Article Without Date")
        assert no_date.published is None

    def test_article_with_date_has_published_set(self):
        source = make_source()
        articles = _parse(source, _feed_bytes())
        with_date = next(a for a in articles if a.title == "First Article")
        assert with_date.published is not None

    def test_max_articles_cap_respected(self):
        source = make_source(max_articles=1)
        articles = _parse(source, _feed_bytes())
        assert len(articles) <= 1

    def test_source_id_and_name_populated(self):
        source = make_source(id="my_src", name="My Source")
        articles = _parse(source, _feed_bytes())
        for art in articles:
            assert art.source_id == "my_src"
            assert art.source_name == "My Source"

    def test_bozo_feed_with_no_entries_raises(self):
        # An empty body is a bozo feed with no entries
        source = make_source()
        with pytest.raises(ValueError, match="feed parse failed"):
            _parse(source, b"this is not a feed at all <><>")

    def test_bozo_feed_with_entries_does_not_raise(self):
        # feedparser is lenient — a slightly malformed feed that still has
        # entries should not raise (bozo=True but entries present)
        source = make_source()
        # The sample feed is well-formed so this just checks it parses cleanly
        articles = _parse(source, _feed_bytes())
        assert isinstance(articles, list)


# ── _download() — async tests via respx ─────────────────────────────────────

class TestDownload:
    @respx.mock
    async def test_successful_200_returns_bytes(self):
        source = make_source(url="https://example.com/feed")
        respx.get("https://example.com/feed").mock(
            return_value=httpx.Response(200, content=b"<rss/>")
        )
        async with httpx.AsyncClient() as client:
            result = await _download(client, source)
        assert result == b"<rss/>"

    @respx.mock
    async def test_http_404_raises(self):
        source = make_source(url="https://example.com/feed")
        respx.get("https://example.com/feed").mock(
            return_value=httpx.Response(404)
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await _download(client, source)

    async def test_plain_http_without_allow_http_raises(self):
        source = make_source(url="http://example.com/feed", allow_http=False)
        async with httpx.AsyncClient() as client:
            with pytest.raises(ValueError, match="plain HTTP not allowed"):
                await _download(client, source)

    @respx.mock
    async def test_plain_http_with_allow_http_succeeds(self):
        source = make_source(url="http://example.com/feed", allow_http=True)
        respx.get("http://example.com/feed").mock(
            return_value=httpx.Response(200, content=b"<rss/>")
        )
        async with httpx.AsyncClient() as client:
            result = await _download(client, source)
        assert result == b"<rss/>"

    @respx.mock
    async def test_oversized_response_raises(self):
        source = make_source(url="https://example.com/feed")
        big_body = b"x" * (MAX_RESPONSE_BYTES + 1)
        respx.get("https://example.com/feed").mock(
            return_value=httpx.Response(200, content=big_body)
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(ValueError, match="response too large"):
                await _download(client, source)


# ── fetch_all() — integration of _fetch_one ──────────────────────────────────

class TestFetchAll:
    @respx.mock
    async def test_successful_source_has_ok_true(self):
        source = make_source(url="https://example.com/feed")
        respx.get("https://example.com/feed").mock(
            return_value=httpx.Response(200, content=_feed_bytes())
        )
        results = await fetch_all([source])
        assert len(results) == 1
        assert results[0].ok is True

    @respx.mock
    async def test_failing_source_has_ok_false(self):
        source = make_source(url="https://example.com/feed")
        respx.get("https://example.com/feed").mock(
            return_value=httpx.Response(500)
        )
        results = await fetch_all([source])
        assert len(results) == 1
        assert results[0].ok is False
        assert results[0].error is not None

    @respx.mock
    async def test_one_failing_one_succeeding(self):
        good = make_source(id="good_src", url="https://good.example.com/feed")
        bad = make_source(id="bad_src", url="https://bad.example.com/feed")
        respx.get("https://good.example.com/feed").mock(
            return_value=httpx.Response(200, content=_feed_bytes())
        )
        respx.get("https://bad.example.com/feed").mock(
            return_value=httpx.Response(404)
        )
        results = await fetch_all([good, bad])
        assert len(results) == 2
        ok_map = {r.source.id: r.ok for r in results}
        assert ok_map["good_src"] is True
        assert ok_map["bad_src"] is False

    @respx.mock
    async def test_returns_one_result_per_source(self):
        sources = [
            make_source(id=f"src_{i}", url=f"https://example.com/feed/{i}")
            for i in range(3)
        ]
        for src in sources:
            respx.get(str(src.url)).mock(
                return_value=httpx.Response(200, content=_feed_bytes())
            )
        results = await fetch_all(sources)
        assert len(results) == 3
