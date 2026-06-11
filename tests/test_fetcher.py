"""Tests for frankenbote.fetcher — _parse() pure tests + async mocked tests."""

from pathlib import Path

import httpx
import pytest
import respx

from frankenbote.fetcher import (
    MAX_RESPONSE_BYTES,
    _download,
    _extract_image_url,
    _first_img_src,
    _is_safe_image_url,
    _parse,
    _strip_img_tags,
    fetch_all,
)
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


# ── Image extraction ─────────────────────────────────────────────────────────

def _feed_with_item(item_inner_xml: str) -> bytes:
    """Build a one-item RSS feed (with the media RSS and metaplus namespaces) for tests."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/"
    xmlns:mp="http://www.tagesschau.de/rss/1.1/modules/metaplus/1.1.1/">
  <channel>
    <title>Test Feed</title>
    <link>https://example.com</link>
    <description>Image extraction tests</description>
    <item>
      <title>Image Article</title>
      <link>https://example.com/article/img</link>
      {item_inner_xml}
    </item>
  </channel>
</rss>""".encode()


class TestImageExtraction:
    def test_media_content_used(self):
        feed = _feed_with_item(
            '<media:content url="https://img.example.com/full.jpg" />'
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.example.com/full.jpg"

    def test_media_content_preferred_over_thumbnail_and_description(self):
        feed = _feed_with_item(
            '<media:content url="https://img.example.com/full.jpg" />'
            '<media:thumbnail url="https://img.example.com/thumb.jpg" />'
            "<description>&lt;img src=\"https://img.example.com/inline.jpg\"&gt; Text.</description>"
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.example.com/full.jpg"

    def test_media_thumbnail_used_when_no_media_content(self):
        feed = _feed_with_item(
            '<media:thumbnail url="https://img.example.com/thumb.jpg" />'
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.example.com/thumb.jpg"

    def test_image_enclosure_used(self):
        # nordbayern.de delivers article images as <enclosure> tags.
        feed = _feed_with_item(
            '<enclosure url="https://images.nordbayern.de/image/contentid/policy:1.123:456/foo.jpg?$p=abc" type="image/jpeg" length="376598"/>'
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://images.nordbayern.de/image/contentid/policy:1.123:456/foo.jpg?$p=abc"

    def test_media_content_preferred_over_enclosure(self):
        feed = _feed_with_item(
            '<media:content url="https://img.example.com/full.jpg" />'
            '<enclosure url="https://img.example.com/enclosure.jpg" type="image/jpeg" length="1"/>'
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.example.com/full.jpg"

    def test_enclosure_preferred_over_description_img(self):
        feed = _feed_with_item(
            '<enclosure url="https://img.example.com/enclosure.jpg" type="image/jpeg" length="1"/>'
            "<description>&lt;img src=\"https://img.example.com/inline.jpg\"&gt; Text.</description>"
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.example.com/enclosure.jpg"

    def test_non_image_enclosure_ignored(self):
        feed = _feed_with_item(
            '<enclosure url="https://audio.example.com/podcast.mp3" type="audio/mpeg" length="1"/>'
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url is None

    def test_mp_image_data_used(self):
        # BR24/tagesschau deliver images via the metaplus <mp:image> schema.
        feed = _feed_with_item(
            "<mp:image>"
            "<mp:width>568</mp:width>"
            "<mp:height>320</mp:height>"
            "<mp:alt>Alt text</mp:alt>"
            "<mp:data> https://img.br.de/pic.jpeg?q=80&amp;w=568&amp;h=320 </mp:data>"
            "</mp:image>"
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.br.de/pic.jpeg?q=80&w=568&h=320"

    def test_mp_image_last_resolution_variant_used(self):
        # metaplus repeats <mp:image> per resolution; feedparser flattens the
        # namespace so only the last variant's mp:data survives.
        feed = _feed_with_item(
            "<mp:image><mp:width>996</mp:width><mp:height>560</mp:height>"
            "<mp:data>https://img.br.de/pic.jpeg?w=996&amp;h=560</mp:data></mp:image>"
            "<mp:image><mp:width>1024</mp:width><mp:height>460</mp:height>"
            "<mp:data>https://img.br.de/pic.jpeg?w=1024&amp;h=460</mp:data></mp:image>"
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.br.de/pic.jpeg?w=1024&h=460"

    def test_media_content_preferred_over_mp_image(self):
        feed = _feed_with_item(
            '<media:content url="https://img.example.com/full.jpg" />'
            "<mp:image><mp:data>https://img.br.de/pic.jpeg</mp:data></mp:image>"
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.example.com/full.jpg"

    def test_mp_image_preferred_over_description_img(self):
        feed = _feed_with_item(
            "<mp:image><mp:data>https://img.br.de/pic.jpeg</mp:data></mp:image>"
            "<description>&lt;img src=\"https://img.example.com/inline.jpg\"&gt; Text.</description>"
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.br.de/pic.jpeg"

    def test_description_img_used_as_last_resort(self):
        feed = _feed_with_item(
            "<description>&lt;img src=\"https://img.example.com/inline.jpg\"&gt; Some text.</description>"
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.example.com/inline.jpg"

    def test_img_tags_stripped_from_summary(self):
        feed = _feed_with_item(
            "<description>Before &lt;img src=\"https://img.example.com/inline.jpg\" alt=\"x\"&gt; after.</description>"
        )
        [article] = _parse(make_source(), feed)
        assert "<img" not in article.summary
        assert "Before" in article.summary
        assert "after." in article.summary

    def test_no_image_anywhere_gives_none(self):
        feed = _feed_with_item("<description>Plain text only.</description>")
        [article] = _parse(make_source(), feed)
        assert article.image_url is None

    def test_non_http_scheme_discarded(self):
        feed = _feed_with_item(
            '<media:content url="javascript:alert(1)" />'
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url is None

    def test_non_http_media_falls_through_to_next_source(self):
        feed = _feed_with_item(
            '<media:content url="data:image/png;base64,AAAA" />'
            '<media:thumbnail url="https://img.example.com/thumb.jpg" />'
        )
        [article] = _parse(make_source(), feed)
        assert article.image_url == "https://img.example.com/thumb.jpg"


class TestImageHelpers:
    def test_is_safe_accepts_http_and_https(self):
        assert _is_safe_image_url("https://example.com/a.jpg") is True
        assert _is_safe_image_url("http://example.com/a.jpg") is True

    def test_is_safe_rejects_other_schemes(self):
        for url in ("javascript:alert(1)", "data:image/png;base64,AAAA",
                    "file:///etc/passwd", "//example.com/a.jpg", "a.jpg"):
            assert _is_safe_image_url(url) is False

    def test_first_img_src_returns_first_of_many(self):
        html = '<p><img src="https://a.jpg"><img src="https://b.jpg"></p>'
        assert _first_img_src(html) == "https://a.jpg"

    def test_first_img_src_none_without_img(self):
        assert _first_img_src("<p>no image</p>") is None

    def test_first_img_src_none_when_src_missing(self):
        assert _first_img_src('<img alt="x">') is None

    def test_strip_img_tags_removes_all_variants(self):
        html = 'a <img src="x.jpg"> b <IMG SRC="y.jpg" /> c </img> d'
        stripped = _strip_img_tags(html)
        assert "img" not in stripped.lower()
        assert " ".join(stripped.split()) == "a b c d"

    def test_extract_prefers_media_content_dict(self):
        entry = {
            "media_content": [{"url": "https://a.jpg"}],
            "media_thumbnail": [{"url": "https://b.jpg"}],
        }
        assert _extract_image_url(entry, "") == "https://a.jpg"

    def test_extract_skips_media_entry_without_url(self):
        entry = {"media_content": [{"type": "video/mp4"}],
                 "media_thumbnail": [{"url": "https://b.jpg"}]}
        assert _extract_image_url(entry, "") == "https://b.jpg"


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
