"""Tests for frankenbote.body_fetcher — article body fetching via respx."""

import httpx
import respx

from frankenbote.body_fetcher import MIN_BODY_CHARS, fetch_bodies, fetch_body

# A realistic news-article page. The body comfortably exceeds
# MIN_BODY_CHARS so trafilatura returns a real extraction.
_ARTICLE_BODY = (
    "Der Nürnberger Christkindlesmarkt zählt zu den bekanntesten "
    "Weihnachtsmärkten Deutschlands und zieht jedes Jahr Millionen "
    "Besucher in die Innenstadt. In diesem Jahr beginnt der Markt "
    "Ende November und dauert bis zum Heiligabend.\n\n"
    "Die Stadtverwaltung rechnet erneut mit großem Andrang und hat das "
    "Sicherheitskonzept gemeinsam mit der Polizei überarbeitet. "
    "Zusätzliche Einsatzkräfte sollen den Bereich rund um den "
    "Hauptmarkt absichern, ohne die festliche Stimmung zu stören.\n\n"
    "Auch die Händler bereiten sich vor: Viele Stände bieten regionale "
    "Spezialitäten an, von Lebkuchen bis zu fränkischen Bratwürsten. "
    "Die Eröffnung wird traditionell vom Christkind vorgenommen."
)

_PAYWALL_BODY = "Dieser Artikel ist nur für Abonnenten verfügbar."

URL = "https://news.example.com/artikel/1"


def _html_page(body: str) -> str:
    paragraphs = "".join(f"<p>{p}</p>" for p in body.split("\n\n"))
    return (
        '<!DOCTYPE html><html lang="de"><head><title>Test</title></head>'
        "<body><nav>Menü</nav>"
        f"<article><h1>Eine Schlagzeile aus Franken</h1>{paragraphs}</article>"
        "<footer>Impressum</footer></body></html>"
    )


# ── fetch_body ───────────────────────────────────────────────────────────────

class TestFetchBody:
    @respx.mock
    async def test_extracts_article_body(self):
        respx.get(URL).mock(
            return_value=httpx.Response(200, html=_html_page(_ARTICLE_BODY))
        )
        async with httpx.AsyncClient() as client:
            body = await fetch_body(client, URL)
        assert body is not None
        assert "Christkindlesmarkt" in body
        assert len(body) >= MIN_BODY_CHARS

    @respx.mock
    async def test_short_body_treated_as_failure(self):
        respx.get(URL).mock(
            return_value=httpx.Response(200, html=_html_page(_PAYWALL_BODY))
        )
        async with httpx.AsyncClient() as client:
            body = await fetch_body(client, URL)
        assert body is None

    @respx.mock
    async def test_http_404_returns_none(self):
        respx.get(URL).mock(return_value=httpx.Response(404))
        async with httpx.AsyncClient() as client:
            body = await fetch_body(client, URL)
        assert body is None

    @respx.mock
    async def test_timeout_returns_none(self):
        respx.get(URL).mock(side_effect=httpx.TimeoutException("slow"))
        async with httpx.AsyncClient() as client:
            body = await fetch_body(client, URL)
        assert body is None

    @respx.mock
    async def test_garbage_html_returns_none(self):
        respx.get(URL).mock(
            return_value=httpx.Response(200, html="<x><x><x")
        )
        async with httpx.AsyncClient() as client:
            body = await fetch_body(client, URL)
        assert body is None


# ── fetch_bodies ─────────────────────────────────────────────────────────────

class TestFetchBodies:
    async def test_empty_list_returns_empty_dict(self):
        assert await fetch_bodies([]) == {}

    @respx.mock
    async def test_maps_each_url_to_result(self):
        good = "https://news.example.com/gut"
        bad = "https://news.example.com/schlecht"
        respx.get(good).mock(
            return_value=httpx.Response(200, html=_html_page(_ARTICLE_BODY))
        )
        respx.get(bad).mock(return_value=httpx.Response(404))
        result = await fetch_bodies([good, bad])
        assert set(result.keys()) == {good, bad}
        assert result[good] is not None
        assert result[bad] is None

    @respx.mock
    async def test_duplicate_urls_fetched_once(self):
        route = respx.get(URL).mock(
            return_value=httpx.Response(200, html=_html_page(_ARTICLE_BODY))
        )
        result = await fetch_bodies([URL, URL])
        assert route.call_count == 1
        assert set(result.keys()) == {URL}
