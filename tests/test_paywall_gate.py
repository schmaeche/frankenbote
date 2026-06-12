"""Tests for frankenbote.paywall_gate — paywall-aware edition selection."""

import json

import httpx
import respx

from frankenbote.paywall_gate import select_edition
from frankenbote.selector import SelectorOptions
from tests.conftest import make_article, make_curated, make_curator_config

CONFIG = make_curator_config()
SOURCE_IDS = ["test_src"]
# Size 2 with three candidates: a dropped article frees a slot for the third.
OPTIONS = SelectorOptions(edition_size=2)

LINKS = [f"https://news.example.com/artikel/{i}" for i in range(3)]


# Long enough that the content-length strategy stays silent — these pages
# are decided (or not) by their metadata alone.
_LONG_BODY = "<p>" + (
    "Die Lage bleibt nach Einschätzung von Beobachtern angespannt, "
    "auch wenn sich einzelne Indikatoren zuletzt leicht verbessert haben. "
) * 30 + "</p>"


def _page(ld_json: str | None) -> str:
    script = (
        f'<script type="application/ld+json">{ld_json}</script>'
        if ld_json is not None
        else ""
    )
    return (
        '<!DOCTYPE html><html lang="de"><head><title>Test</title>'
        f"{script}</head>"
        f"<body><article><h1>Eine Schlagzeile</h1>{_LONG_BODY}</article></body></html>"
    )


def _ld(accessible: bool) -> str:
    return json.dumps({"@type": "NewsArticle", "isAccessibleForFree": accessible})


_FREE_PAGE = _page(_ld(True))
_PAYWALLED_PAGE = _page(_ld(False))
_NO_SIGNAL_PAGE = _page(None)

# FAZ-style: no usable metadata, only a teaser-sized body — paywalled via
# the content-length strategy.
_TEASER_PAGE = (
    '<!DOCTYPE html><html lang="de"><head><title>Test</title></head>'
    "<body><article><h1>Eine Schlagzeile</h1>"
    "<p>Nur der Anriss, dann kommt das Abo-Angebot.</p>"
    "</article></body></html>"
)


def _candidates() -> list:
    # Scores 9, 8, 7 → the selector prefers lower link indices first.
    return [
        make_curated(
            article=make_article(link=LINKS[i], title=f"Artikel {i}"),
            relevance_score=9.0 - i,
        )
        for i in range(3)
    ]


def _links_in(edition) -> list[str]:
    return [a.article.link for s in edition.sections for a in s.articles]


def _mock_pages(pages: dict[str, str | httpx.Response]) -> dict[str, respx.Route]:
    routes = {}
    for link, page in pages.items():
        response = page if isinstance(page, httpx.Response) else httpx.Response(200, html=page)
        routes[link] = respx.get(link).mock(return_value=response)
    return routes


class TestSelectEdition:
    @respx.mock
    async def test_paywalled_article_replaced_by_next_best(self):
        _mock_pages({LINKS[0]: _PAYWALLED_PAGE, LINKS[1]: _FREE_PAGE, LINKS[2]: _FREE_PAGE})
        edition, skipped = await select_edition(_candidates(), CONFIG, SOURCE_IDS, OPTIONS)
        assert _links_in(edition) == [LINKS[1], LINKS[2]]
        assert skipped == [LINKS[0]]
        assert edition.stats.selected == 2

    @respx.mock
    async def test_teaser_page_replaced_by_next_best(self):
        _mock_pages({LINKS[0]: _TEASER_PAGE, LINKS[1]: _FREE_PAGE, LINKS[2]: _FREE_PAGE})
        edition, skipped = await select_edition(_candidates(), CONFIG, SOURCE_IDS, OPTIONS)
        assert _links_in(edition) == [LINKS[1], LINKS[2]]
        assert skipped == [LINKS[0]]

    @respx.mock
    async def test_nothing_paywalled_keeps_top_selection(self):
        _mock_pages({LINKS[0]: _FREE_PAGE, LINKS[1]: _FREE_PAGE})
        edition, skipped = await select_edition(_candidates(), CONFIG, SOURCE_IDS, OPTIONS)
        assert _links_in(edition) == [LINKS[0], LINKS[1]]
        assert skipped == []

    @respx.mock
    async def test_unknown_verdict_keeps_article(self):
        _mock_pages({LINKS[0]: _NO_SIGNAL_PAGE, LINKS[1]: _FREE_PAGE})
        edition, skipped = await select_edition(_candidates(), CONFIG, SOURCE_IDS, OPTIONS)
        assert _links_in(edition) == [LINKS[0], LINKS[1]]
        assert skipped == []

    @respx.mock
    async def test_failed_fetch_keeps_article(self):
        _mock_pages({LINKS[0]: httpx.Response(404), LINKS[1]: _FREE_PAGE})
        respx.get(LINKS[2]).mock(side_effect=httpx.TimeoutException("slow"))
        candidates = _candidates()
        edition, skipped = await select_edition(
            candidates, CONFIG, SOURCE_IDS, SelectorOptions(edition_size=3)
        )
        assert _links_in(edition) == LINKS
        assert skipped == []

    @respx.mock
    async def test_each_link_fetched_at_most_once(self):
        routes = _mock_pages(
            {LINKS[0]: _PAYWALLED_PAGE, LINKS[1]: _FREE_PAGE, LINKS[2]: _FREE_PAGE}
        )
        await select_edition(_candidates(), CONFIG, SOURCE_IDS, OPTIONS)
        assert all(route.call_count == 1 for route in routes.values())

    @respx.mock
    async def test_all_paywalled_yields_empty_edition(self):
        _mock_pages({link: _PAYWALLED_PAGE for link in LINKS})
        edition, skipped = await select_edition(_candidates(), CONFIG, SOURCE_IDS, OPTIONS)
        assert _links_in(edition) == []
        assert edition.stats.selected == 0
        assert skipped == sorted(LINKS)
