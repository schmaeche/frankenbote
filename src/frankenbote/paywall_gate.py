"""Paywall gate — keeps paywalled articles out of the final edition.

Sits between the curator and the selector, which stays a pure function:
run the selection, fetch the chosen articles' pages, and exclude every
article the paywall detector (frankenbote.paywall) definitively flags.
The freed slots are refilled by re-running the selector, so a paywalled
article is replaced by the next-best candidate instead of shrinking the
edition. Repeats until the selection contains no known-paywalled article.

Filtering at selection time — before summarize/wrap-up — avoids spending
LLM calls on articles whose full body is unreachable anyway. Only a
definitive "paywalled" verdict excludes an article: an unknown verdict or
a failed page fetch keeps it in, since a missing signal must not shrink
the edition.
"""

import asyncio
from datetime import datetime

import httpx

from frankenbote.body_fetcher import MAX_RESPONSE_BYTES, TIMEOUT_SECONDS, USER_AGENT
from frankenbote.curator import CuratorConfig
from frankenbote.models import CuratedArticle, Edition
from frankenbote.paywall import is_paywalled
from frankenbote.selector import SelectorOptions, select


async def _fetch_html(client: httpx.AsyncClient, url: str) -> str | None:
    """Fetch one article page as raw HTML — no extraction, the paywall
    detector needs the original markup. Returns None on any failure."""
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        if len(response.content) > MAX_RESPONSE_BYTES:
            return None
        return response.text
    except Exception:  # broad on purpose — one bad page shouldn't abort the run
        return None


async def _paywalled_links(client: httpx.AsyncClient, links: list[str]) -> dict[str, bool]:
    """Map each link to True when its page is definitively paywalled.

    Unreachable pages and unknown verdicts map to False — only a
    definitive signal may cost an article its slot.
    """
    htmls = await asyncio.gather(*(_fetch_html(client, link) for link in links))
    verdicts: dict[str, bool] = {}
    for link, html in zip(links, htmls):
        result = is_paywalled(html, link) if html is not None else None
        verdicts[link] = result is not None and result.paywalled
    return verdicts


async def select_edition(
    curated: list[CuratedArticle],
    config: CuratorConfig,
    source_ids_in_order: list[str],
    options: SelectorOptions | None = None,
    edition_date: datetime | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> tuple[Edition, list[str]]:
    """Build the final edition, skipping paywalled articles.

    Same parameters as selector.select(). Returns the edition plus the
    links skipped as paywalled, for reporting. Each link is fetched at
    most once; re-selection rounds only fetch the replacement articles.
    """
    checked: dict[str, bool] = {}  # link → definitively paywalled?
    excluded: set[str] = set()

    headers = {"User-Agent": USER_AGENT}
    timeout = httpx.Timeout(TIMEOUT_SECONDS)
    limits = httpx.Limits(max_connections=10)

    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        while True:
            pool = [c for c in curated if c.article.link not in excluded]
            edition = select(
                pool,
                config,
                source_ids_in_order,
                options,
                edition_date,
                window_start,
                window_end,
            )
            links = [a.article.link for sec in edition.sections for a in sec.articles]
            unchecked = [link for link in dict.fromkeys(links) if link not in checked]
            checked.update(await _paywalled_links(client, unchecked))
            newly_paywalled = {link for link in unchecked if checked[link]}
            if not newly_paywalled:
                return edition, sorted(excluded)
            excluded |= newly_paywalled
