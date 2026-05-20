"""Body fetcher — downloads an article page and extracts its main text.

Distinct from fetcher.py, which fetches RSS/Atom *feeds*. This module
fetches the *article page itself* and runs trafilatura to strip away
navigation, ads, comments and boilerplate, leaving the readable body.

Used by the summarizer's wrap-up step, which needs the full article
text — not just the short feed snippet — to write a longer digest.
"""

import asyncio

import httpx
import trafilatura

USER_AGENT = "FrankenboteBot/0.1 (personal news aggregator)"
TIMEOUT_SECONDS = 15.0
MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB — a sane upper bound

# Extractions shorter than this are treated as junk — a paywall teaser,
# an error page, or a cookie wall — and reported as a failed fetch.
MIN_BODY_CHARS = 400


async def fetch_body(client: httpx.AsyncClient, url: str) -> str | None:
    """Fetch one article page and extract its main text.

    Returns the extracted body, or None when the page could not be
    fetched, extraction failed, or the result is too short to be a real
    article body. Never raises — a failed fetch yields None so the
    caller can fall back to the feed snippet.
    """
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        if len(response.content) > MAX_RESPONSE_BYTES:
            return None
        body = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            deduplicate=True,
            favor_recall=True,
        )
    except Exception:  # broad on purpose — one bad page shouldn't abort the run
        return None

    if body is None or len(body.strip()) < MIN_BODY_CHARS:
        return None
    return body.strip()


async def fetch_bodies(urls: list[str]) -> dict[str, str | None]:
    """Fetch many article pages in parallel.

    Returns a dict mapping each input URL to its extracted body, or to
    None when that page could not be fetched or extracted. Duplicate
    URLs are fetched only once.
    """
    unique = list(dict.fromkeys(urls))
    if not unique:
        return {}

    headers = {"User-Agent": USER_AGENT}
    timeout = httpx.Timeout(TIMEOUT_SECONDS)
    limits = httpx.Limits(max_connections=10)

    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        bodies = await asyncio.gather(*(fetch_body(client, u) for u in unique))

    return dict(zip(unique, bodies))
