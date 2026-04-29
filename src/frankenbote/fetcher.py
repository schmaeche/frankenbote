"""Fetcher — downloads and parses RSS/Atom feeds.

Fetches all enabled sources concurrently using async httpx, then parses
each feed with feedparser. Returns Article objects ready for the next
pipeline stage.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from time import mktime

import feedparser
import httpx

from frankenbote.models import Article, Source

# --- Limits and identification ---

USER_AGENT = "FrankenboteBot/0.1 (personal news aggregator)"
TIMEOUT_SECONDS = 15.0
MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB — a sane upper bound


@dataclass
class FetchResult:
    """Result of fetching one source — successful or failed."""

    source: Source
    articles: list[Article]
    error: str | None = None  # None on success, error description on failure

    @property
    def ok(self) -> bool:
        return self.error is None


async def _download(client: httpx.AsyncClient, source: Source) -> bytes:
    """Download a feed's bytes with size + timeout limits.

    Raises httpx.HTTPError on network failure or invalid response.
    """
    url = str(source.url)

    # Reject plain HTTP unless the source explicitly allows it.
    if url.startswith("http://") and not source.allow_http:
        raise ValueError(f"plain HTTP not allowed for {source.id}; set allow_http: true if intentional")

    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()

    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError(f"response too large: {len(response.content)} bytes")

    return response.content


def _parse(source: Source, raw_bytes: bytes) -> list[Article]:
    """Parse feed bytes into Article objects."""
    parsed = feedparser.parse(raw_bytes)

    if parsed.bozo and not parsed.entries:
        # bozo=True means feedparser had trouble; if there are also no entries,
        # treat it as a real parse failure.
        reason = getattr(parsed, "bozo_exception", "unknown parse error")
        raise ValueError(f"feed parse failed: {reason}")

    now = datetime.now(timezone.utc)
    articles: list[Article] = []

    for entry in parsed.entries[: source.max_articles]:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue  # skip entries missing basic fields

        # Try to extract a publication date — feeds use various field names.
        published: datetime | None = None
        for key in ("published_parsed", "updated_parsed"):
            time_struct = entry.get(key)
            if time_struct:
                published = datetime.fromtimestamp(mktime(time_struct), tz=timezone.utc)
                break

        articles.append(
            Article(
                source_id=source.id,
                source_name=source.name,
                title=title,
                link=link,
                summary=(entry.get("summary") or "").strip(),
                published=published,
                fetched_at=now,
            )
        )

    return articles


async def _fetch_one(client: httpx.AsyncClient, source: Source) -> FetchResult:
    """Fetch and parse a single source. Never raises — errors land in FetchResult."""
    try:
        raw = await _download(client, source)
        articles = _parse(source, raw)
        return FetchResult(source=source, articles=articles)
    except Exception as e:  # broad on purpose — one bad feed shouldn't kill the run
        return FetchResult(source=source, articles=[], error=f"{type(e).__name__}: {e}")


async def fetch_all(sources: list[Source]) -> list[FetchResult]:
    """Fetch every source in parallel. Always returns one FetchResult per source."""
    headers = {"User-Agent": USER_AGENT}
    timeout = httpx.Timeout(TIMEOUT_SECONDS)
    limits = httpx.Limits(max_connections=10)

    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        return await asyncio.gather(*(_fetch_one(client, s) for s in sources))