"""Fetcher — downloads and parses RSS/Atom feeds.

Fetches all enabled sources concurrently using async httpx, then parses
each feed with feedparser. Returns Article objects ready for the next
pipeline stage.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from time import mktime
from urllib.parse import urlsplit

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


# <img> is a void element, so a single-tag pattern is enough; the stray
# closing form some feeds emit is matched by the optional slash.
_IMG_TAG_RE = re.compile(r"</?img\b[^>]*>", re.IGNORECASE)


class _FirstImgSrcParser(HTMLParser):
    """Collects the src of the first <img> in an HTML snippet."""

    def __init__(self) -> None:
        super().__init__()
        self.src: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img" and self.src is None:
            self.src = dict(attrs).get("src")


def _is_safe_image_url(url: str) -> bool:
    """Feed-supplied URLs are untrusted — only allow http(s) schemes."""
    try:
        return urlsplit(url).scheme in ("http", "https")
    except ValueError:
        return False


def _first_img_src(html: str) -> str | None:
    parser = _FirstImgSrcParser()
    parser.feed(html)
    return parser.src


def _strip_img_tags(html: str) -> str:
    """Remove all <img> tags from a description.

    The image is extracted into Article.image_url instead; leaving the tags
    in would leak markup into the LLM prompts and duplicate the image in
    the rendered summary.
    """
    return _IMG_TAG_RE.sub("", html)


def _extract_image_url(entry, description: str) -> str | None:
    """Best image URL for a feed entry, or None.

    Priority: media:content, then media:thumbnail, then the first <img>
    in the description HTML. Candidates with non-http(s) schemes are
    discarded and the next source is tried.
    """
    candidates: list[str] = []
    for key in ("media_content", "media_thumbnail"):
        for media in entry.get(key) or []:
            url = (media.get("url") or "").strip()
            if url:
                candidates.append(url)

    img_src = _first_img_src(description)
    if img_src and img_src.strip():
        candidates.append(img_src.strip())

    for url in candidates:
        if _is_safe_image_url(url):
            return url
    return None


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

        summary_html = (entry.get("summary") or "").strip()

        articles.append(
            Article(
                source_id=source.id,
                source_name=source.name,
                title=title,
                link=link,
                summary=_strip_img_tags(summary_html).strip(),
                image_url=_extract_image_url(entry, summary_html),
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