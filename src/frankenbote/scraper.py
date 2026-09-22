"""Scraper — extracts articles from HTML pages that have no RSS feed.

Sites without a feed often still mark up their teasers semantically: an
`<article>` holding a heading, a paragraph and a link. `parse()` turns such
a page into the same Article objects the RSS path produces, so both land
in one candidate pool. Only the listing page is read; article pages are
never fetched.

The network side (robots.txt, per-host politeness delay) lives in
`ScrapeSession`; the download itself is shared with the RSS path in
fetcher.py, which also dispatches between the two.

Example sources.yaml entry — every `scrape:` key is optional:

    - id: nuernberg_stadtportal
      name: "Stadt Nürnberg – Stadtportal"
      url: "https://www.nuernberg.de/internet/stadtportal/index.html"
      category: municipal
      type: scrape
      scrape:
        article_selector: "article[data-publish]"
        title_selector: "h2, h3"
        summary_selector: "p"
        link_attr: "href"
"""

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup, FeatureNotFound, Tag
from dateutil import parser as dateparser
from dateutil.parser import isoparse

from frankenbote.models import Article, ScrapeConfig, Source

logger = logging.getLogger(__name__)

MIN_HOST_INTERVAL_SECONDS = 1.0

# Naive dates on German sites are local time.
_LOCAL_TZ = ZoneInfo("Europe/Berlin")


# --- Network: robots.txt and politeness ---


class ScrapeSession:
    """Per-run state shared by all scraped sources.

    Caches one parsed robots.txt per origin and enforces a minimum interval
    between requests to the same host, so several sources on one site
    don't hammer it in parallel.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str,
        min_interval: float = MIN_HOST_INTERVAL_SECONDS,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._min_interval = min_interval
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser] = {}
        self._robots_locks: dict[str, asyncio.Lock] = {}

    async def wait_turn(self, url: str) -> None:
        """Sleep until this host may receive the next request."""
        host = urlsplit(url).netloc
        loop = asyncio.get_running_loop()
        async with self._host_locks.setdefault(host, asyncio.Lock()):
            last = self._last_request.get(host)
            if last is not None:
                delay = last + self._min_interval - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            self._last_request[host] = loop.time()

    async def ensure_allowed(self, url: str) -> None:
        """Raise PermissionError if robots.txt disallows `url` for us.

        Network failures and 5xx on robots.txt raise httpx errors, which
        fail the source the same way a failed feed download does.
        """
        rules = await self._robots_for(url)
        if not rules.can_fetch(self._user_agent, url):
            raise PermissionError(f"robots.txt disallows {url}")

    async def _robots_for(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        async with self._robots_locks.setdefault(origin, asyncio.Lock()):
            if origin not in self._robots:
                self._robots[origin] = await self._load_robots(f"{origin}/robots.txt")
            return self._robots[origin]

    async def _load_robots(self, robots_url: str) -> RobotFileParser:
        await self.wait_turn(robots_url)
        response = await self._client.get(robots_url, follow_redirects=True)
        rules = RobotFileParser(robots_url)
        # Same status handling as urllib.robotparser.RobotFileParser.read().
        if response.status_code in (401, 403):
            rules.parse(["User-agent: *", "Disallow: /"])
        elif 400 <= response.status_code < 500:
            rules.parse([])  # no robots.txt — nothing is restricted
        else:
            response.raise_for_status()
            rules.parse(response.text.splitlines())
        return rules


# --- Parsing ---


def _make_soup(raw: bytes) -> BeautifulSoup:
    try:
        return BeautifulSoup(raw, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(raw, "html.parser")


def _clean_text(element: Tag) -> str:
    return " ".join(element.get_text(" ", strip=True).split())


def _is_http_url(url: str) -> bool:
    try:
        return urlsplit(url).scheme in ("http", "https")
    except ValueError:
        return False


def _first_text(node: Tag, selector: str, exclude: str = "") -> tuple[Tag | None, str]:
    """First element matching `selector` with non-empty text (other than `exclude`)."""
    for element in node.select(selector):
        text = _clean_text(element)
        if text and text != exclude:
            return element, text
    return None, ""


def _attr_url(element: Tag, attr: str, base_url: str) -> str | None:
    value = element.get(attr)
    if not isinstance(value, str) or not value.strip() or value.strip().startswith("#"):
        return None
    url = urljoin(base_url, value.strip())
    return url if _is_http_url(url) else None


def _extract_link(node: Tag, title_el: Tag | None, attr: str, base_url: str) -> str | None:
    """Article link: prefer the one in the heading, then the article's own.

    For `href` only anchors count — an `<svg><use href>` icon inside the
    teaser is not a link to the article.
    """
    selector = f"a[{attr}]" if attr == "href" else f"[{attr}]"
    candidates: list[Tag] = []
    if title_el is not None:
        candidates += title_el.select(selector)
    candidates.append(node)
    candidates += node.select(selector)
    for element in candidates:
        url = _attr_url(element, attr, base_url)
        if url:
            return url
    return None


def _extract_image(node: Tag, base_url: str) -> str | None:
    for img in node.select("img[src]"):
        url = _attr_url(img, "src", base_url)
        if url:
            return url
    return None


_NUMBER_WORDS = {"ein": 1, "eine": 1, "einem": 1, "einer": 1, "a": 1, "an": 1, "one": 1}
_UNITS = {
    "minute": "minutes", "minuten": "minutes", "min": "minutes", "minutes": "minutes",
    "stunde": "hours", "stunden": "hours", "std": "hours", "hour": "hours", "hours": "hours",
    "tag": "days", "tage": "days", "tagen": "days", "day": "days", "days": "days",
    "woche": "weeks", "wochen": "weeks", "week": "weeks", "weeks": "weeks",
}
_AMOUNT = r"(\d+|" + "|".join(_NUMBER_WORDS) + r")"
_UNIT = r"(" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")\.?"
_RELATIVE_RES = (
    re.compile(rf"^vor\s+{_AMOUNT}\s+{_UNIT}$", re.IGNORECASE),  # vor 2 Tagen
    re.compile(rf"^{_AMOUNT}\s+{_UNIT}\s+ago$", re.IGNORECASE),  # 5 days ago
)
_DAY_WORDS = {"heute": 0, "today": 0, "gestern": 1, "yesterday": 1}
_DAY_WORD_RE = re.compile(
    r"^(heute|today|gestern|yesterday)(?:\s*,?\s*(?:um\s+)?(\d{1,2}):(\d{2})(?:\s*uhr)?)?$",
    re.IGNORECASE,
)


# German words translated to English before dateutil sees the text.
# Filler words map to a space: "am" would otherwise be read as a.m.
_GERMAN_WORDS = {
    "januar": "January", "jänner": "January", "jän": "Jan", "februar": "February",
    "märz": "March", "mär": "Mar", "mai": "May", "juni": "June", "juli": "July",
    "okt": "Oct", "oktober": "October", "dez": "Dec", "dezember": "December",
    "montag": "Monday", "dienstag": "Tuesday", "mittwoch": "Wednesday",
    "donnerstag": "Thursday", "freitag": "Friday", "samstag": "Saturday", "sonntag": "Sunday",
    "mo": "Mon", "di": "Tue", "mi": "Wed", "do": "Thu", "fr": "Fri", "sa": "Sat", "so": "Sun",
    "am": " ", "um": " ", "uhr": " ", "stand": " ",
}
_GERMAN_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(_GERMAN_WORDS, key=len, reverse=True)) + r")\b:?", re.IGNORECASE
)
_PARSER_INFO = dateparser.parserinfo(dayfirst=True)


def _to_english(text: str) -> str:
    return _GERMAN_WORD_RE.sub(lambda m: _GERMAN_WORDS[m.group(1).lower()], text)


def parse_date(text: str, now: datetime) -> datetime | None:
    """Parse a date as found in teaser markup; None if it can't be read.

    Understands Unix timestamps (seconds or milliseconds, as in
    `data-publish="1789711200000"`), relative dates ("vor 2 Tagen",
    "5 days ago", "gestern, 14:30") and absolute dates via dateutil
    (ISO 8601, "7. Mai 2026", "May 7, 2026", "07.05.2026"). Numeric
    dates are read day-first. Naive results are taken as German local
    time; the return value is always timezone-aware UTC.
    """
    text = " ".join(text.split())
    if not text:
        return None

    if text.isdigit():
        if len(text) == 13:
            return datetime.fromtimestamp(int(text) / 1000, tz=UTC)
        if len(text) == 10:
            return datetime.fromtimestamp(int(text), tz=UTC)
        return None

    for pattern in _RELATIVE_RES:
        match = pattern.match(text)
        if match:
            amount_text, unit = match.group(1).lower(), match.group(2).lower()
            amount = _NUMBER_WORDS.get(amount_text) or int(amount_text)
            return now - timedelta(**{_UNITS[unit]: amount})

    match = _DAY_WORD_RE.match(text)
    if match:
        local_now = now.astimezone(_LOCAL_TZ)
        day = local_now - timedelta(days=_DAY_WORDS[match.group(1).lower()])
        if match.group(2):
            day = day.replace(hour=int(match.group(2)), minute=int(match.group(3)), second=0, microsecond=0)
        return day.astimezone(UTC)

    try:
        # ISO first: day-first parsing would swap 2026-05-07 into July.
        parsed = isoparse(text)
    except ValueError:
        default = now.astimezone(_LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        try:
            parsed = dateparser.parse(_to_english(text), parserinfo=_PARSER_INFO, default=default)
        except (ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_LOCAL_TZ)
    return parsed.astimezone(UTC)


def _date_candidates(node: Tag) -> list[tuple[str, str]]:
    """(origin, raw value) pairs that may hold the publish date, best first."""
    candidates: list[tuple[str, str]] = []
    for time_el in node.select("time"):
        value = time_el.get("datetime")
        if isinstance(value, str):
            candidates.append(("<time datetime>", value))
        candidates.append(("<time> text", _clean_text(time_el)))
    for element in node.select('[itemprop="datePublished"]'):
        for attr in ("content", "datetime"):
            value = element.get(attr)
            if isinstance(value, str):
                candidates.append((f"itemprop {attr}", value))
    # Attributes on the article element itself, e.g. data-publish.
    for keyword in ("publish", "date"):
        for name, value in node.attrs.items():
            if keyword in name.lower() and isinstance(value, str):
                candidates.append((f"@{name}", value))
    return [(origin, value) for origin, value in candidates if value.strip()]


def _extract_published(node: Tag, now: datetime, label: str) -> datetime | None:
    candidates = _date_candidates(node)
    for origin, value in candidates:
        published = parse_date(value, now)
        if published is not None:
            logger.debug("%s: date %s from %s", label, published.isoformat(), origin)
            return published
    if candidates:
        tried = ", ".join(f"{origin}={value!r}" for origin, value in candidates)
        logger.warning("%s: unparseable date (%s) — published left empty", label, tried)
    else:
        logger.debug("%s: no date found — published left empty", label)
    return None


def _extract(node: Tag, source: Source, config: ScrapeConfig, now: datetime, label: str) -> Article | None:
    base_url = str(source.url)
    title_el, title = _first_text(node, config.title_selector)
    _, summary = _first_text(node, config.summary_selector, exclude=title)
    link = _extract_link(node, title_el, config.link_attr, base_url)

    missing = [name for name, value in (("url", link), ("title", title), ("summary", summary)) if not value]
    if missing or link is None:
        logger.warning("%s: skipped, missing %s", label, ", ".join(missing))
        return None

    return Article(
        source_id=source.id,
        source_name=source.name,
        title=title,
        link=link,
        summary=summary,
        image_url=_extract_image(node, base_url),
        published=_extract_published(node, now, label),
        fetched_at=now,
    )


def parse(source: Source, raw: bytes, now: datetime | None = None) -> list[Article]:
    """Extract up to `source.max_articles` articles from a listing page.

    Items missing a link, title or summary are skipped with a warning; a
    missing or unreadable date leaves `published` as None. An error in one
    item never stops the others.
    """
    config = source.scrape or ScrapeConfig()
    now = now or datetime.now(UTC)
    soup = _make_soup(raw)
    nodes = soup.select(config.article_selector)
    logger.debug("%s: %d element(s) match %r", source.id, len(nodes), config.article_selector)

    articles: list[Article] = []
    for position, node in enumerate(nodes, start=1):
        if len(articles) >= source.max_articles:
            break
        label = f"{source.id} item {position}"
        try:
            article = _extract(node, source, config, now, label)
        except Exception as e:  # broad on purpose — one odd teaser shouldn't drop the page
            logger.warning("%s: skipped, %s: %s", label, type(e).__name__, e)
            continue
        if article is not None:
            articles.append(article)
    return articles
