"""Filter — turn raw fetched articles into a clean candidate pool.

Applies, in order:
  1. Date window (configurable, default: previous Saturday 00:00 → now)
  2. Title block list (case-insensitive substring match)
  3. URL deduplication (first occurrence wins)

This stage is deterministic. Topic-level judgements happen in the curator.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, field_validator

from frankenbote.models import Article


class FilterConfig(BaseModel):
    """Validated structure of filter.yaml."""

    class Window(BaseModel):
        anchor: str = "previous_saturday"
        timezone: str = "Europe/Berlin"

        @field_validator("anchor")
        @classmethod
        def _valid_anchor(cls, v: str) -> str:
            allowed = {"previous_saturday", "rolling_7d"}
            if v not in allowed:
                raise ValueError(
                    f"unknown anchor {v!r}; must be one of {sorted(allowed)}"
                )
            return v

        @field_validator("timezone")
        @classmethod
        def _valid_timezone(cls, v: str) -> str:
            try:
                ZoneInfo(v)
            except ZoneInfoNotFoundError as e:
                raise ValueError(f"unknown timezone {v!r}: {e}") from e
            return v

    window: Window = Window()
    drop_if_title_contains: list[str] = []


@dataclass
class FilterStats:
    """How many articles were dropped at each stage. Useful for the CLI report."""

    input_count: int = 0
    dropped_no_date_kept: int = 0  # articles whose date came from fetched_at
    dropped_outside_window: int = 0
    dropped_blocked_title: int = 0
    dropped_duplicates: int = 0
    output_count: int = 0


@dataclass
class FilterResult:
    articles: list[Article]
    window_start: datetime
    window_end: datetime
    stats: FilterStats = field(default_factory=FilterStats)


def load_filter_config(path: Path | str = "config/filter.yaml") -> FilterConfig:
    """Load filter.yaml. Returns defaults if file is missing.

    Raises pydantic.ValidationError if the file contains an invalid timezone
    or anchor. This is intentional — fail fast at startup beats failing
    cryptically deep inside the pipeline.
    """
    path = Path(path)
    if not path.exists():
        return FilterConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "filter" not in raw:
        raise ValueError(f"{path} must contain a top-level 'filter:' key")
    return FilterConfig(**raw["filter"])


def compute_window(
    config: FilterConfig,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Return (start, end) of the time window in the configured timezone.

    The end is always 'now' (in the configured timezone). The start depends
    on the anchor:
      - 'previous_saturday': most recent Saturday 00:00 strictly in the past
      - 'rolling_7d':        now - 7 days
    """
    # Safe: timezone is validated at config load time.
    tz = ZoneInfo(config.window.timezone)

    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    if config.window.anchor == "rolling_7d":
        return now - timedelta(days=7), now

    # 'previous_saturday' — find the most recent Saturday strictly before today.
    # weekday(): Mon=0 ... Sat=5, Sun=6
    days_since_sat = (now.weekday() - 5) % 7
    if days_since_sat == 0:
        # Today is Saturday — go back to the previous Saturday.
        days_since_sat = 7
    last_saturday = (now - timedelta(days=days_since_sat)).date()

    start = datetime.combine(last_saturday, time.min, tzinfo=tz)
    return start, now


def _effective_date(article: Article) -> datetime:
    """Per our spec: use published if present, else fetched_at."""
    return article.published or article.fetched_at


def _is_blocked(article: Article, blocklist: list[str]) -> bool:
    """Case-insensitive substring match against title."""
    title_lower = article.title.lower()
    return any(needle.lower() in title_lower for needle in blocklist)


def filter_articles(
    articles: list[Article],
    config: FilterConfig,
    now: datetime | None = None,
) -> FilterResult:
    """Apply the full filter pipeline. Pure function — no I/O."""
    stats = FilterStats(input_count=len(articles))
    start, end = compute_window(config, now=now)
    tz = start.tzinfo  # already validated and resolved by compute_window

    # Tracks which dates came from the fallback path. Informational only.
    for art in articles:
        if art.published is None:
            stats.dropped_no_date_kept += 1

    # Stage 1: date window (timezone-aware comparison).
    in_window: list[Article] = []
    for art in articles:
        date = _effective_date(art).astimezone(tz)
        if start <= date <= end:
            in_window.append(art)
        else:
            stats.dropped_outside_window += 1

    # Stage 2: title block list.
    after_blocklist: list[Article] = []
    for art in in_window:
        if _is_blocked(art, config.drop_if_title_contains):
            stats.dropped_blocked_title += 1
        else:
            after_blocklist.append(art)

    # Stage 3: deduplicate by URL. First occurrence wins so that the order
    # in sources.yaml acts as an implicit priority.
    seen_links: set[str] = set()
    deduped: list[Article] = []
    for art in after_blocklist:
        if art.link in seen_links:
            stats.dropped_duplicates += 1
            continue
        seen_links.add(art.link)
        deduped.append(art)

    # Sort newest first — handy for any later inspection of the JSON.
    deduped.sort(key=_effective_date, reverse=True)

    stats.output_count = len(deduped)
    return FilterResult(articles=deduped, window_start=start, window_end=end, stats=stats)