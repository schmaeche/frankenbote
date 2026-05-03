"""Selector — turn the curator's full set of decisions into a final edition.

Algorithm:
  1. Drop curator-rejected articles (section is None).
  2. Compute per-priority quotas from soft targets and the requested
     edition size.
  3. Sort all eligible articles by (priority asc, relevance_score desc,
     source_order asc, published desc) — the priority field comes first
     so quotas can be respected; later fields break ties deterministically.
  4. Greedily pick articles, respecting per-priority quotas:
        - while quotas remain unfilled and articles remain, pick the
          best-scoring article whose priority still has slots
        - then fill any leftover edition slots from the remaining pool
          regardless of priority (so soft targets can give way to supply)
  5. Group selected articles by section, sort within each section by
     relevance_score (descending) with deterministic tiebreakers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from frankenbote.curator import CuratorConfig
from frankenbote.models import (
    CuratedArticle,
    Edition,
    EditionSection,
    EditionStats,
    Priority,
)


# Soft target distribution across priorities. Sums to 1.0.
DEFAULT_TARGETS: dict[Priority, float] = {
    Priority.P1: 0.50,
    Priority.P2: 0.25,
    Priority.P3: 0.15,
    Priority.P4: 0.10,
}

DEFAULT_EDITION_SIZE = 25


@dataclass(frozen=True)
class SelectorOptions:
    """Configurable knobs for the selector. All have sensible defaults."""

    edition_size: int = DEFAULT_EDITION_SIZE
    targets: dict[Priority, float] | None = None  # None → DEFAULT_TARGETS

    @property
    def effective_targets(self) -> dict[Priority, float]:
        return self.targets or DEFAULT_TARGETS


def _compute_quotas(size: int, targets: dict[Priority, float]) -> dict[Priority, int]:
    """Round target percentages to integer slot counts that sum to `size`.

    Uses largest-remainder rounding so we don't lose articles to floor()
    or gain phantom ones from round().
    """
    raw = {p: size * fraction for p, fraction in targets.items()}
    floors = {p: int(v) for p, v in raw.items()}
    remainders = sorted(
        ((p, raw[p] - floors[p]) for p in raw),
        key=lambda x: x[1],
        reverse=True,
    )
    leftover = size - sum(floors.values())
    for i in range(leftover):
        floors[remainders[i % len(remainders)][0]] += 1
    return floors


def _source_order_index(source_id: str, source_ids_in_order: list[str]) -> int:
    """Earlier in sources.yaml = lower index = preferred on ties.

    Articles from unknown sources go to the end (high index).
    """
    try:
        return source_ids_in_order.index(source_id)
    except ValueError:
        return len(source_ids_in_order)


def select(
    curated: list[CuratedArticle],
    config: CuratorConfig,
    source_ids_in_order: list[str],
    options: SelectorOptions | None = None,
    edition_date: datetime | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> Edition:
    """Build the final Edition from curated articles.

    Pure function — no I/O. Caller owns timestamps; if not given, edition_date
    defaults to 'now' and the window timestamps default to current time too.
    """
    options = options or SelectorOptions()

    eligible = [c for c in curated if c.section is not None]
    quotas = _compute_quotas(options.edition_size, options.effective_targets)

    # Stable sort with the deterministic tiebreaker key.
    # Sort key: (priority order, -relevance_score, source order, -published)
    # We negate score and recency by inverting them inside a key tuple.
    def sort_key(c: CuratedArticle):
        priority_idx = list(Priority).index(c.priority)
        published_ts = c.article.published.timestamp() if c.article.published else 0.0
        return (
            priority_idx,
            -c.relevance_score,
            _source_order_index(c.article.source_id, source_ids_in_order),
            -published_ts,
        )

    pool = sorted(eligible, key=sort_key)

    # Greedy selection respecting quotas.
    chosen: list[CuratedArticle] = []
    remaining_quota = dict(quotas)
    leftovers: list[CuratedArticle] = []

    for c in pool:
        if remaining_quota.get(c.priority, 0) > 0:
            chosen.append(c)
            remaining_quota[c.priority] -= 1
        else:
            leftovers.append(c)

    # Fill any unused slots (a priority's supply was short) from leftovers,
    # preserving the same global ordering.
    while len(chosen) < options.edition_size and leftovers:
        chosen.append(leftovers.pop(0))

    # Group by section, preserving the section order from sections.yaml.
    section_order = [s.id for s in config.sections]
    section_displays = {s.id: s.display_name for s in config.sections}

    by_section: dict[str, list[CuratedArticle]] = {sid: [] for sid in section_order}
    for c in chosen:
        if c.section in by_section:
            by_section[c.section].append(c)

    # Within each section, sort by relevance_score desc with the same
    # deterministic tiebreakers (but no priority key — display order is
    # purely about how good the article is).
    def section_sort_key(c: CuratedArticle):
        published_ts = c.article.published.timestamp() if c.article.published else 0.0
        return (
            -c.relevance_score,
            _source_order_index(c.article.source_id, source_ids_in_order),
            -published_ts,
        )

    sections_out: list[EditionSection] = []
    for sid in section_order:
        articles = sorted(by_section[sid], key=section_sort_key)
        if not articles:
            continue  # skip empty sections in the output
        sections_out.append(EditionSection(
            id=sid,
            display_name=section_displays[sid],
            articles=articles,
        ))

    # Stats
    by_priority_count: dict[str, int] = {}
    by_section_count: dict[str, int] = {}
    for c in chosen:
        by_priority_count[c.priority.value] = by_priority_count.get(c.priority.value, 0) + 1
        if c.section:
            by_section_count[c.section] = by_section_count.get(c.section, 0) + 1

    now = datetime.now()
    return Edition(
        edition_date=(edition_date or now).date().isoformat(),
        window_start=window_start or now,
        window_end=window_end or now,
        sections=sections_out,
        stats=EditionStats(
            candidates_in=len(curated),
            curated_kept=len(eligible),
            curated_dropped=len(curated) - len(eligible),
            selected=len(chosen),
            by_priority=by_priority_count,
            by_section=by_section_count,
        ),
    )