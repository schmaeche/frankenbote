"""Shared fixture factories for the frankenbote test suite."""

from datetime import datetime, timezone

import pytest

from frankenbote.curator import CuratorConfig
from frankenbote.models import (
    Article,
    Category,
    CuratedArticle,
    Priority,
    Source,
)

# A fixed "now" used across tests — a Wednesday, well inside any rolling window.
FIXED_NOW = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)

# A timestamp inside a typical previous-Saturday window (published on Monday).
IN_WINDOW_DATE = datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)


def make_article(**overrides) -> Article:
    defaults = dict(
        source_id="test_src",
        source_name="Test Source",
        title="Test Article",
        link="https://example.com/article/1",
        summary="A short test summary.",
        published=IN_WINDOW_DATE,
        fetched_at=FIXED_NOW,
    )
    defaults.update(overrides)
    return Article(**defaults)


def make_source(**overrides) -> Source:
    defaults = dict(
        id="test_src",
        name="Test Source",
        url="https://example.com/feed",
        category=Category.LOCAL,
    )
    defaults.update(overrides)
    return Source(**defaults)


def make_curated(**overrides) -> CuratedArticle:
    article = overrides.pop("article", make_article())
    defaults = dict(
        article=article,
        section="politik_verwaltung",
        priority=Priority.P1,
        relevance_score=7.0,
        rationale="Test rationale.",
        is_lead=False,
    )
    defaults.update(overrides)
    return CuratedArticle(**defaults)


def make_curator_config(**overrides) -> CuratorConfig:
    """Build a minimal CuratorConfig without loading any YAML files."""
    raw = dict(
        model="claude-sonnet-4-6",
        guidance="Prioritise local Franconian news.",
        priorities=[
            {"id": "P1", "label": "Lokal", "description": "Local Franconia news"},
            {"id": "P2", "label": "Regional", "description": "Bavaria news"},
            {"id": "P3", "label": "National", "description": "Germany news"},
            {"id": "P4", "label": "Überregional", "description": "International news"},
        ],
        sections=[
            {
                "id": "politik_verwaltung",
                "display_name": "Politik & Verwaltung",
                "description": "Local politics and administration.",
            },
            {
                "id": "wirtschaft",
                "display_name": "Wirtschaft",
                "description": "Economy and business.",
            },
            {
                "id": "kultur",
                "display_name": "Kultur",
                "description": "Culture and events.",
            },
        ],
    )
    raw.update(overrides)
    return CuratorConfig(**raw)
