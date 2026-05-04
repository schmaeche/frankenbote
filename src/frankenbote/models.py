"""Pydantic models — the shape of our data.

Pydantic models define what each piece of data looks like and validate it
automatically. If something tries to create an Article without a title,
or with an invalid URL, Pydantic raises a clear error.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class Category(str, Enum):
    """The high-level category of a source."""

    LOCAL = "local"
    MUNICIPAL = "municipal"
    NATIONAL = "national"
    TABLOID = "tabloid"


class Source(BaseModel):
    """A configured news source — one entry from sources.yaml."""

    id: str = Field(..., min_length=1, pattern=r"^[a-z0-9_]+$")
    name: str = Field(..., min_length=1)
    url: HttpUrl
    category: Category
    enabled: bool = True
    allow_http: bool = False
    max_articles: int = Field(default=50, ge=1, le=500)


class Article(BaseModel):
    """A single article retrieved from a feed.

    This is what the fetcher produces. Later pipeline stages (filter,
    summarizer, renderer) will read these.
    """

    source_id: str
    source_name: str
    title: str
    link: str  # not HttpUrl — some feeds emit unusual but valid URLs
    summary: str = ""  # short description from the feed (may be empty)
    published: datetime | None = None  # not all feeds reliably include this
    fetched_at: datetime


# ---------------- Curator models ----------------

class Priority(str, Enum):
    """Geographic relevance tier."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class CuratedArticle(BaseModel):
    """An Article enriched with the curator's per-article decisions.

    section is None when the curator decided to drop the article.
    """

    article: Article
    section: str | None  # one of the section IDs from sections.yaml, or None
    priority: Priority
    relevance_score: float = Field(..., ge=0.0, le=10.0)
    rationale: str = Field(..., max_length=300)
    is_lead: bool = False
    ai_summary: str | None = None


class CuratorDecision(BaseModel):
    """Strict shape of a single decision in the LLM's JSON response.

    The LLM gets the schema in the prompt and is told to match it exactly.
    article_index ties the decision back to the input list.
    """

    article_index: int = Field(..., ge=0)
    section: str | None
    priority: Priority
    relevance_score: float = Field(..., ge=0.0, le=10.0)
    rationale: str = Field(..., max_length=300)


class CuratorResponse(BaseModel):
    """Top-level shape the LLM must return: a list of decisions."""

    decisions: list[CuratorDecision]


# ---------------- Edition models ----------------

class EditionSection(BaseModel):
    """A section in the final edition: ID, display name, ordered articles."""

    id: str
    display_name: str
    articles: list[CuratedArticle]


class EditionStats(BaseModel):
    """Statistics about how the edition came together."""

    candidates_in: int
    curated_kept: int
    curated_dropped: int
    selected: int
    by_priority: dict[str, int]   # e.g., {"P1": 12, "P2": 6, ...}
    by_section: dict[str, int]    # e.g., {"politik_verwaltung": 5, ...}


class Edition(BaseModel):
    """The final edition, ready for rendering."""

    edition_date: str             # ISO date string
    window_start: datetime
    window_end: datetime
    sections: list[EditionSection]
    stats: EditionStats