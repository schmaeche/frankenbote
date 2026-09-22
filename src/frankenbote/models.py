"""Pydantic models — the shape of our data.

Pydantic models define what each piece of data looks like and validate it
automatically. If something tries to create an Article without a title,
or with an invalid URL, Pydantic raises a clear error.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Category(str, Enum):
    """The high-level category of a source."""

    LOCAL = "local"
    MUNICIPAL = "municipal"
    NATIONAL = "national"
    TABLOID = "tabloid"


class ScrapeConfig(BaseModel):
    """CSS selectors for a source scraped from HTML instead of a feed.

    The defaults target semantic markup (`<article>` with a heading and a
    paragraph), so most sites need no selectors at all. Selectors are
    applied inside each matched article element.
    """

    model_config = ConfigDict(extra="forbid")

    article_selector: str = Field(default="article", min_length=1)
    title_selector: str = Field(default="h2, h3", min_length=1)
    summary_selector: str = Field(default="p", min_length=1)
    # An attribute name, not a selector — it is interpolated into one.
    link_attr: str = Field(default="href", pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$")


class Source(BaseModel):
    """A configured news source — one entry from sources.yaml."""

    id: str = Field(..., min_length=1, pattern=r"^[a-z0-9_]+$")
    name: str = Field(..., min_length=1)
    url: HttpUrl
    category: Category
    enabled: bool = True
    allow_http: bool = False
    max_articles: int = Field(default=50, ge=1, le=500)
    type: Literal["rss", "scrape"] = "rss"
    scrape: ScrapeConfig | None = None  # filled with defaults for type: scrape

    @model_validator(mode="after")
    def _check_scrape_block(self) -> Self:
        if self.type == "rss" and self.scrape is not None:
            raise ValueError(f"source {self.id!r}: 'scrape:' is only valid with 'type: scrape'")
        if self.type == "scrape" and self.scrape is None:
            self.scrape = ScrapeConfig()
        return self


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
    image_url: str | None = None  # best image from the feed item, http(s) only
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
    wrap_up: str | None = None


class CuratorDecision(BaseModel):
    """Strict shape of a single decision in the LLM's JSON response.

    The tool schema the LLM must match is derived from this model
    (see llm/tasks/curate.py). article_index ties the decision back to
    the input list.
    """

    model_config = ConfigDict(extra="forbid")

    article_index: int = Field(..., ge=0)
    section: str | None
    priority: Priority
    relevance_score: float = Field(..., ge=0.0, le=10.0)
    rationale: str = Field(..., max_length=300)


class CuratorResponse(BaseModel):
    """Top-level shape the LLM must return: a list of decisions."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[CuratorDecision]


# ---------------- Curator configuration ----------------

class PrioritySpec(BaseModel):
    """One geographic priority tier, as configured in sections.yaml."""

    id: str
    label: str
    description: str


class SectionSpec(BaseModel):
    """One edition section, as configured in sections.yaml."""

    id: str
    display_name: str
    description: str


class CuratorConfig(BaseModel):
    """Validated structure of sections.yaml -> curator block.

    Lives here rather than in curator.py because the curator *task*
    (llm/tasks/curate.py) is built from it — it supplies both the section
    enum in the tool schema and the sections, tiers and guidance in the
    prompt. The model is not part of this config any more — see
    config/config.yaml.
    """

    priorities: list[PrioritySpec]
    sections: list[SectionSpec]
    guidance: str


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