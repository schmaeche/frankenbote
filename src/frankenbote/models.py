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