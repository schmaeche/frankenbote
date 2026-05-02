"""Storage — read/write edition JSON files under data/editions/.

The JSON file is the canonical record of an edition's content (per
hybrid rendering architecture). The HTML is regenerable from it.
"""

import json
from datetime import datetime
from pathlib import Path

from frankenbote.models import Article


EDITIONS_DIR = Path("data/editions")


def candidates_path(edition_date: datetime) -> Path:
    """Path to the candidates JSON for a given edition date."""
    return EDITIONS_DIR / f"{edition_date.date().isoformat()}-candidates.json"


def save_candidates(
    articles: list[Article],
    edition_date: datetime,
    window_start: datetime,
    window_end: datetime,
) -> Path:
    """Save filtered candidate articles to disk. Returns the path written."""
    EDITIONS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "edition_date": edition_date.date().isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "article_count": len(articles),
        "articles": [a.model_dump(mode="json") for a in articles],
    }

    path = candidates_path(edition_date)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_candidates(edition_date: datetime) -> list[Article]:
    """Load previously saved candidates. Useful for re-running later stages."""
    path = candidates_path(edition_date)
    if not path.exists():
        raise FileNotFoundError(f"No candidates file at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Article(**a) for a in raw["articles"]]