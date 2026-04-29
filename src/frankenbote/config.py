"""Loading and validating configuration files."""

from pathlib import Path

import yaml
from pydantic import BaseModel

from frankenbote.models import Source


class SourcesConfig(BaseModel):
    """Top-level structure of sources.yaml."""

    sources: list[Source]


def load_sources(path: Path | str = "config/sources.yaml") -> list[Source]:
    """Load the sources.yaml file and return a list of validated Sources.
    Disabled sources are filtered out — callers only see the active ones.
    Raises ValueError with a helpful message if the file is missing or invalid.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Sources config not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(raw, dict) or "sources" not in raw:
        raise ValueError(f"{path} must contain a top-level 'sources:' key")

    config = SourcesConfig(**raw)
    return [s for s in config.sources if s.enabled]