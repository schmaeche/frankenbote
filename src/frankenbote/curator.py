"""Curator — uses the LLM to classify candidate articles.

Per article, the curator decides:
  - section:         which section it belongs to (or None to drop)
  - priority:        geographic relevance tier (P1..P4)
  - relevance_score: 0-10, ranking within that priority
  - rationale:       one-sentence justification

This file is responsible for:
  - Loading sections.yaml
  - Running the curator task (`llm/tasks/curate.py`, which owns the prompts,
    the tool schema and the reading of the model's answer) through the
    injected LLMClient, which owns model selection and retries
  - Producing CuratedArticle objects ready for the selector / renderer

It contains no prompt text and no index bookkeeping — the task returns one
decision per candidate, in order.

It does NOT decide which articles end up in the final edition — that's
the selector's job.
"""

from pathlib import Path

import click
import yaml

from frankenbote.llm import LLMClient
from frankenbote.llm.tasks import curator_task
from frankenbote.models import (
    Article,
    CuratedArticle,
    CuratorConfig,
)

# Re-exported for the pipeline stages that type against it (selector,
# paywall_gate); it lives in models.py so llm/tasks/curate.py can build the
# task from it without importing this module.
__all__ = ["CuratorConfig", "curate", "load_curator_config"]


# -------- Config loading --------


def load_curator_config(path: Path | str = "config/sections.yaml") -> CuratorConfig:
    """Load and validate sections.yaml."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Sections config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "curator" not in raw:
        raise ValueError(f"{path} must contain a top-level 'curator:' key")
    _reject_moved_model_keys(raw, path)
    return CuratorConfig(**raw["curator"])


def _reject_moved_model_keys(raw: dict, path: Path) -> None:
    """Model names moved to config/config.yaml — fail loudly on stale keys
    instead of silently ignoring a model the user thinks they configured."""
    curator_block = raw.get("curator")
    if isinstance(curator_block, dict) and "model" in curator_block:
        raise ValueError(
            f"{path}: 'curator.model' has moved to config/config.yaml "
            "(llm.models.curator)"
        )
    if "summarizer" in raw:
        raise ValueError(
            f"{path}: the 'summarizer:' block has moved to config/config.yaml "
            "(llm.models.summarizer / llm.models.wrap_up)"
        )


# -------- Public API --------


def curate(
    candidates: list[Article],
    config: CuratorConfig,
    client: LLMClient,
) -> list[CuratedArticle]:
    """Classify candidates with the LLM via tool use.

    Returns one CuratedArticle per input candidate, preserving order.
    Raises RuntimeError on persistent failure; the client retries once and
    saves debug context to disk.
    """
    if not candidates:
        return []

    call_desc = (
        "Batches API, polling until done"
        if client.use_batch
        else "streaming API call, may take 3-7 minutes"
    )

    def announce(attempt: int) -> None:
        click.echo(
            f"\nCurating {len(candidates)} candidates (attempt {attempt})… "
            f"({call_desc})"
        )

    outcome = client.run_task(
        curator_task(config), candidates, on_attempt=announce
    )
    for note in outcome.notes:
        title = candidates[note.index].title if note.index is not None else "?"
        click.echo(f"  ⚠ {title[:60]}: {note.reason}", err=True)

    return [
        CuratedArticle(
            article=article,
            section=decision.section,
            priority=decision.priority,
            relevance_score=decision.relevance_score,
            rationale=decision.rationale,
        )
        for article, decision in zip(candidates, outcome.values, strict=True)
    ]
