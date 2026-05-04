"""Renderer — turn edition JSON into HTML output for upload.

Reads:
  - data/editions/YYYY-MM-DD.json (final editions, written by the selector)
  - templates/*.j2                (Jinja2 templates)
  - assets/*                      (CSS, SVG)

Writes (to output/):
  - editions/YYYY-MM-DD.html      (one per edition)
  - index.html                    (archive listing the last N editions)
  - assets/style.css              (copy of source asset)
  - assets/frankenrechen.svg      (copy of source asset)

Retention: only the most recent N editions are written; older HTML files
in the output directory are removed. The data/editions/*.json files are
kept indefinitely — they're the canonical source of truth.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from frankenbote.models import Edition
from frankenbote.storage import EDITIONS_DIR, load_edition


# ---------- Paths (configurable) ----------

DEFAULT_TEMPLATES_DIR = Path("templates")
DEFAULT_ASSETS_DIR = Path("assets")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_RETENTION = 5


@dataclass
class RenderConfig:
    """Configurable knobs for rendering. Reasonable defaults."""

    templates_dir: Path = DEFAULT_TEMPLATES_DIR
    assets_dir: Path = DEFAULT_ASSETS_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    retention: int = DEFAULT_RETENTION
    sections_config: Path = Path("config/sections.yaml")


# ---------- Public API ----------


def render_all(config: RenderConfig | None = None) -> dict[str, int]:
    """Render every edition we keep, the index, and copy assets.

    Returns a small stats dict: {'editions_rendered', 'editions_pruned', ...}.
    """
    config = config or RenderConfig()

    editions = _list_recent_editions(config.retention)
    output_editions_dir = config.output_dir / "editions"
    output_assets_dir = config.output_dir / "assets"
    output_editions_dir.mkdir(parents=True, exist_ok=True)
    output_assets_dir.mkdir(parents=True, exist_ok=True)

    env = _make_jinja_env(config.templates_dir)
    priority_labels = _load_priority_labels(config.sections_config)

    # Render every kept edition.
    for edition in editions:
        html = _render_edition(env, edition, priority_labels)
        out_path = output_editions_dir / f"{edition.edition_date}.html"
        out_path.write_text(html, encoding="utf-8")

    # Render the archive index page.
    index_entries = [_index_entry(e) for e in editions]
    index_html = _render_index(env, index_entries)
    (config.output_dir / "index.html").write_text(index_html, encoding="utf-8")

    # Copy assets (CSS, SVG). Cheap; do it every render so changes propagate.
    pruned_count = _prune_old_html(output_editions_dir, kept_dates={e.edition_date for e in editions})
    copied = _copy_assets(config.assets_dir, output_assets_dir)

    return {
        "editions_rendered": len(editions),
        "editions_pruned": pruned_count,
        "assets_copied": copied,
    }


# ---------- Internals ----------


def _make_jinja_env(templates_dir: Path) -> Environment:
    """Build the Jinja2 environment with autoescape on."""
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(enabled_extensions=("j2", "html")),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _list_recent_editions(retention: int) -> list[Edition]:
    """Scan data/editions/ for final edition JSON files, return newest first."""
    if not EDITIONS_DIR.exists():
        return []

    # Final-edition files have the bare 'YYYY-MM-DD.json' shape — not the
    # '-candidates.json' or '-curated-raw.json' intermediates.
    candidates = sorted(
        (p for p in EDITIONS_DIR.glob("*.json")
         if not p.stem.endswith(("-candidates", "-curated-raw"))),
        key=lambda p: p.stem,
        reverse=True,
    )

    editions: list[Edition] = []
    for path in candidates[:retention]:
        try:
            edition_date = datetime.fromisoformat(path.stem)
            editions.append(load_edition(edition_date))
        except (ValueError, FileNotFoundError):
            continue
    return editions


def _render_edition(
    env: Environment,
    edition: Edition,
    priority_labels: dict[str, str],
) -> str:
    template = env.get_template("edition.html.j2")
    return template.render(
        edition=edition,
        priority_labels=priority_labels,
        generated_at=datetime.now(),
    )


def _render_index(env: Environment, entries: list[dict]) -> str:
    template = env.get_template("index.html.j2")
    return template.render(editions=entries)


def _index_entry(edition: Edition) -> dict:
    """Shape an Edition for the index template."""
    iso = edition.edition_date  # 'YYYY-MM-DD'
    parsed = datetime.fromisoformat(iso)
    return {
        "filename": f"editions/{iso}.html",
        "date_label": parsed.strftime("%d.%m.%Y"),
        "article_count": edition.stats.selected,
    }


def _prune_old_html(output_editions_dir: Path, kept_dates: set[str]) -> int:
    """Remove any edition HTML files not in kept_dates. Returns count removed."""
    removed = 0
    for path in output_editions_dir.glob("*.html"):
        if path.stem not in kept_dates:
            path.unlink()
            removed += 1
    return removed


def _copy_assets(src_dir: Path, dst_dir: Path) -> int:
    """Copy every file from assets/ to output/assets/. Returns count copied."""
    if not src_dir.exists():
        return 0
    copied = 0
    for src in src_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, dst_dir / src.name)
            copied += 1
    return copied


def _load_priority_labels(sections_config: Path) -> dict[str, str]:
    """Build a {priority_id: label} lookup from sections.yaml.

    Returns an empty dict if the config can't be loaded — the template
    falls back to the raw 'P1' / 'P2' values in that case.
    """
    try:
        from frankenbote.curator import load_curator_config
        config = load_curator_config(sections_config)
        return {p.id: p.label for p in config.priorities}
    except Exception:
        return {}