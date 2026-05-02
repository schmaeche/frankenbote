"""Command-line interface for Frankenbote."""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import click

from frankenbote.config import load_sources
from frankenbote.fetcher import fetch_all
from frankenbote.filter import compute_window, filter_articles, load_filter_config
from frankenbote.storage import save_candidates


@click.group()
@click.version_option()
def main() -> None:
    """Frankenbote — your personal weekly news digest."""


@main.command()
def hello() -> None:
    """Print a greeting and basic environment info — used to verify setup."""
    env = os.environ.get("FRANKENBOTE_ENV", "unset")
    log_level = os.environ.get("LOG_LEVEL", "unset")
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    click.echo("Servus! Frankenbote is running.")
    click.echo(f"  Python:           {sys.version.split()[0]}")
    click.echo(f"  Environment:      {env}")
    click.echo(f"  Log level:        {log_level}")
    click.echo(f"  API key present:  {'yes' if has_api_key else 'no'}")


@main.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/sources.yaml",
    show_default=True,
    help="Path to the sources YAML file.",
)
def fetch(config: Path) -> None:
    """Fetch all enabled sources and print a summary report."""
    try:
        sources = load_sources(config)
    except ValueError as e:
        click.echo(f"❌ Failed to load config: {e}", err=True)
        sys.exit(1)

    click.echo(f"Fetching {len(sources)} source(s)…\n")
    results = asyncio.run(fetch_all(sources))

    ok_count = sum(1 for r in results if r.ok)
    total_articles = sum(len(r.articles) for r in results)

    click.echo(f"{'Source':40s} {'Articles':>8s}  Status")
    click.echo("─" * 70)

    for r in sorted(results, key=lambda x: x.source.name):
        status = f"✓" if r.ok else f"✗ {r.error}"
        click.echo(f"{r.source.name[:40]:40s} {len(r.articles):>8d}  {status}")

    click.echo("─" * 70)
    click.echo(f"{'Total':40s} {total_articles:>8d}  {ok_count}/{len(results)} sources OK")


@main.command()
@click.option(
    "--sources",
    "sources_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/sources.yaml",
    show_default=True,
)
@click.option(
    "--filter-config",
    "filter_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/filter.yaml",
    show_default=True,
)
def pipeline(sources_path: Path, filter_path: Path) -> None:
    """Fetch + filter, save candidates JSON. Foundation for upcoming AI steps."""
    try:
        sources = load_sources(sources_path)
        filter_cfg = load_filter_config(filter_path)
    except ValueError as e:
        click.echo(f"❌ Failed to load config: {e}", err=True)
        sys.exit(1)

    # 1. Fetch
    click.echo(f"Fetching {len(sources)} source(s)…")
    fetch_results = asyncio.run(fetch_all(sources))
    all_articles = [a for r in fetch_results for a in r.articles]
    failed = [r for r in fetch_results if not r.ok]
    click.echo(f"  → {len(all_articles)} articles fetched, {len(failed)} source(s) failed")
    for r in failed:
        click.echo(f"    ✗ {r.source.name}: {r.error}", err=True)

    # 2. Filter
    now = datetime.now(ZoneInfo(filter_cfg.window.timezone))
    result = filter_articles(all_articles, filter_cfg, now=now)
    s = result.stats
    click.echo("\nFilter:")
    click.echo(f"  Window:           {result.window_start.isoformat()}  →  {result.window_end.isoformat()}")
    click.echo(f"  Input:            {s.input_count}")
    click.echo(f"    no date (used fetched_at): {s.dropped_no_date_kept}")
    click.echo(f"    outside window:            {s.dropped_outside_window}")
    click.echo(f"    blocked title:             {s.dropped_blocked_title}")
    click.echo(f"    duplicates:                {s.dropped_duplicates}")
    click.echo(f"  Output:           {s.output_count} candidates")

    # 3. Save
    edition_date = result.window_end  # Saturday's edition covers up through Friday
    out_path = save_candidates(
        result.articles, edition_date, result.window_start, result.window_end
    )
    click.echo(f"\n  → Saved to {out_path}")


if __name__ == "__main__":
    main()