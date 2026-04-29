"""Command-line interface for Frankenbote."""

import asyncio
import os
import sys
from pathlib import Path

import click

from frankenbote.config import load_sources
from frankenbote.fetcher import fetch_all

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


if __name__ == "__main__":
    main()