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
from frankenbote.filter import filter_articles, load_filter_config
from frankenbote.models import Priority
from frankenbote.selector import select, SelectorOptions, load_selector_targets
from frankenbote.storage import ( 
    load_edition,
    save_candidates,
    load_candidates,
    save_curated_raw,
    load_curated_raw,
    save_edition
)
from frankenbote.curator import load_curator_config, curate
from frankenbote.renderer import render_all
from frankenbote.summarizer import (
    generate_wrap_ups,
    load_summarizer_config,
    summarize_edition,
)
from frankenbote.publisher import load_publisher_config_from_env, publish


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
@click.option(
    "--sections-config",
    "sections_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/sections.yaml",
    show_default=True,
)
@click.option(
    "--size",
    type=click.IntRange(min=5, max=100),
    default=25,
    show_default=True,
)
@click.option(
    "--no-curate",
    is_flag=True,
    help="Stop after the filter step (don't call the LLM). Useful for dev.",
)
@click.option(
    "--wrap-up",
    "wrap_up",
    is_flag=True,
    help=(
        "Generate longer wrap-ups for lead articles. Off by default — "
        "wrapping up source texts may raise legal concerns."
    ),
)
def pipeline(
    sources_path: Path,
    filter_path: Path,
    sections_path: Path,
    size: int,
    no_curate: bool,
    wrap_up: bool,
) -> None:
    """Run the full pipeline: fetch → filter → curate → select → summarize → render."""
    try:
        sources = load_sources(sources_path)
        filter_cfg = load_filter_config(filter_path)
        curator_cfg = load_curator_config(sections_path)
        summarizer_cfg = load_summarizer_config(sections_path)
        targets = load_selector_targets(sections_path)
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
    click.echo(f"  Window:  {result.window_start.isoformat()}  →  {result.window_end.isoformat()}")
    click.echo(f"  Kept:    {s.output_count} / {s.input_count}")

    edition_date = result.window_end
    save_candidates(result.articles, edition_date, result.window_start, result.window_end)

    if no_curate:
        click.echo("\nStopped before curation (--no-curate).")
        return

    # 3. Curate
    click.echo(f"\nCurating {s.output_count} article(s) using {curator_cfg.model}…")
    try:
        curated = curate(result.articles, curator_cfg)
    except RuntimeError as e:
        click.echo(f"❌ Curator failed: {e}", err=True)
        sys.exit(1)
    save_curated_raw(curated, edition_date)

    # 4. Select
    options = SelectorOptions(edition_size=size, targets=targets)
    edition = select(
        curated=curated,
        config=curator_cfg,
        source_ids_in_order=[s.id for s in sources],
        options=options,
        edition_date=edition_date,
        window_start=result.window_start,
        window_end=result.window_end,
    )
    es = edition.stats
    effective = options.effective_targets
    target_str = " ".join(f"{p}={effective[Priority(p)]:.0%}" for p in ("P1", "P2", "P3", "P4"))
    click.echo(f"\nFinal edition: {es.selected} articles")
    click.echo(f"By priority (target: {target_str}):")
    for p in ("P1", "P2", "P3", "P4"):
        count = es.by_priority.get(p, 0)
        pct = (count / es.selected * 100) if es.selected else 0.0
        click.echo(f"  {p}: {count:3d}  ({pct:.0f}%)")
    click.echo("By section:")
    for sec in edition.sections:
        click.echo(f"  {sec.display_name}: {len(sec.articles)}")

    out_path = save_edition(edition, edition_date)
    click.echo(f"\n  → Saved to {out_path}")

    # 5. Summarize
    try:
        edition = summarize_edition(edition, model=summarizer_cfg.model)
    except RuntimeError as e:
        click.echo(f"❌ Summarizer failed: {e}", err=True)
        sys.exit(1)
    save_edition(edition, edition_date)
    with_summary = sum(
        1 for s in edition.sections for a in s.articles if a.ai_summary
    )
    total = sum(len(s.articles) for s in edition.sections)
    click.echo(f"  → {with_summary}/{total} summaries written")

    # 5b. Wrap-ups for lead articles (opt-in via --wrap-up)
    if wrap_up:
        try:
            edition = generate_wrap_ups(
                edition, model=summarizer_cfg.wrap_up_model or summarizer_cfg.model
            )
        except RuntimeError as e:
            click.echo(f"❌ Wrap-up generation failed: {e}", err=True)
            sys.exit(1)
        save_edition(edition, edition_date)
        with_wrap_up = sum(
            1 for s in edition.sections for a in s.articles if a.wrap_up
        )
        click.echo(f"  → {with_wrap_up} wrap-up(s) written")
    else:
        click.echo("  → Skipped wrap-ups (enable with --wrap-up)")

    # 6. Render
    stats = render_all()
    click.echo(f"\nRendered {stats['editions_rendered']} edition(s) to output/.")

    # 7. Publish
    try:
        pub_config = load_publisher_config_from_env()
        pub_stats = publish(pub_config)
        click.echo(
            f"\nPublished to {pub_config.host}: "
            f"{pub_stats['uploaded']} uploaded, {pub_stats['pruned']} pruned."
        )
    except RuntimeError as e:
        # If publishing isn't configured (e.g., during dev), warn but don't fail.
        click.echo(f"\n⚠ Skipped publish: {e}", err=True)
    except Exception as e:
        click.echo(f"\n❌ Publish failed: {type(e).__name__}: {e}", err=True)
        # Don't sys.exit — the local edition is still valid.


@main.command(name="curate")
@click.option(
    "--candidates-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Edition date (YYYY-MM-DD) of the candidates JSON to curate.",
)
@click.option(
    "--sections-config",
    "sections_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/sections.yaml",
    show_default=True,
)
def curate_cmd(candidates_date, sections_path: Path) -> None:
    """Run the AI curator on a previously-saved candidates JSON file."""
    try:
        config = load_curator_config(sections_path)
    except ValueError as e:
        click.echo(f"❌ Failed to load sections config: {e}", err=True)
        sys.exit(1)

    try:
        candidates = load_candidates(candidates_date)
    except FileNotFoundError as e:
        click.echo(f"❌ {e}", err=True)
        click.echo("    Run `frankenbote pipeline` first to generate candidates.", err=True)
        sys.exit(1)

    click.echo(f"Curating {len(candidates)} article(s) using {config.model}…")
    click.echo("(One API call. This may take 30–90 seconds.)\n")

    try:
        curated = curate(candidates, config)
    except RuntimeError as e:
        click.echo(f"❌ Curator failed: {e}", err=True)
        sys.exit(1)

    # Stats
    kept = [c for c in curated if c.section is not None]
    dropped = len(curated) - len(kept)
    by_priority: dict[str, int] = {}
    by_section: dict[str, int] = {}
    for c in kept:
        by_priority[c.priority.value] = by_priority.get(c.priority.value, 0) + 1
        by_section[c.section or "?"] = by_section.get(c.section or "?", 0) + 1

    click.echo(f"Result: {len(kept)} kept, {dropped} dropped\n")
    click.echo("By priority:")
    for p in ("P1", "P2", "P3", "P4"):
        click.echo(f"  {p}: {by_priority.get(p, 0)}")
    click.echo("\nBy section:")
    for sec_id in sorted(by_section):
        click.echo(f"  {sec_id}: {by_section[sec_id]}")

    out_path = save_curated_raw(curated, candidates_date)
    click.echo(f"\n  → Saved to {out_path}")


@main.command(name="select")
@click.option(
    "--curated-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Edition date (YYYY-MM-DD) of the curated-raw JSON.",
)
@click.option(
    "--sections-config",
    "sections_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/sections.yaml",
    show_default=True,
)
@click.option(
    "--sources-config",
    "sources_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/sources.yaml",
    show_default=True,
)
@click.option(
    "--size",
    type=click.IntRange(min=5, max=100),
    default=25,
    show_default=True,
    help="Target number of articles in the final edition.",
)
def select_cmd(
    curated_date,
    sections_path: Path,
    sources_path: Path,
    size: int,
) -> None:
    """Run the selector on a previously-saved curated-raw JSON file."""
    try:
        config = load_curator_config(sections_path)
        source_ids = [s.id for s in load_sources(sources_path)]
        targets = load_selector_targets(sections_path)
    except ValueError as e:
        click.echo(f"❌ Failed to load config: {e}", err=True)
        sys.exit(1)

    try:
        curated = load_curated_raw(curated_date)
    except FileNotFoundError as e:
        click.echo(f"❌ {e}", err=True)
        click.echo("    Run `frankenbote curate ...` first.", err=True)
        sys.exit(1)

    options = SelectorOptions(edition_size=size, targets=targets)
    edition = select(
        curated=curated,
        config=config,
        source_ids_in_order=source_ids,
        options=options,
        edition_date=curated_date,
    )

    s = edition.stats
    effective = options.effective_targets
    target_str = " ".join(f"{p}={effective[Priority(p)]:.0%}" for p in ("P1", "P2", "P3", "P4"))
    click.echo(f"Selection from {s.candidates_in} candidates ({s.curated_kept} eligible after curation):")
    click.echo(f"  → {s.selected} articles in final edition\n")
    click.echo(f"By priority (target: {target_str}):")
    for p in ("P1", "P2", "P3", "P4"):
        count = s.by_priority.get(p, 0)
        pct = (count / s.selected * 100) if s.selected else 0.0
        click.echo(f"  {p}: {count:3d}  ({pct:.0f}%)")
    click.echo("\nBy section:")
    for sec in edition.sections:
        click.echo(f"  {sec.display_name}: {len(sec.articles)}")

    out_path = save_edition(edition, curated_date)
    click.echo(f"\n  → Saved to {out_path}")


@main.command(name="render")
def render_cmd() -> None:
    """Render all kept editions to HTML in output/."""
    stats = render_all()
    click.echo(f"Rendered {stats['editions_rendered']} edition(s).")
    if stats["editions_pruned"]:
        click.echo(f"Pruned {stats['editions_pruned']} old HTML file(s).")
    click.echo(f"Copied {stats['assets_copied']} asset file(s).")
    click.echo(f"\nOpen output/index.html in a browser to view.")


@main.command(name="summarize")
@click.option(
    "--edition-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Edition date (YYYY-MM-DD) of the edition JSON to summarize.",
)
@click.option(
    "--sections-config",
    "sections_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/sections.yaml",
    show_default=True,
)
def summarize_cmd(edition_date, sections_path: Path) -> None:
    """Run the AI summarizer on a previously-saved edition JSON file."""
    try:
        summarizer_cfg = load_summarizer_config(sections_path)
    except ValueError as e:
        click.echo(f"❌ Failed to load sections config: {e}", err=True)
        sys.exit(1)

    try:
        edition = load_edition(edition_date)
    except FileNotFoundError as e:
        click.echo(f"❌ {e}", err=True)
        click.echo("    Run `frankenbote pipeline` first.", err=True)
        sys.exit(1)

    try:
        edition = summarize_edition(edition, model=summarizer_cfg.model)
    except RuntimeError as e:
        click.echo(f"❌ Summarizer failed: {e}", err=True)
        sys.exit(1)

    # Stats
    total = sum(len(s.articles) for s in edition.sections)
    with_summary = sum(
        1 for s in edition.sections for a in s.articles if a.ai_summary
    )
    click.echo(f"\nSummarized: {with_summary}/{total} articles "
               f"({total - with_summary} returned null)")

    out_path = save_edition(edition, edition_date)
    click.echo(f"  → Updated {out_path}")


@main.command(name="wrap-up")
@click.option(
    "--edition-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Edition date (YYYY-MM-DD) of the edition JSON to generate wrap-ups for.",
)
@click.option(
    "--sections-config",
    "sections_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/sections.yaml",
    show_default=True,
)
def wrap_up_cmd(edition_date, sections_path: Path) -> None:
    """Generate longer wrap-ups for lead articles in a saved edition JSON.

    Runs only the wrap-up LLM call — useful for iterating on that step
    without re-running the summarizer or the rest of the pipeline.
    """
    try:
        summarizer_cfg = load_summarizer_config(sections_path)
    except ValueError as e:
        click.echo(f"❌ Failed to load sections config: {e}", err=True)
        sys.exit(1)

    try:
        edition = load_edition(edition_date)
    except FileNotFoundError as e:
        click.echo(f"❌ {e}", err=True)
        click.echo("    Run `frankenbote pipeline` first.", err=True)
        sys.exit(1)

    try:
        edition = generate_wrap_ups(
            edition, model=summarizer_cfg.wrap_up_model or summarizer_cfg.model
        )
    except RuntimeError as e:
        click.echo(f"❌ Wrap-up generation failed: {e}", err=True)
        sys.exit(1)

    with_wrap_up = sum(
        1 for s in edition.sections for a in s.articles if a.wrap_up
    )
    click.echo(f"\nWrap-ups written: {with_wrap_up}")

    out_path = save_edition(edition, edition_date)
    click.echo(f"  → Updated {out_path}")


@main.command(name="publish")
def publish_cmd() -> None:
    """Upload the local output/ tree to netcup via SFTP."""
    try:
        config = load_publisher_config_from_env()
    except RuntimeError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)

    click.echo(
        f"Publishing to {config.username}@{config.host}:{config.remote_dir}…"
    )
    try:
        stats = publish(config)
    except Exception as e:
        click.echo(f"❌ Publish failed: {type(e).__name__}: {e}", err=True)
        sys.exit(1)

    click.echo(
        f"  → Uploaded: {stats['uploaded']}, pruned: {stats['pruned']}"
    )
    

if __name__ == "__main__":
    main()