"""Command-line interface for Frankenbote."""

import os
import sys

import click


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
    click.echo(f"  API key present:  {'yes' if has_api_key else 'no (ok for now)'}")


if __name__ == "__main__":
    main()