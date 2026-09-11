# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, Gemini CLI,
Cursor, etc.) when working with code in this repository. It is the single
source of project context, independent of which tool reads it — `CLAUDE.md`
is a symlink to this file.

## What this project is

Frankenbote is a personal weekly news digest for Franconia, Bavaria, and
Germany: an AI pipeline that fetches RSS feeds, curates and summarizes
articles with Claude, and publishes a static HTML newsletter via SFTP. See
[README.md](README.md) for the full pipeline description, CLI reference, and
production deployment notes — this file focuses on things that aren't
already spelled out there.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run the CLI
frankenbote hello              # verify env/config
frankenbote pipeline           # full run: fetch → filter → curate → select → summarize → render → publish
frankenbote pipeline --no-curate   # stop after filtering, no LLM calls — fast dev loop

# Tests
pytest -q                                          # full suite
pytest tests/test_paywall.py -q                    # one file
pytest tests/test_paywall.py::test_name -q          # one test
ptw . -- -x --tb=short -q                           # watch mode
pytest --cov=frankenbote --cov-report=term-missing -q   # with coverage (fail_under = 70)

# Or via the dedicated test container (no local Python/venv needed)
docker compose run --rm test
```

`publisher.py`, `cli.py`, `__main__.py`, and the `curate()` /
`summarize_edition()` entry points are excluded from coverage — they require
a live SFTP server or live Anthropic API calls.

There is no linter/formatter configured in `pyproject.toml` — don't assume
`ruff`/`black`/`mypy` are wired in without checking first.

## Architecture

### Pipeline data flow

Each stage is a plain function/module, wired together in `cli.py`, that
reads/writes intermediate JSON via `storage.py` — every stage is independently
re-runnable against a previously saved date (`frankenbote curate
--candidates-date ...`, `frankenbote select --curated-date ...`, etc.) without
repeating earlier (expensive, LLM-billed) stages:

```
fetcher.py → filter.py → paywall_gate.py + curator.py → selector.py → summarizer.py → renderer.py → publisher.py
   (Article)   (Article)      (CuratedArticle)             (Edition)      (Edition)      (HTML)         (SFTP)
```

- `data/editions/YYYY-MM-DD-candidates.json` — post-filter articles
- `data/editions/YYYY-MM-DD-curated-raw.json` — curator output before selection
- `data/editions/YYYY-MM-DD.json` — the final `Edition`, the **canonical
  record**; HTML in `output/` is always regenerable from it (`frankenbote
  render`)

Pydantic models in `models.py` are the shape contract threaded through every
stage (`Article` → `CuratedArticle` → `Edition`); read that file before
touching any pipeline stage.

### Paywall detection (strategy pattern)

`paywall/` uses a three-state chain-of-responsibility: each `PaywallStrategy`
in `paywall/strategies/` returns a definitive `PaywallResult` or `None`
("no signal, try the next one"). `paywall/detector.py` holds the ordered
`_STRATEGIES` tuple — cheapest/most-authoritative first — that callers never
see directly. To add a new strategy: implement `PaywallStrategy` in its own
module under `strategies/`, export it from `strategies/__init__.py`, and
append an instance to `_STRATEGIES` in `detector.py`; call sites
(`paywall_gate.py`) don't change.

`paywall_gate.py` sits between the curator and the selector: it runs
`selector.select()`, fetches the HTML of the chosen articles, excludes any
definitively-paywalled ones, and **re-runs selection** so a dropped article
is backfilled by the next-best candidate rather than shrinking the edition.
Only a *definitive* paywalled verdict excludes an article — unknown/failed
fetches always keep it in.

### LLM integration (curator.py, summarizer.py)

Both use Anthropic **tool use** (forced `tool_choice`) instead of parsing
free-text JSON, so the API guarantees schema-valid output. Both support the
**Batches API** (default) or synchronous streaming (`--batch-off`), selected
via a `use_batch` flag that swaps `_call_llm_batch` / `_call_llm` but keeps
the surrounding retry logic identical. Both retry exactly once on failure
(network error, non-`tool_use` stop reason, or schema validation failure);
on a second failure they call `_debug.save_failure()` to dump raw output to
`data/debug/` before raising.

**Prompt language convention**: prompts are written in English (there's an
open ticket to make output language configurable, and English source
prompts make that easier), but every prompt has an explicit clause forcing
German output regardless of prompt language (e.g. `summarizer.py`'s
`LANGUAGE` clause, "Auf Deutsch, klare Sprache..."). When adding a new LLM
prompt: write the instructions in English, add an explicit German-output
clause, and do not retrofit this onto existing prompts unless asked.

Article titles/summaries are external, untrusted RSS content — both
prompts explicitly tell the model to treat `<article>`-tagged content as
data, not instructions, and every prompt says so in a "CRITICAL SAFETY
RULES" / similar block. Preserve this framing in any new prompt that embeds
feed content.

### Config

Three YAML files under `config/` drive behavior (see README for full field
docs): `sources.yaml` (feeds), `filter.yaml` (time window + keywords),
`sections.yaml` (curator sections/priorities/guidance *and* the
curator/summarizer model names, including optional `wrap_up_model`). Each
loader (`config.py::load_sources`, `curator.py::load_curator_config`,
`summarizer.py::load_summarizer_config`, `selector.py::load_selector_targets`)
validates via Pydantic and raises `ValueError` with a message meant to be
shown directly to the CLI user — keep that pattern when adding new config.

### Deployment specifics

Production runs as a one-shot Docker command via cron/systemd timer, not a
long-running service (see README's "Production deployment" section for the
full rationale). Notable non-obvious point: the container runs as UID 1000
(`frankenbote`), so any host-mounted `data/`, `output/`, `logs/`, or SSH key
must be `chown`'d to `1000:1000` — mismatched ownership is called out in the
README as the most common deployment error.
