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

`publisher.py`, `cli.py`, `__main__.py`, and `generate_wrap_ups()` are
excluded from coverage — they require a live SFTP server or fetch article
bodies over the network. `curate()` and `summarize_edition()` are tested
by injecting a scripted `LLMClient`; the Anthropic client is tested with a
mocked SDK.

There is no linter/formatter configured in `pyproject.toml` — don't assume
`ruff`/`black`/`mypy` are wired in without checking first. Editors may still
surface ruff findings via IDE integration even though it isn't part of the
project's own toolchain. When you touch a file for a task, it's fine to fix
ruff issues in that file as part of the change; don't do a repo-wide ruff
cleanup unless asked.

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
                                    └── llm/ (LLMClient) ──────────────────┘
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

### LLM integration (llm/, curator.py, summarizer.py)

All provider calls go through `llm/`. `llm/base.py` is provider-agnostic:
it defines the request/result shapes (`ToolCallRequest`, `ToolCallResult`),
the error hierarchy (`LLMError` → `LLMTransientError` → `LLMBatchTimeout`,
all `RuntimeError` subclasses so the CLI's existing handlers report them),
and the abstract `LLMClient` with four primitives (`call_tool`,
`submit_batch`, `wait_for_batch`, `batch_results`) plus the **retry loops**
built on them (`call_tool_with_retry`, `run_batch_with_retry`).
`llm/anthropic_client.py` is the only module that imports the Anthropic
SDK: `AnthropicLLMClient` reads `ANTHROPIC_API_KEY` on instantiation,
implements the primitives, and translates SDK exceptions into the
hierarchy above. To add a provider, subclass `LLMClient` in a new module
under `llm/` and re-export it from `llm/__init__.py`; the retry loops are
inherited, and pipeline code doesn't change.

`curator.py` and `summarizer.py` only build prompts and tool schemas, wrap
them in a `ToolCallRequest`, and hand them to the client together with a
`parse` callable (their Pydantic validation). Every call is a **forced tool
call** (`tool_choice`), so the API guarantees schema-valid output. The
`use_batch` flag (`--batch-off` on the CLI) selects the **Batches API**
(default) or synchronous streaming; both paths share the retry policy: two
attempts, no backoff, retry on transient errors (network, timeout, batch
timeout), a non-`tool_use` stop reason, a missing tool block, or a
`ValidationError` from `parse`; on the final failure the loop calls
`_debug.save_failure()` to dump raw output to `data/debug/` and raises
`RuntimeError`. Attempts and backoff are constructor arguments of the
client (`max_attempts`, `backoff_seconds`) with no config/env keys — only
tests change them. The public entry points accept an optional
`client: LLMClient` for injecting a fake in tests (see
`tests/conftest.py::ScriptedLLMClient`).

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
