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

The same applies to **Pylance** (VS Code's Pyright-based type checker):
there is no `pyrightconfig.json` or `[tool.pyright]` section, so its
findings come from the editor's defaults, not the project. Treat them like
ruff findings — resolve them in files you touch, don't sweep the repo. Two
things to know when writing type hints here: the project requires Python
3.14, so the short generic forms are fine (`Generator[None]` instead of
`Generator[None, None, None]`); and a function decorated with
`@contextmanager` must be annotated `-> Generator[T]`, not `-> Iterator[T]`
— Pylance flags the latter as deprecated (see `llm/anthropic_client.py`).

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

Everything LLM-related lives in `llm/`; the pipeline files never name a
provider, a model, or a prompt:

- `llm/base.py` — provider-agnostic core. `LLMClient` is the abstract base
  with four primitives (`call_tool`, `submit_batch`, `wait_for_batch`,
  `batch_results`), the **model selection** (`resolve_model(task)` reads
  the `ModelConfig` it was constructed with), the task API the pipeline
  calls (`run_task`, `run_task_batch`, `build_request`) and the **retry
  loops**. Also the request/result shapes (`ToolCallRequest` carries a
  `task` name, not a model) and the error hierarchy (`LLMError` →
  `LLMTransientError` → `LLMBatchTimeout`, all `RuntimeError` subclasses
  so the CLI's existing handlers report them).
- `llm/task.py` — `TaskSpec`: one AI step, provider-neutral (name, label,
  system prompt, forced tool name/description, Pydantic response model,
  output-token budget, optional normalizer). The tool's JSON schema is
  **derived from the response model** via `tool_schema()` — there is no
  hand-written schema; add a field to the model and the schema follows.
  Response models set `extra="forbid"` so the schema carries
  `additionalProperties: false`.
- `llm/tasks/` — the concrete specs: `curate.py` (`curator_task(section_ids)`,
  a factory because the section enum comes from config), `summarize.py`
  (`SUMMARIZER_TASK`), `wrap_up.py` (`WRAP_UP_TASK`). To add a step: new
  module, export it from `tasks/__init__.py`, add its name to `TASK_NAMES`
  and a model field in `llm/config.py`, and a key in `config/config.yaml`.
- `llm/config.py` — `config/config.yaml` loader: provider, `use_batch`
  default, one model per task (`ModelConfig.for_task()`, with the
  `wrap_up` → `summarizer` fallback).
- `llm/anthropic_client.py` — the only module importing the Anthropic SDK.
  Implements the primitives, maps task → model when building the API
  request, translates SDK exceptions into the hierarchy above.
- `llm/factory.py` — `create_client(config, use_batch=...)`, the only place
  a provider is chosen. `cli.py` builds one client per command and passes
  it down; every LLM-calling function takes `client: LLMClient` as a
  required argument. To add a provider (#44): subclass `LLMClient`, add it
  to the factory and to the `provider` literal in `llm/config.py`.

`curator.py` and `summarizer.py` only build the *user* prompts (article
data with prompt-injection framing), call `client.run_task(SPEC, prompt,
n_items=...)`, and map the parsed response back onto articles. Every call
is a **forced tool call** (`tool_choice`), so the API guarantees
schema-valid output. Batch vs. synchronous is a client-level setting
(`use_batch` from config.yaml, `--batch-off` overrides per run); the
per-article wrap-up path always runs synchronously. Retry policy, shared by
every call: two attempts, no backoff, retry on transient errors (network,
timeout, batch timeout), a non-`tool_use` stop reason, a missing tool
block, or a `ValidationError` from the spec's `parse`; on the final
failure the loop calls `_debug.save_failure()` to dump raw output to
`data/debug/` and raises `RuntimeError`. Attempts and backoff are
constructor arguments of the client with no config keys — only tests
change them. Tests inject `tests/conftest.py::ScriptedLLMClient`.

**Prompt language convention**: prompts are written in English (there's an
open ticket to make output language configurable, and English source
prompts make that easier), but every prompt has an explicit clause forcing
German output regardless of prompt language (e.g. the `LANGUAGE` clause
in `llm/tasks/wrap_up.py`, or "Auf Deutsch, klare Sprache..." in
`llm/tasks/summarize.py`). When adding a new LLM
prompt: write the instructions in English, add an explicit German-output
clause, and do not retrofit this onto existing prompts unless asked.

Article titles/summaries are external, untrusted RSS content — both
prompts explicitly tell the model to treat `<article>`-tagged content as
data, not instructions, and every prompt says so in a "CRITICAL SAFETY
RULES" / similar block. Preserve this framing in any new prompt that embeds
feed content.

### Config

Four YAML files under `config/` drive behavior (see README for full field
docs): `sources.yaml` (feeds), `filter.yaml` (time window + keywords),
`sections.yaml` (curator sections/priorities/guidance, selector targets) and
`config.yaml` (LLM provider, batch default, one model per AI step). Each
loader (`config.py::load_sources`, `curator.py::load_curator_config`,
`selector.py::load_selector_targets`, `llm/config.py::load_llm_config`)
validates via Pydantic and raises `ValueError` with a message meant to be
shown directly to the CLI user — keep that pattern when adding new config.
Model names used to live in `sections.yaml`; `load_curator_config` rejects
the stale `curator.model` / `summarizer:` keys with a pointer to
`config.yaml` rather than silently ignoring them.

### Deployment specifics

Production runs as a one-shot Docker command via cron/systemd timer, not a
long-running service (see README's "Production deployment" section for the
full rationale). Notable non-obvious point: the container runs as UID 1000
(`frankenbote`), so any host-mounted `data/`, `output/`, `logs/`, or SSH key
must be `chown`'d to `1000:1000` — mismatched ownership is called out in the
README as the most common deployment error.
