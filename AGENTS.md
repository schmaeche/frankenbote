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

Everything LLM-related lives in `llm/`. The split is: **a task owns what
the model sees and how its answer is read; the pipeline owns I/O and
domain objects.** `curator.py` and `summarizer.py` contain no prompt text,
no tool name, no `article_index` bookkeeping and no custom-id strings —
they prepare inputs, make one call, and apply the results.

- `llm/base.py` — provider-agnostic core. `LLMClient` is the abstract base
  with four primitives (`call_tool`, `submit_batch`, `wait_for_batch`,
  `batch_results`), the **model selection** (`resolve_model(task)` reads
  the `ModelConfig` it was constructed with) and the **retry loops**. Two
  layers of API sit on top: `run_task(task, inputs)` — the only thing the
  pipeline calls — renders the inputs through the task, runs the call(s)
  and returns a `TaskOutcome`; `run_prompt()` / `run_prompt_batch()` /
  `build_request()` are the prompt-level layer it is built on. Also the
  error hierarchy (`LLMError` → `LLMTransientError` → `LLMBatchTimeout`,
  all `RuntimeError` subclasses so the CLI's existing handlers report them).
- `llm/types.py` — the request/result shapes providers speak
  (`ToolCallRequest` carries a `task` name, not a model; `ToolCallResult`).
  Separate from `base.py` because tasks read results back too. `base.py`
  re-exports all three.
- `llm/task.py` — the task contract. `Task` is the provider-neutral base
  (name, label, system prompt, forced tool name/description, Pydantic
  response model, `max_tokens_for()`, optional `normalize()`). The tool's
  JSON schema is **derived from the response model** via `tool_schema()` —
  there is no hand-written schema; add a field to the model and the schema
  follows. Response models set `extra="forbid"` so the schema carries
  `additionalProperties: false`. Two subclasses for the two shapes of AI
  step:
  - `SingleCallTask` — one call for a list of inputs, addressed by index:
    `render(inputs) -> str`, `interpret(response, inputs) -> TaskOutcome`.
  - `PerItemTask` — one call per input: `render(item)`, `read(response)`,
    `missing()`. The batch custom-id scheme (`custom_id()` /
    `parse_custom_id()`) and the mapping of results back onto inputs are
    implemented once on the base, so construction and parsing cannot drift.

  Both return a `TaskOutcome`: `values` aligned one-to-one with the inputs
  (a task fills its own stand-in for an item the model skipped) plus
  `notes` — `ItemNote(index, reason)` per-item problems. **Tasks never
  print**; the pipeline logs the notes with article titles.
- `llm/tasks/` — the concrete tasks: `curate.py` (`curator_task(config)`, a
  factory because the section enum and the prompt both come from the
  loaded `CuratorConfig`), `summarize.py` (`SUMMARIZER_TASK`), `wrap_up.py`
  (`WRAP_UP_TASK`, taking `(article, body)` pairs). To add a step: new
  module, subclass `SingleCallTask` or `PerItemTask`, export it from
  `tasks/__init__.py`, add its name to `TASK_NAMES` and a model field in
  `llm/config.py`, and a key in `config/config.yaml`.
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

What is left in `curator.py` and `summarizer.py` is I/O and domain
mapping: loading `sections.yaml`, flattening the edition, fetching article
bodies and choosing between body and feed snippet (`_select_body`), then
writing the aligned outputs onto `CuratedArticle` / `Edition` and logging
whatever notes came back. Tasks return per-input results
(`CuratorDecision`, `str | None`), never `CuratedArticle` or `Edition`.

`CuratorConfig` lives in `models.py`, not `curator.py`, so
`llm/tasks/curate.py` can be built from it without importing the pipeline
stage; `curator.py` re-exports it.

Every call is a **forced tool call** (`tool_choice`), so the API guarantees
schema-valid output. Batch vs. synchronous is a client-level setting
(`use_batch` from config.yaml, `--batch-off` overrides per run) and is
handled inside `run_task`: a `PerItemTask` goes out as one batch or as one
synchronous call per input, rendered and read by the same task code either
way. Retry policy, shared by every call: two attempts, no backoff, retry
on transient errors (network, timeout, batch timeout), a non-`tool_use`
stop reason, a missing tool block, or a `ValidationError` from the task's
`parse`; on the final failure the loop calls `_debug.save_failure()` to
dump raw output to `data/debug/` and raises `RuntimeError`. The exception
is a `PerItemTask` running synchronously: a persistently failing *item*
becomes an `ItemNote` and the task's `missing()` value rather than
aborting the run, and writes no debug dump. Attempts and backoff are
constructor arguments of the client with no config keys — only tests
change them. Tests inject `tests/conftest.py::ScriptedLLMClient`.

**Prompt regression tests**: the rendered user prompts are pinned
byte-for-byte against fixtures in `tests/fixtures/prompts/`. A diff there
means the model sees something different — update the fixture
deliberately, never to make a test pass.

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
