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

# Lint and type-check (both configured in pyproject.toml, both in [dev])
ruff check .                   # lint
ruff check . --fix             # apply the safe fixes
pyright                        # type-check src/ (see below for why not tests/)
```

`publisher.py`, `cli.py`, `__main__.py`, and `generate_wrap_ups()` are
excluded from coverage — they require a live SFTP server or fetch article
bodies over the network. `curate()` and `summarize_edition()` are tested
by injecting a scripted `LLMClient`; the Anthropic and OpenAI clients are
tested with mocked SDKs.

**Ruff** is configured in `pyproject.toml` (`[tool.ruff.lint]`) and shipped
in the `dev` extra. The selection is ruff's defaults (`E4`/`E7`/`E9`/`F`)
plus `UP` (pyupgrade) and `TRY` (exception handling). No formatter is
configured — `ruff format` and `black` are *not* wired in, so don't
reformat files.

`TRY003` is deliberately ignored: config loaders raise `ValueError` with a
message meant to be shown to the CLI user verbatim (see "Config" below),
and TRY003 flags every one of them. That convention is the design, not an
oversight. Pin rule changes in `pyproject.toml` rather than in editor
settings, so a local run and an editor squiggle report the same thing.

`ruff check .` is clean except for four findings left standing on purpose:

- `UP042` on `Category` and `Priority` (`str, Enum` → `enum.StrEnum`) —
  changes `str()` output and can alter pydantic serialization, so it is not
  a mechanical fix. Leave it unless you are ready to check the rendered
  edition and the stored JSON.
- `TRY300` in `paywall_gate.py` and `E741` (`l`) in `tests/test_fetcher.py`
  — cosmetic; fix them if you are editing those lines anyway.

**Pyright** (the engine behind VS Code's Pylance) is configured too, at
`typeCheckingMode = "standard"` to match Pylance's editor default, but
scoped to `include = ["src"]` and expected to stay at **zero errors** —
if `pyright` reports something, you introduced it.

`tests/` is deliberately excluded: the suite builds pydantic models from
literals (`"local"` for a `Category`, a `str` for `HttpUrl`, a dict for
`Window`). Pydantic coerces those at runtime, but pyright types `__init__`
from the field annotations and reported ~190 false positives. Don't
"fix" the tests to satisfy it, and don't widen `include` without reading
that pile first. Note both `include = ["src"]` *and* `ignore = ["tests"]`
are needed: `include` scopes a command-line run, but the language server
analyses whatever file is open in the editor regardless, so without
`ignore` a test file open in VS Code still lights up.

Editor setup: both extensions read `pyproject.toml` themselves, so there
is nothing to duplicate in VS Code settings — and rule lists must *not* be
duplicated there, since `ruff.lint.select` in editor settings overrides
pyproject and reintroduces the drift this config exists to prevent. Worth
setting: `ruff.importStrategy: "fromEnvironment"` (use the pinned ruff, not
the extension's bundled copy) and `python.analysis.diagnosticMode:
"workspace"` (match the CLI instead of checking only open files).
`.vscode/` is gitignored, so those stay per-machine.

Two things to know when writing type hints here: the project requires
Python 3.14, so the short generic forms are fine (`Generator[None]` instead
of `Generator[None, None, None]`) and PEP 695 syntax is preferred
(`class Task[TIn, TOut, TResp: BaseModel](ABC)` — keep the bound, or you
lose type information and end up reaching for `cast`); and a function
decorated with `@contextmanager` must be annotated `-> Generator[T]`, not
`-> Iterator[T]` — Pylance flags the latter as deprecated (see
`llm/anthropic_client.py`).

Two suppressions exist and should not be removed casually: the
`# noqa: TRY004` in `llm/task.py` (the malformed-tool-input guard raises
`ValueError` to match its sibling branch, not `TypeError` — it validates
untrusted model output, not a caller's argument type) and the
`ExtractSettings` TypedDict in `body_fetcher.py` (a plain dict collapses to
`dict[str, bool]` and every `**EXTRACT_SETTINGS` unpack looks like a bool
handed to `url`).

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
- `llm/openai_client.py` — the same for the OpenAI SDK, against the
  Responses API (sync: raw event stream; batch: JSONL upload to
  `/v1/responses`, output + error files). Two things live in code, not
  config: the tool is sent with `strict: true`, and `reasoning.effort` is
  set per task from the `_REASONING_EFFORT` mapping in `llm/factory.py`
  (passed to every client as `reasoning_effort`; the Anthropic client
  ignores it; an unlisted task gets `"none"`; levels are not validated).
  `max_output_tokens` counts reasoning tokens and the tasks'
  `max_tokens_for()` budgets have no room for them, so the client adds
  the level's headroom from `REASONING_HEADROOM` (`"none"` and unknown
  levels add 0). Strict mode requires
  every property to be `required` and no `default`s, so a response model
  with an optional field breaks OpenAI; `tests/test_llm_openai.py` checks
  every task's derived schema. Responses are translated onto the existing
  `stop_reason` labels (`max_output_tokens` → `"max_tokens"`, refusal →
  `"refusal"`), so the retry loop needs no provider knowledge.
- `llm/factory.py` — `create_client(config, use_batch=...)`, the only place
  a provider is chosen, plus `api_key_env(provider)` (used by `frankenbote
  hello`). One provider per run. `cli.py` builds one client per command and
  passes it down; every LLM-calling function takes `client: LLMClient` as a
  required argument. To add a provider: subclass `LLMClient` (with an
  `API_KEY_ENV` constant), add it to `_CLIENTS` in the factory and to the
  `provider` literal in `llm/config.py`.

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
