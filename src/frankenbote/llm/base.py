"""Provider-agnostic LLM client interface.

This module knows nothing about any concrete provider: no SDK imports, no
credentials. It defines

  - the error hierarchy that providers translate their SDK exceptions into,
  - LLMClient, the abstract base with four provider primitives
    (call_tool, submit_batch, wait_for_batch, batch_results), the model
    selection for each task, and the whole-call retry loops built on top.

The request/result shapes providers speak (ToolCallRequest,
ToolCallResult) live in `llm/types.py` and are re-exported here.

Every pipeline call is a *forced tool call*: one system prompt, one user
prompt, one tool the model must invoke — described by a Task
(`llm/task.py`, concrete tasks in `llm/tasks/`).

There are two layers to the API:

  - `run_task(task, inputs)` is what the pipeline calls. It renders the
    inputs through the task, runs the call(s) — one for a SingleCallTask,
    one per input for a PerItemTask, batched or not — and returns a
    TaskOutcome whose values line up one-to-one with the inputs.
  - `run_prompt()` / `run_prompt_batch()` are the prompt-level layer it is
    built on: give them a task and ready-made user prompts and they pick
    the model from the client's ModelConfig, build the request, run it
    with retries and return the parsed response model.

Retry policy (identical for every caller):
  - `max_attempts` attempts (default 2, i.e. one retry), optional
    `backoff_seconds` between attempts (default 0, i.e. none).
  - A transient provider error (network, timeout, batch timeout) is
    retried; on the final attempt it is re-raised as RuntimeError.
  - A non-`tool_use` stop reason, a missing tool block, or a response that
    fails the task's `parse` step is retried; on the final attempt the raw
    output is dumped via `_debug.save_failure()` and RuntimeError is raised.
  - Anything else (a 4xx, a bug) propagates immediately.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import click
from pydantic import BaseModel, ValidationError

from frankenbote._debug import save_failure
from frankenbote.llm.config import ModelConfig
from frankenbote.llm.task import (
    ItemNote,
    PerItemTask,
    SingleCallTask,
    Task,
    TaskOutcome,
)
from frankenbote.llm.types import ToolCallParams, ToolCallRequest, ToolCallResult

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

__all__ = [
    "LLMBatchTimeout",
    "LLMClient",
    "LLMError",
    "LLMTransientError",
    "ToolCallParams",
    "ToolCallRequest",
    "ToolCallResult",
]


# -------- Errors --------


class LLMError(RuntimeError):
    """A provider failure surfaced by an LLMClient (subclass of RuntimeError
    so the CLI's existing `except RuntimeError` handlers report it)."""


class LLMTransientError(LLMError):
    """A failure worth retrying: network error, request timeout, …"""


class LLMBatchTimeout(LLMTransientError):
    """A batch did not finish within the client's batch timeout."""


# -------- Abstract client --------


class LLMClient(ABC):
    """Abstract LLM client: provider primitives, model selection, retry loops.

    Subclasses implement the four primitives and translate their SDK's
    errors into LLMError / LLMTransientError. Everything else — which model
    a task uses, request building, batching, retries — is inherited.
    """

    def __init__(
        self,
        models: ModelConfig,
        *,
        use_batch: bool = True,
        max_attempts: int = 2,
        backoff_seconds: float = 0.0,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")
        self.models = models
        self.use_batch = use_batch
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    # ---- model selection ----

    def resolve_model(self, task: str) -> str:
        """The model id configured for a task name (see ModelConfig.for_task)."""
        return self.models.for_task(task)

    # ---- provider primitives ----

    @abstractmethod
    def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        """Run one synchronous tool call and return its result."""

    @abstractmethod
    def submit_batch(self, requests: Sequence[ToolCallRequest]) -> str:
        """Submit a batch of tool calls; return the provider's batch id."""

    @abstractmethod
    def wait_for_batch(self, batch_id: str) -> None:
        """Block until the batch has finished processing.

        Raises LLMBatchTimeout if it does not finish in time.
        """

    @abstractmethod
    def batch_results(self, batch_id: str) -> list[ToolCallResult]:
        """Fetch the results of a finished batch, one entry per item."""

    # ---- task API (what the pipeline calls) ----

    def run_task(
        self,
        task: Task[TIn, TOut, Any],
        inputs: Sequence[TIn],
        *,
        on_attempt: Callable[[int], None] | None = None,
    ) -> TaskOutcome[TOut]:
        """Run a task over `inputs` and return outputs aligned with them.

        The task renders the prompt(s) and interprets the response(s); this
        method only decides how many calls to make and how to route them:

          - SingleCallTask — one call covering every input;
          - PerItemTask — one call per input, submitted as a single batch
            when the client is in batch mode, otherwise run synchronously
            one after another. A per-item failure becomes a note and the
            task's `missing()` value; it never aborts the run.

        on_attempt is called with the attempt number before each attempt at
        the task as a whole, so the caller can print its own progress line.
        """
        if not inputs:
            return TaskOutcome([], [])

        if isinstance(task, SingleCallTask):
            response = self.run_prompt(
                task, task.render(inputs), n_items=len(inputs), on_attempt=on_attempt
            )
            outcome = task.interpret(response, inputs)
        elif isinstance(task, PerItemTask):
            outcome = (
                self._run_per_item_batched(task, inputs, on_attempt)
                if self.use_batch
                else self._run_per_item_sync(task, inputs, on_attempt)
            )
        else:
            raise TypeError(
                f"{type(task).__name__} is neither a SingleCallTask "
                "nor a PerItemTask"
            )

        if len(outcome.values) != len(inputs):
            raise RuntimeError(
                f"{task.label} returned {len(outcome.values)} values "
                f"for {len(inputs)} inputs"
            )
        return outcome

    def _run_per_item_batched(
        self,
        task: PerItemTask[TIn, TOut, Any],
        inputs: Sequence[TIn],
        on_attempt: Callable[[int], None] | None,
    ) -> TaskOutcome[TOut]:
        results = self.run_prompt_batch(task, task.items(inputs), on_attempt=on_attempt)
        return task.interpret(results, len(inputs))

    def _run_per_item_sync(
        self,
        task: PerItemTask[TIn, TOut, Any],
        inputs: Sequence[TIn],
        on_attempt: Callable[[int], None] | None,
    ) -> TaskOutcome[TOut]:
        """One synchronous call per input, each with its own retries.

        A single item that keeps failing is recorded and skipped — debug
        dumps are off here, as one thin article is not worth a dump.
        """
        if on_attempt is not None:
            on_attempt(1)
        values: list[TOut] = []
        notes: list[ItemNote] = []
        for index, item in enumerate(inputs):
            try:
                response = self.run_prompt(
                    task, task.render(item), use_batch=False, save_debug=False
                )
            except RuntimeError as exc:  # includes LLMError from the provider
                values.append(task.missing())
                notes.append(ItemNote(index, str(exc)))
                continue
            values.append(task.read(response))
        return TaskOutcome(values, notes)

    # ---- prompt API (the layer run_task is built on) ----

    def build_request(
        self,
        task: Task[Any, Any, Any],
        user_prompt: str,
        *,
        custom_id: str | None = None,
        n_items: int = 1,
    ) -> ToolCallRequest:
        """Turn a task plus a ready-made user prompt into a provider request."""
        return ToolCallRequest(
            task=task.name,
            custom_id=task.name if custom_id is None else custom_id,
            params=ToolCallParams(
                system=task.system_prompt,
                user_prompt=user_prompt,
                tool=task.tool,
                max_tokens=task.max_tokens_for(n_items),
            ),
        )

    def run_prompt(
        self,
        task: Task[Any, Any, M],
        user_prompt: str,
        *,
        n_items: int = 1,
        use_batch: bool | None = None,
        on_attempt: Callable[[int], None] | None = None,
        save_debug: bool = True,
    ) -> M:
        """Run one call of a task with retries and return the parsed response.

        use_batch=None uses the client's configured default.
        """
        request = self.build_request(task, user_prompt, n_items=n_items)
        return self.call_tool_with_retry(
            request,
            task.parse,
            component=task.name,
            label=task.label,
            use_batch=self.use_batch if use_batch is None else use_batch,
            on_attempt=on_attempt,
            save_debug=save_debug,
        )

    def run_prompt_batch(
        self,
        task: Task[Any, Any, Any],
        items: Sequence[tuple[str, str]],
        *,
        n_items: int = 1,
        on_attempt: Callable[[int], None] | None = None,
    ) -> list[ToolCallResult]:
        """Run many calls of one task as a single batch, retrying the whole
        batch on transient errors.

        items are (custom_id, user_prompt) pairs. Per-item failures are not
        retried; they come back as results for the task to interpret.
        """
        requests = [
            self.build_request(task, prompt, custom_id=custom_id, n_items=n_items)
            for custom_id, prompt in items
        ]
        return self.run_batch_with_retry(
            requests,
            component=task.name,
            label=f"{task.label} batch",
            on_attempt=on_attempt,
        )

    # ---- composed calls (provider-agnostic) ----

    def run_batch(self, requests: Sequence[ToolCallRequest]) -> list[ToolCallResult]:
        """submit → wait → results, with the progress line the CLI prints."""
        batch_id = self.submit_batch(requests)
        click.echo(f"\n  Batch {batch_id} submitted, polling", nl=False)
        self.wait_for_batch(batch_id)
        click.echo()
        return self.batch_results(batch_id)

    def call_tool_batched(self, request: ToolCallRequest) -> ToolCallResult:
        """Run a single tool call through the batch path."""
        for result in self.run_batch([request]):
            if result.custom_id == request["custom_id"]:
                return result
        return ToolCallResult(request["custom_id"], None, "no_result")

    # ---- retry loops ----

    def call_tool_with_retry(
        self,
        request: ToolCallRequest,
        parse: Callable[[dict], T],
        *,
        component: str,
        label: str | None = None,
        use_batch: bool = True,
        on_attempt: Callable[[int], None] | None = None,
        save_debug: bool = True,
    ) -> T:
        """Run one tool call, retrying on failure, and return `parse(tool_input)`.

        parse:       turns the raw tool input into the caller's validated
                     model; must raise pydantic.ValidationError on bad output.
        component:   name for debug dumps ("curator", …).
        label:       human label for messages; defaults to the capitalized
                     component.
        use_batch:   route the call through the batch API or the
                     synchronous API.
        on_attempt:  called with the attempt number before each attempt, so
                     the caller can print its own progress line.
        save_debug:  dump the raw output to data/debug on the final failure.
        """
        label = component.capitalize() if label is None else label
        times = self._times_word()
        call = self.call_tool_batched if use_batch else self.call_tool

        for attempt in range(1, self.max_attempts + 1):
            last = attempt == self.max_attempts
            if on_attempt is not None:
                on_attempt(attempt)

            try:
                result = call(request)
            except LLMTransientError as exc:
                click.echo(f"\n  Network error on attempt {attempt}: {exc}")
                if last:
                    raise RuntimeError(
                        f"{label} failed {times} due to network errors. "
                        f"attempt {attempt}: network error: {exc}"
                    ) from exc
                self._backoff(attempt)
                continue

            if result.stop_reason != "tool_use":
                last_error = f"attempt {attempt}: {_describe_stop(result.stop_reason)}"
                if last:
                    raise RuntimeError(
                        f"{label} failed {times}. {last_error}"
                        + self._debug_suffix(save_debug, component, attempt, last_error, result.raw)
                    )
                self._backoff(attempt)
                continue

            if result.tool_input is None:
                last_error = f"attempt {attempt}: no tool_use block in response"
                if last:
                    raise RuntimeError(
                        f"{label} failed {times}. {last_error}"
                        + self._debug_suffix(save_debug, component, attempt, last_error, result.raw)
                    )
                self._backoff(attempt)
                continue

            try:
                return parse(result.tool_input)
            except ValidationError as exc:
                last_error = f"attempt {attempt}: validation: {exc}"
                if last:
                    raise RuntimeError(
                        f"{label} tool output invalid after retry. {last_error}"
                        + self._debug_suffix(
                            save_debug, component, attempt, last_error, result.tool_input
                        )
                    ) from exc
                self._backoff(attempt)
                continue

        raise AssertionError("unreachable")  # pragma: no cover

    def run_batch_with_retry(
        self,
        requests: Sequence[ToolCallRequest],
        *,
        component: str,
        label: str | None = None,
        on_attempt: Callable[[int], None] | None = None,
    ) -> list[ToolCallResult]:
        """Run a multi-item batch, retrying the whole batch on transient errors.

        Per-item failures are *not* retried — they come back as results with
        a non-"tool_use" stop_reason for the caller to handle.
        """
        label = component.capitalize() if label is None else label
        times = self._times_word()

        for attempt in range(1, self.max_attempts + 1):
            last = attempt == self.max_attempts
            if on_attempt is not None:
                on_attempt(attempt)
            try:
                return self.run_batch(requests)
            except LLMTransientError as exc:
                click.echo(f"\n  Error on attempt {attempt}: {exc}", err=True)
                if last:
                    raise RuntimeError(
                        f"{label} failed {times}. Last error: {exc}"
                    ) from exc
                self._backoff(attempt)

        raise AssertionError("unreachable")  # pragma: no cover

    # ---- helpers ----

    def _times_word(self) -> str:
        return "twice" if self.max_attempts == 2 else f"{self.max_attempts} times"

    def _backoff(self, attempt: int) -> None:
        """Sleep before the next attempt: backoff_seconds * 2^(attempt-1)."""
        if self.backoff_seconds > 0:
            time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

    @staticmethod
    def _debug_suffix(
        save_debug: bool, component: str, attempt: int, error: str, raw: Any
    ) -> str:
        if not save_debug:
            return ""
        path = save_failure(component, attempt, error, raw)
        return f"\n  Debug context saved to {path}"


def _describe_stop(stop_reason: str) -> str:
    if stop_reason == "max_tokens":
        return "response truncated (max_tokens hit)"
    if stop_reason == "refusal":
        return "Claude refused on safety grounds"
    return f"unexpected stop_reason {stop_reason!r}"
