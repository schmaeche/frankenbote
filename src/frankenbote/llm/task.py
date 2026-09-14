"""Task — one AI step: everything the model sees, and how its answer is read.

A task is a forced tool call. It owns, for its step:

  - the system prompt and the tool the model must invoke,
  - the Pydantic model describing the tool's input — the JSON schema sent
    to the provider is *derived* from it, so schema and validation have a
    single source of truth,
  - the rendering of its inputs into the user prompt,
  - the interpretation of the validated response into outputs *aligned
    one-to-one with those inputs*.

Two shapes exist, because two shapes of AI step exist:

  - `SingleCallTask` — one API call for a list of inputs, which the model
    addresses by index (curate, summarize);
  - `PerItemTask` — one API call per input, addressed by custom id when
    the calls are batched (wrap_up). The custom-id scheme is built and
    parsed here, in one place.

Concrete tasks live in `llm/tasks/`. The pipeline never sees a prompt, a
tool name or an index: it hands domain objects to
`LLMClient.run_task(task, inputs)` and gets a `TaskOutcome` back.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import click
from pydantic import BaseModel, ValidationError

from frankenbote.llm.types import ToolCallResult

TIn = TypeVar("TIn")  # one task input (an Article, a CuratedArticle, …)
TOut = TypeVar("TOut")  # one output per input
TResp = TypeVar("TResp", bound=BaseModel)  # the validated tool input


# -------- Results --------


@dataclass(frozen=True)
class ItemNote:
    """A per-item problem worth logging.

    `index` addresses the input it belongs to; None when the problem
    cannot be attributed to one (an unusable custom id, say). `reason` is
    a short phrase — the caller decides how to present it.
    """

    index: int | None
    reason: str


@dataclass(frozen=True)
class TaskOutcome(Generic[TOut]):
    """A task's results, aligned one-to-one with the inputs it was given.

    `values` always has exactly as many entries as there were inputs — a
    task fills in its own stand-in for an item the model did not answer.
    `notes` carries the problems behind those stand-ins; tasks never
    print, the pipeline logs.
    """

    values: list[TOut]
    notes: list[ItemNote] = field(default_factory=list)


# -------- The task contract --------


class Task(ABC, Generic[TIn, TOut, TResp]):
    """Provider-neutral definition of one AI step.

    Subclasses set the six attributes below (as class attributes, or in
    `__init__` when they depend on configuration) and implement
    `max_tokens_for` plus the rendering/interpretation methods of either
    `SingleCallTask` or `PerItemTask`.

    name:             task id — the key in config.yaml `llm.models` and
                      the component name of debug dumps ("curator", …).
    label:            human label used in error messages ("Curator").
    system_prompt:    the instructions.
    tool_name /
    tool_description: the forced tool the model must call.
    response_model:   Pydantic model of the tool input; its JSON schema is
                      the tool's input_schema, its validation is `parse()`.
    """

    name: str
    label: str
    system_prompt: str
    tool_name: str
    tool_description: str
    response_model: type[TResp]

    @property
    def tool(self) -> dict[str, Any]:
        """The tool definition in the provider-neutral JSON-schema form."""
        return {
            "name": self.tool_name,
            "description": self.tool_description,
            "input_schema": tool_schema(self.response_model),
        }

    @abstractmethod
    def max_tokens_for(self, n_items: int) -> int:
        """Output-token budget for a call covering `n_items` inputs."""

    def normalize(self, tool_input: dict) -> dict:
        """Pre-validation fix-up of the raw tool input. Identity by default."""
        return tool_input

    def parse(self, tool_input: dict) -> TResp:
        """Validate raw tool input; raises pydantic.ValidationError on bad output."""
        return self.response_model.model_validate(self.normalize(tool_input))


class SingleCallTask(Task[TIn, TOut, TResp]):
    """One API call for a list of inputs; the model addresses them by index."""

    @abstractmethod
    def render(self, inputs: Sequence[TIn]) -> str:
        """The user prompt for the whole list of inputs."""

    @abstractmethod
    def interpret(
        self, response: TResp, inputs: Sequence[TIn]
    ) -> TaskOutcome[TOut]:
        """Turn the validated response into one output per input, in order."""


class PerItemTask(Task[TIn, TOut, TResp]):
    """One API call per input, addressed by custom id when batched.

    Subclasses render and read a single item; the addressing scheme and
    the mapping of a batch's results back onto the inputs are implemented
    here, so construction and parsing can never drift apart.
    """

    @abstractmethod
    def render(self, item: TIn) -> str:
        """The user prompt for a single input."""

    @abstractmethod
    def read(self, response: TResp) -> TOut:
        """The output carried by one validated response."""

    @abstractmethod
    def missing(self) -> TOut:
        """The stand-in output for an input whose call did not succeed."""

    # ---- addressing ----

    def custom_id(self, index: int) -> str:
        """The batch custom id for the input at `index`."""
        return f"{self.name}-{index}"

    def parse_custom_id(self, custom_id: str) -> int | None:
        """The input index a custom id addresses, or None if unusable."""
        prefix = f"{self.name}-"
        if not custom_id.startswith(prefix):
            return None
        try:
            return int(custom_id[len(prefix) :])
        except ValueError:
            return None

    def items(self, inputs: Sequence[TIn]) -> list[tuple[str, str]]:
        """(custom_id, user_prompt) pairs, one per input, in order."""
        return [(self.custom_id(i), self.render(item)) for i, item in enumerate(inputs)]

    # ---- interpretation ----

    def interpret(
        self, results: Sequence[ToolCallResult], n_inputs: int
    ) -> TaskOutcome[TOut]:
        """Map per-item results back onto the inputs by custom id.

        Every input gets a value: the read output on success, `missing()`
        otherwise. Items that failed, arrived unaddressable or failed
        validation are reported as notes rather than raised — one bad item
        must not cost the whole run.
        """
        values = [self.missing() for _ in range(n_inputs)]
        notes: list[ItemNote] = []

        for result in results:
            index = self.parse_custom_id(result.custom_id)
            if index is None or not 0 <= index < n_inputs:
                notes.append(
                    ItemNote(None, f"unusable custom id {result.custom_id!r}")
                )
                continue
            if result.stop_reason != "tool_use":
                notes.append(ItemNote(index, result.stop_reason))
                continue
            try:
                values[index] = self.read(self.parse(result.tool_input or {}))
            except ValidationError as exc:
                notes.append(ItemNote(index, f"validation: {exc}"))

        return TaskOutcome(values, notes)


# -------- Schema derivation --------


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for a tool's input, derived from a Pydantic model.

    Pydantic's output is trimmed for the model's benefit: auto-generated
    `title` annotations and class-docstring `description`s on object nodes
    are dropped (they are noise tokens). Property-level descriptions from
    `Field(description=...)` are kept. Response models should set
    `extra="forbid"` so the schema carries `additionalProperties: false`.
    """
    return _strip(model.model_json_schema())


def _strip(node: Any, *, keyed: bool = False) -> Any:
    """Recursively remove schema annotations.

    `keyed` marks a mapping whose keys are user names (properties / $defs),
    not schema keywords — those keys are never stripped, only their values.
    """
    if isinstance(node, list):
        return [_strip(item) for item in node]
    if not isinstance(node, dict):
        return node
    if keyed:
        return {key: _strip(value) for key, value in node.items()}
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title":
            continue
        if key == "description" and "properties" in node:
            continue
        out[key] = _strip(value, keyed=key in ("properties", "$defs"))
    return out


# -------- Normalizers --------


def normalize_array_field(tool_input: dict, key: str) -> dict:
    """Repair `key` when it arrives as a JSON string instead of an array.

    The tool-use API is supposed to deliver typed arguments, but the model
    occasionally serializes a nested array as a string. Pure data fix-up —
    no semantic change. Raises ValueError when the string is not a JSON list.
    """
    value = tool_input.get(key)
    if not isinstance(value, str):
        return tool_input

    click.echo(f"WARN: Detected {key} as JSON string, parsing it…")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"{key} was a string but not valid JSON: {e}") from e
    if not isinstance(parsed, list):
        raise ValueError(
            f"{key} was a string but its JSON content is {type(parsed).__name__}"
        )
    return {**tool_input, key: parsed}
