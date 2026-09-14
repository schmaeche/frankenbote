"""TaskSpec — everything that defines one AI step, independent of provider.

A task is a forced tool call: a system prompt, a tool the model must invoke,
and a Pydantic model describing the tool's input. The JSON schema sent to
the provider is *derived* from that model, so the schema and the validation
have a single source of truth.

Concrete tasks live in `llm/tasks/`; the pipeline only supplies the user
prompt (article data) and maps the parsed result back onto articles.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import click
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class TaskSpec(Generic[T]):
    """Provider-neutral definition of one AI step.

    name:             task id — the key in config.yaml `llm.models` and the
                      component name of debug dumps ("curator", …).
    label:            human label used in error messages ("Curator").
    system_prompt:    the instructions; user prompts are built by the caller.
    tool_name /
    tool_description: the forced tool the model must call.
    response_model:   Pydantic model of the tool input; its JSON schema is
                      the tool's input_schema, its validation is `parse()`.
    max_tokens:       output budget as a function of the number of items in
                      the user prompt.
    normalize:        optional pre-validation fix-up of the raw tool input.
    """

    name: str
    label: str
    system_prompt: str
    tool_name: str
    tool_description: str
    response_model: type[T]
    max_tokens: Callable[[int], int]
    normalize: Callable[[dict], dict] | None = None

    @property
    def tool(self) -> dict[str, Any]:
        """The tool definition in the provider-neutral JSON-schema form."""
        return {
            "name": self.tool_name,
            "description": self.tool_description,
            "input_schema": tool_schema(self.response_model),
        }

    def max_tokens_for(self, n_items: int) -> int:
        return self.max_tokens(n_items)

    def parse(self, tool_input: dict) -> T:
        """Validate raw tool input; raises pydantic.ValidationError on bad output."""
        if self.normalize is not None:
            tool_input = self.normalize(tool_input)
        return self.response_model.model_validate(tool_input)


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


def normalize_array_field(key: str) -> Callable[[dict], dict]:
    """Build a normalizer that repairs `key` when it arrives as a JSON string.

    The tool-use API is supposed to deliver typed arguments, but the model
    occasionally serializes a nested array as a string. The returned
    function detects that case and parses it back into a real list. Pure
    data fix-up — no semantic change. Raises ValueError when the string is
    not a JSON list.
    """

    def normalize(tool_input: dict) -> dict:
        value = tool_input.get(key)
        if isinstance(value, str):
            click.echo(f"WARN: Detected {key} as JSON string, parsing it…")
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"{key} was a string but not valid JSON: {e}") from e
            if not isinstance(parsed, list):
                raise ValueError(
                    f"{key} was a string but its JSON content is {type(parsed).__name__}"
                )
            tool_input = {**tool_input, key: parsed}
        return tool_input

    return normalize
