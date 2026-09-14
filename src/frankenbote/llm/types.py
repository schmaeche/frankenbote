"""Provider-neutral wire shapes for one forced tool call.

These live apart from `llm/base.py` because both sides of the seam need
them: the client builds requests and returns results, and a task in
`llm/tasks/` reads those results back (see `PerItemTask.interpret`).
Neither this module nor anything importing it knows about a provider.

`llm/base.py` re-exports all three, so `from frankenbote.llm import
ToolCallResult` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


class ToolCallParams(TypedDict):
    """Everything a provider needs to run one forced tool call, except the
    model — that is resolved by the client from the request's task."""

    system: str
    user_prompt: str
    tool: dict[str, Any]  # JSON-schema tool: name / description / input_schema
    max_tokens: int


class ToolCallRequest(TypedDict):
    """One tool call for a task, addressable by `custom_id` when batched."""

    task: str  # Task.name — selects the model via ModelConfig
    custom_id: str
    params: ToolCallParams


@dataclass(frozen=True)
class ToolCallResult:
    """Provider-neutral outcome of one tool call.

    stop_reason is "tool_use" on success. Otherwise it is an anomaly label:
    the provider's own stop reason ("max_tokens", "refusal", …), a batch
    item status ("errored", "expired", "canceled"), "no_tool_use_block",
    or "no_result" when a batch did not return the item at all.

    raw is the provider's response object, kept only for debug dumps.
    """

    custom_id: str
    tool_input: dict | None
    stop_reason: str
    raw: Any = None
