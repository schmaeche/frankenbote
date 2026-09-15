"""OpenAI implementation of LLMClient.

The only module in the pipeline that imports the OpenAI SDK. It owns:

  - credentials: the API key comes from the `api_key` argument or the
    OPENAI_API_KEY environment variable, read on instantiation;
  - the synchronous streaming call and the Batch API calls (JSONL upload,
    create / retrieve / cancel, output and error files) with their polling
    constants — both against the Responses API (`/v1/responses`);
  - translation of Responses API output into ToolCallResult and of SDK
    exceptions into the provider-neutral LLMError hierarchy.

Two OpenAI specifics are fixed here rather than configured:

  - the tool is sent with `strict: true`, so the arguments are guaranteed
    to match the schema. The schemas derived from the response models
    satisfy strict mode (every property required, additionalProperties
    false) — `tests/test_llm_openai.py` checks every task;
  - reasoning effort is always "none". `max_output_tokens` counts
    reasoning tokens, and the task budgets are sized for the answer alone.

Model selection and retry behaviour are inherited unchanged from LLMClient.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from typing import Any

import click
import httpx
import openai

from frankenbote.llm.base import (
    LLMBatchTimeout,
    LLMClient,
    LLMError,
    LLMTransientError,
    ToolCallRequest,
    ToolCallResult,
)
from frankenbote.llm.config import ModelConfig

logger = logging.getLogger(__name__)

# Errors worth a retry. Everything else in openai.APIError (4xx, 5xx the SDK
# already retried internally, …) is surfaced as a plain LLMError.
_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    httpx.RemoteProtocolError,
)

# Stream events that carry the finished response.
_FINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")

# Batch statuses after which the batch no longer changes.
_BATCH_DONE = ("completed", "expired", "cancelled")

# Batch error codes for items that never ran, mapped onto the item labels
# ToolCallResult documents; any other per-item error is "errored".
_ITEM_ERROR_LABELS = {"batch_expired": "expired", "batch_cancelled": "canceled"}


@contextmanager
def _translate_errors() -> Generator[None]:
    """Map SDK exceptions onto the provider-neutral hierarchy."""
    try:
        yield
    except _TRANSIENT_ERRORS as exc:
        raise LLMTransientError(str(exc)) from exc
    except openai.APIError as exc:
        raise LLMError(str(exc)) from exc


class OpenAILLMClient(LLMClient):
    """LLMClient backed by the OpenAI Python SDK (Responses API)."""

    API_KEY_ENV = "OPENAI_API_KEY"
    BATCH_ENDPOINT = "/v1/responses"
    BATCH_POLL_INTERVAL = 30  # seconds between status checks
    BATCH_TIMEOUT = 3_600  # 60-minute hard limit
    PROGRESS_EVERY = 25  # streamed argument deltas per progress dot
    REASONING_EFFORT = "none"

    def __init__(
        self,
        models: ModelConfig,
        *,
        use_batch: bool = True,
        api_key: str | None = None,
        max_attempts: int = 2,
        backoff_seconds: float = 0.0,
        batch_poll_interval: float | None = None,
        batch_timeout: float | None = None,
        sdk_client: Any = None,
    ):
        """Create the client.

        sdk_client lets tests inject a stand-in for openai.OpenAI; when
        given, no API key is required.
        """
        super().__init__(
            models,
            use_batch=use_batch,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        if sdk_client is None:
            api_key = api_key or os.environ.get(self.API_KEY_ENV)
            if not api_key:
                raise RuntimeError(f"{self.API_KEY_ENV} is not set")
            sdk_client = openai.OpenAI(api_key=api_key)
        self._client = sdk_client
        self.batch_poll_interval = (
            self.BATCH_POLL_INTERVAL
            if batch_poll_interval is None
            else batch_poll_interval
        )
        self.batch_timeout = (
            self.BATCH_TIMEOUT if batch_timeout is None else batch_timeout
        )

    # ---- primitives ----

    def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        """Streaming Responses call with the tool forced via tool_choice.

        The raw event stream is consumed rather than the SDK's stream
        helper: the helper JSON-parses strict tool arguments itself and
        would raise on arguments truncated by max_output_tokens, which must
        come back as a "max_tokens" result instead.
        """
        response = None
        deltas_seen = 0
        with _translate_errors():
            stream = self._client.responses.create(
                **self._response_kwargs(request), stream=True
            )
            for event in stream:
                if event.type == "response.function_call_arguments.delta":
                    deltas_seen += 1
                    if deltas_seen % self.PROGRESS_EVERY == 0:
                        click.echo(".", nl=False)
                elif event.type in _FINAL_EVENTS:
                    response = event.response

        if response is None:
            raise LLMTransientError("response stream ended without a final response")
        return result_from_response(
            request["custom_id"], response.model_dump(mode="json"), response
        )

    def submit_batch(self, requests: Sequence[ToolCallRequest]) -> str:
        """Upload the requests as a JSONL file and start a batch on it."""
        jsonl = "".join(
            json.dumps(self._to_batch_line(r), ensure_ascii=False) + "\n"
            for r in requests
        )
        with _translate_errors():
            batch_file = self._client.files.create(
                file=("frankenbote-batch.jsonl", jsonl.encode("utf-8")),
                purpose="batch",
            )
            batch = self._client.batches.create(
                input_file_id=batch_file.id,
                endpoint=self.BATCH_ENDPOINT,
                completion_window="24h",
            )
        return batch.id

    def wait_for_batch(self, batch_id: str) -> None:
        """Poll until the batch is completed, expired or cancelled.

        A batch that failed validation raises LLMError (nothing ran). On
        timeout, cancels the batch (best-effort) and raises LLMBatchTimeout.
        """
        deadline = time.monotonic() + self.batch_timeout
        while True:
            with _translate_errors():
                batch = self._client.batches.retrieve(batch_id)
            if batch.status in _BATCH_DONE:
                return
            if batch.status == "failed":
                raise LLMError(f"Batch {batch_id} failed: {_describe_batch_errors(batch)}")
            if time.monotonic() >= deadline:
                click.echo()
                click.echo(f"  Timeout — cancelling batch {batch_id}…")
                try:
                    self._client.batches.cancel(batch_id)
                except Exception:  # noqa: BLE001, S110
                    pass
                raise LLMBatchTimeout(
                    f"Batch {batch_id} timed out after {self.batch_timeout}s"
                )
            click.echo(".", nl=False)
            time.sleep(
                min(self.batch_poll_interval, max(1, deadline - time.monotonic()))
            )

    def batch_results(self, batch_id: str) -> list[ToolCallResult]:
        """Read the batch's output file and error file, in that order."""
        lines: list[str] = []
        with _translate_errors():
            batch = self._client.batches.retrieve(batch_id)
            for file_id in (batch.output_file_id, batch.error_file_id):
                if file_id:
                    lines.extend(self._client.files.content(file_id).text.splitlines())
        return parse_batch_results(lines)

    # ---- request translation ----

    def _response_kwargs(self, request: ToolCallRequest) -> dict[str, Any]:
        """Responses-API parameters for a request; shared by sync and batch.

        This is where the task name becomes a concrete model id.
        """
        params = request["params"]
        tool = params["tool"]
        return {
            "model": self.resolve_model(request["task"]),
            "max_output_tokens": params["max_tokens"],
            "instructions": params["system"],
            "input": [{"role": "user", "content": params["user_prompt"]}],
            "tools": [
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": tool["name"]},
            "reasoning": {"effort": self.REASONING_EFFORT},
        }

    def _to_batch_line(self, request: ToolCallRequest) -> dict[str, Any]:
        return {
            "custom_id": request["custom_id"],
            "method": "POST",
            "url": self.BATCH_ENDPOINT,
            "body": self._response_kwargs(request),
        }


# -------- Pure translation helpers (unit-tested without the SDK) --------


def result_from_response(custom_id: str, body: dict, raw: Any = None) -> ToolCallResult:
    """Translate one Responses-API response body into a ToolCallResult.

    stop_reason labels, in order of precedence:
      - "max_tokens" when the output was cut off by max_output_tokens
        (any other incomplete reason is passed through as-is);
      - "refusal" when the model answered with a refusal instead of the tool;
      - the response status when it is not "completed" ("failed", …);
      - "tool_use" when a function_call with usable JSON arguments is
        present, else "no_tool_use_block".
    """
    output = body.get("output") or []
    status = body.get("status")
    if status == "incomplete":
        reason = (body.get("incomplete_details") or {}).get("reason")
        label = "max_tokens" if reason == "max_output_tokens" else reason or "incomplete"
        return ToolCallResult(custom_id, None, label, raw)
    if _has_refusal(output):
        return ToolCallResult(custom_id, None, "refusal", raw)
    if status != "completed":
        return ToolCallResult(custom_id, None, status or "unknown", raw)
    tool_input = _find_tool_input(output)
    stop_reason = "tool_use" if tool_input is not None else "no_tool_use_block"
    return ToolCallResult(custom_id, tool_input, stop_reason, raw)


def parse_batch_results(lines: Iterable[str]) -> list[ToolCallResult]:
    """Translate the lines of a batch's output and error files.

    Items that did not run or did not return HTTP 200 carry "expired",
    "canceled" or "errored" as stop_reason; the rest are translated by
    result_from_response. Blank lines are skipped. Output order is not
    guaranteed by OpenAI — callers match on custom_id.
    """
    results: list[ToolCallResult] = []
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id", "")
        error = item.get("error")
        response = item.get("response") or {}
        if error or response.get("status_code") != 200:
            code = (error or {}).get("code") or ""
            label = _ITEM_ERROR_LABELS.get(code, "errored")
            results.append(ToolCallResult(custom_id, None, label, item))
            continue
        results.append(result_from_response(custom_id, response.get("body") or {}, item))
    return results


def _find_tool_input(output: list[dict]) -> dict | None:
    """Return the decoded arguments of the first function_call item, or None.

    Under forced tool_choice the API guarantees the call belongs to the
    requested tool, so the name is not checked. Arguments that are not a
    JSON object are treated as absent.
    """
    for item in output:
        if item.get("type") == "function_call":
            try:
                arguments = json.loads(item.get("arguments") or "")
            except json.JSONDecodeError:
                return None
            return arguments if isinstance(arguments, dict) else None
    return None


def _has_refusal(output: list[dict]) -> bool:
    return any(
        part.get("type") == "refusal"
        for item in output
        if item.get("type") == "message"
        for part in item.get("content") or []
    )


def _describe_batch_errors(batch: Any) -> str:
    errors = getattr(getattr(batch, "errors", None), "data", None) or []
    messages = [getattr(e, "message", None) or str(e) for e in errors]
    return "; ".join(messages) or "no error details"
