"""Anthropic implementation of LLMClient.

The only module in the pipeline that imports the Anthropic SDK. It owns:

  - credentials: the API key comes from the `api_key` argument or the
    ANTHROPIC_API_KEY environment variable, read on instantiation;
  - the synchronous streaming call and the Message Batches calls
    (create / retrieve / cancel / results) with their polling constants;
  - translation of Anthropic responses into ToolCallResult and of SDK
    exceptions into the provider-neutral LLMError hierarchy.

Model selection and retry behaviour are inherited unchanged from LLMClient.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from typing import Any

import anthropic
import click
import httpx
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request as BatchRequest

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

# Errors worth a retry. Everything else in anthropic.APIError (4xx, 5xx the
# SDK already retried internally, …) is surfaced as a plain LLMError.
_TRANSIENT_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    httpx.RemoteProtocolError,
)


@contextmanager
def _translate_errors() -> Generator[None]:
    """Map SDK exceptions onto the provider-neutral hierarchy."""
    try:
        yield
    except _TRANSIENT_ERRORS as exc:
        raise LLMTransientError(str(exc)) from exc
    except anthropic.APIError as exc:
        raise LLMError(str(exc)) from exc


class AnthropicLLMClient(LLMClient):
    """LLMClient backed by the Anthropic Python SDK."""

    BATCH_POLL_INTERVAL = 30  # seconds between status checks
    BATCH_TIMEOUT = 3_600  # 60-minute hard limit
    PROGRESS_EVERY = 25  # streamed text chunks per progress dot

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

        sdk_client lets tests inject a stand-in for anthropic.Anthropic; when
        given, no API key is required.
        """
        super().__init__(
            models,
            use_batch=use_batch,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        if sdk_client is None:
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            sdk_client = anthropic.Anthropic(api_key=api_key)
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
        """Streaming Messages call with the tool forced via tool_choice."""
        with (
            _translate_errors(),
            self._client.messages.stream(**self._message_kwargs(request)) as stream,
        ):
            for chunks_seen, _chunk in enumerate(stream.text_stream, start=1):
                if chunks_seen % self.PROGRESS_EVERY == 0:
                    click.echo(".", nl=False)
            msg = stream.get_final_message()

        tool_input = _find_tool_input(msg)
        return ToolCallResult(
            request["custom_id"], tool_input, msg.stop_reason or "unknown", msg
        )

    def submit_batch(self, requests: Sequence[ToolCallRequest]) -> str:
        with _translate_errors():
            batch = self._client.messages.batches.create(
                requests=[self._to_batch_request(r) for r in requests]
            )
        return batch.id

    def wait_for_batch(self, batch_id: str) -> None:
        """Poll until processing_status == 'ended'.

        On timeout, cancels the batch (best-effort) and raises LLMBatchTimeout.
        """
        deadline = time.monotonic() + self.batch_timeout
        while True:
            with _translate_errors():
                batch = self._client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return
            if time.monotonic() >= deadline:
                click.echo()
                click.echo(f"  Timeout — cancelling batch {batch_id}…")
                try:
                    self._client.messages.batches.cancel(batch_id)
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
        with _translate_errors():
            return parse_batch_results(self._client.messages.batches.results(batch_id))

    # ---- request translation ----

    def _message_kwargs(self, request: ToolCallRequest) -> dict[str, Any]:
        """Messages-API parameters for a request; shared by sync and batch.

        This is where the task name becomes a concrete model id.
        """
        params = request["params"]
        tool = params["tool"]
        return {
            "model": self.resolve_model(request["task"]),
            "max_tokens": params["max_tokens"],
            "system": params["system"],
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool["name"]},
            "messages": [{"role": "user", "content": params["user_prompt"]}],
        }

    def _to_batch_request(self, request: ToolCallRequest) -> BatchRequest:
        return BatchRequest(
            custom_id=request["custom_id"],
            params=MessageCreateParamsNonStreaming(**self._message_kwargs(request)),
        )


# -------- Pure translation helpers (unit-tested without the SDK) --------


def parse_batch_results(results_iter: Iterable[Any]) -> list[ToolCallResult]:
    """Translate an Anthropic batch results iterator into ToolCallResults.

    Items that did not succeed carry their batch status ("errored",
    "expired", "canceled") as stop_reason. Succeeded items yield "tool_use"
    when a tool_use block is present, else "no_tool_use_block".
    """
    results: list[ToolCallResult] = []
    for item in results_iter:
        if item.result.type != "succeeded":
            results.append(ToolCallResult(item.custom_id, None, item.result.type, item))
            continue
        tool_input = _find_tool_input(item.result.message)
        stop_reason = "tool_use" if tool_input is not None else "no_tool_use_block"
        results.append(ToolCallResult(item.custom_id, tool_input, stop_reason, item))
    return results


def _find_tool_input(msg: Any) -> dict | None:
    """Return the input of the first tool_use block, or None.

    Under forced tool_choice the API guarantees the block belongs to the
    requested tool, so the name is not checked. A non-dict input (the API
    contract is a dict) is treated as absent.
    """
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use":
            block_input = getattr(block, "input", None)
            return block_input if isinstance(block_input, dict) else None
    return None
