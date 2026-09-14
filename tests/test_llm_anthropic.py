"""Tests for frankenbote.llm.anthropic_client — SDK fully mocked, no network."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest
from anthropic.types import ToolUseBlock

from frankenbote.llm import (
    AnthropicLLMClient,
    LLMBatchTimeout,
    LLMError,
    LLMTransientError,
    ModelConfig,
    ToolCallParams,
    ToolCallRequest,
)
from frankenbote.llm import anthropic_client as ac_module
from frankenbote.llm.anthropic_client import parse_batch_results

# ── helpers ──────────────────────────────────────────────────────────────────

_MODELS = ModelConfig(curator="claude-curator", summarizer="claude-summarizer")

_TOOL = {
    "name": "submit_things",
    "description": "d",
    "input_schema": {"type": "object", "properties": {}},
}


def _request(custom_id: str = "req", task: str = "curator") -> ToolCallRequest:
    return ToolCallRequest(
        task=task,
        custom_id=custom_id,
        params=ToolCallParams(
            system="SYSTEM",
            user_prompt="USER",
            tool=_TOOL,
            max_tokens=123,
        ),
    )


def _tool_block(tool_input, name: str = "submit_things") -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id="toolu_1", name=name, input=tool_input)


def _message(content, stop_reason="tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def _succeeded(custom_id: str, content) -> SimpleNamespace:
    """Mock of MessageBatchIndividualResponse with a succeeded result."""
    result = SimpleNamespace(type="succeeded", message=SimpleNamespace(content=content))
    return SimpleNamespace(custom_id=custom_id, result=result)


def _failed(custom_id: str, result_type: str) -> SimpleNamespace:
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type=result_type))


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.test"))


def _bad_request_error() -> anthropic.BadRequestError:
    req = httpx.Request("POST", "https://api.test")
    return anthropic.BadRequestError(
        message="bad request", response=httpx.Response(400, request=req), body=None
    )


class _FakeTime:
    """Deterministic stand-in for the `time` module used by the client."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def sdk() -> MagicMock:
    return MagicMock(name="anthropic.Anthropic")


@pytest.fixture
def client(sdk) -> AnthropicLLMClient:
    return AnthropicLLMClient(_MODELS, sdk_client=sdk, batch_poll_interval=5, batch_timeout=100)


def _install_stream(sdk, message, text_chunks=()):
    stream = MagicMock()
    stream.text_stream = iter(text_chunks)
    stream.get_final_message.return_value = message
    sdk.messages.stream.return_value.__enter__.return_value = stream
    return stream


# ── construction / credentials ───────────────────────────────────────────────

class TestConstruction:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
            AnthropicLLMClient(_MODELS)

    def test_reads_api_key_from_env(self, monkeypatch):
        created = {}

        def fake_anthropic(api_key):
            created["api_key"] = api_key
            return MagicMock()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        monkeypatch.setattr(ac_module.anthropic, "Anthropic", fake_anthropic)
        AnthropicLLMClient(_MODELS)
        assert created["api_key"] == "sk-env"

    def test_explicit_key_wins_over_env(self, monkeypatch):
        created = {}
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        monkeypatch.setattr(
            ac_module.anthropic, "Anthropic",
            lambda api_key: created.setdefault("api_key", api_key) and MagicMock(),
        )
        AnthropicLLMClient(_MODELS, api_key="sk-explicit")
        assert created["api_key"] == "sk-explicit"

    def test_injected_sdk_client_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        c = AnthropicLLMClient(_MODELS, sdk_client=MagicMock())
        assert c.models is _MODELS
        assert c.use_batch is True
        assert c.max_attempts == 2
        assert c.backoff_seconds == 0
        assert c.batch_poll_interval == AnthropicLLMClient.BATCH_POLL_INTERVAL
        assert c.batch_timeout == AnthropicLLMClient.BATCH_TIMEOUT

    def test_settings_are_constructor_args(self):
        c = AnthropicLLMClient(
            _MODELS, sdk_client=MagicMock(), use_batch=False, max_attempts=3, backoff_seconds=1.5
        )
        assert c.use_batch is False
        assert c.max_attempts == 3
        assert c.backoff_seconds == 1.5


# ── call_tool (synchronous streaming) ────────────────────────────────────────

class TestCallTool:
    def test_builds_forced_tool_use_request(self, client, sdk):
        _install_stream(sdk, _message([_tool_block({"a": 1})]))
        client.call_tool(_request(task="curator"))
        kwargs = sdk.messages.stream.call_args.kwargs
        assert kwargs["model"] == "claude-curator"
        assert kwargs["max_tokens"] == 123
        assert kwargs["system"] == "SYSTEM"
        assert kwargs["tools"] == [_TOOL]
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_things"}
        assert kwargs["messages"] == [{"role": "user", "content": "USER"}]

    def test_model_resolved_per_task(self, client, sdk):
        _install_stream(sdk, _message([_tool_block({})]))
        client.call_tool(_request(task="summarizer"))
        assert sdk.messages.stream.call_args.kwargs["model"] == "claude-summarizer"

    def test_wrap_up_falls_back_to_summarizer_model(self, client, sdk):
        _install_stream(sdk, _message([_tool_block({})]))
        client.call_tool(_request(task="wrap_up"))
        assert sdk.messages.stream.call_args.kwargs["model"] == "claude-summarizer"

    def test_unknown_task_raises_before_calling_sdk(self, client, sdk):
        with pytest.raises(ValueError, match="Unknown LLM task"):
            client.call_tool(_request(task="headline"))
        sdk.messages.stream.assert_not_called()

    def test_returns_tool_input_and_stop_reason(self, client, sdk):
        msg = _message([_tool_block({"a": 1})])
        _install_stream(sdk, msg)
        result = client.call_tool(_request("cid"))
        assert result.custom_id == "cid"
        assert result.tool_input == {"a": 1}
        assert result.stop_reason == "tool_use"
        assert result.raw is msg

    def test_no_tool_block_returns_none_with_stop_reason(self, client, sdk):
        _install_stream(sdk, _message([SimpleNamespace(type="text", text="hi")], "max_tokens"))
        result = client.call_tool(_request())
        assert result.tool_input is None
        assert result.stop_reason == "max_tokens"

    def test_missing_stop_reason_becomes_unknown(self, client, sdk):
        _install_stream(sdk, _message([_tool_block({"a": 1})], stop_reason=None))
        assert client.call_tool(_request()).stop_reason == "unknown"

    def test_non_dict_tool_input_treated_as_absent(self, client, sdk):
        _install_stream(sdk, _message([SimpleNamespace(type="tool_use", name="t", input="str")]))
        assert client.call_tool(_request()).tool_input is None

    def test_consumes_text_stream(self, client, sdk):
        stream = _install_stream(sdk, _message([_tool_block({})]), text_chunks=["x"] * 60)
        client.call_tool(_request())
        stream.get_final_message.assert_called_once()

    def test_connection_error_becomes_transient(self, client, sdk):
        sdk.messages.stream.side_effect = _connection_error()
        with pytest.raises(LLMTransientError):
            client.call_tool(_request())

    def test_timeout_error_becomes_transient(self, client, sdk):
        sdk.messages.stream.side_effect = anthropic.APITimeoutError(
            request=httpx.Request("POST", "https://api.test")
        )
        with pytest.raises(LLMTransientError):
            client.call_tool(_request())

    def test_remote_protocol_error_becomes_transient(self, client, sdk):
        sdk.messages.stream.side_effect = httpx.RemoteProtocolError("peer closed")
        with pytest.raises(LLMTransientError):
            client.call_tool(_request())

    def test_status_error_becomes_llm_error_not_transient(self, client, sdk):
        sdk.messages.stream.side_effect = _bad_request_error()
        with pytest.raises(LLMError) as excinfo:
            client.call_tool(_request())
        assert not isinstance(excinfo.value, LLMTransientError)
        assert isinstance(excinfo.value, RuntimeError)
        assert isinstance(excinfo.value.__cause__, anthropic.BadRequestError)

    def test_unrelated_exceptions_propagate_untouched(self, client, sdk):
        sdk.messages.stream.side_effect = KeyError("boom")
        with pytest.raises(KeyError):
            client.call_tool(_request())


# ── submit_batch ─────────────────────────────────────────────────────────────

class TestSubmitBatch:
    def test_returns_batch_id(self, client, sdk):
        sdk.messages.batches.create.return_value.id = "msgbatch_1"
        assert client.submit_batch([_request("a")]) == "msgbatch_1"

    def test_builds_one_request_per_item_with_resolved_models(self, client, sdk):
        sdk.messages.batches.create.return_value.id = "b"
        client.submit_batch([_request("a", task="curator"), _request("b", task="wrap_up")])
        requests = sdk.messages.batches.create.call_args.kwargs["requests"]
        assert [r["custom_id"] for r in requests] == ["a", "b"]
        assert requests[0]["params"]["model"] == "claude-curator"
        assert requests[1]["params"]["model"] == "claude-summarizer"
        assert requests[0]["params"]["tool_choice"] == {"type": "tool", "name": "submit_things"}
        assert requests[0]["params"]["messages"] == [{"role": "user", "content": "USER"}]
        assert requests[0]["params"]["system"] == "SYSTEM"
        assert requests[0]["params"]["max_tokens"] == 123

    def test_connection_error_becomes_transient(self, client, sdk):
        sdk.messages.batches.create.side_effect = _connection_error()
        with pytest.raises(LLMTransientError):
            client.submit_batch([_request()])


# ── wait_for_batch ───────────────────────────────────────────────────────────

class TestWaitForBatch:
    def _statuses(self, sdk, *statuses):
        sdk.messages.batches.retrieve.side_effect = [
            SimpleNamespace(processing_status=s) for s in statuses
        ]

    def test_returns_when_ended(self, client, sdk, monkeypatch):
        fake = _FakeTime()
        monkeypatch.setattr(ac_module, "time", fake)
        self._statuses(sdk, "in_progress", "in_progress", "ended")
        client.wait_for_batch("b1")
        assert sdk.messages.batches.retrieve.call_count == 3
        assert fake.sleeps == [5, 5]
        sdk.messages.batches.cancel.assert_not_called()

    def test_immediately_ended_does_not_sleep(self, client, sdk, monkeypatch):
        fake = _FakeTime()
        monkeypatch.setattr(ac_module, "time", fake)
        self._statuses(sdk, "ended")
        client.wait_for_batch("b1")
        assert fake.sleeps == []

    def test_timeout_cancels_and_raises(self, client, sdk, monkeypatch):
        fake = _FakeTime()
        monkeypatch.setattr(ac_module, "time", fake)
        sdk.messages.batches.retrieve.return_value = SimpleNamespace(
            processing_status="in_progress"
        )
        with pytest.raises(LLMBatchTimeout, match="timed out after 100s"):
            client.wait_for_batch("b1")
        sdk.messages.batches.cancel.assert_called_once_with("b1")
        # 100s budget / 5s polls → deadline reached after 20 sleeps
        assert len(fake.sleeps) == 20

    def test_timeout_is_transient(self):
        assert issubclass(LLMBatchTimeout, LLMTransientError)

    def test_cancel_failure_is_swallowed(self, client, sdk, monkeypatch):
        fake = _FakeTime()
        monkeypatch.setattr(ac_module, "time", fake)
        sdk.messages.batches.retrieve.return_value = SimpleNamespace(
            processing_status="in_progress"
        )
        sdk.messages.batches.cancel.side_effect = RuntimeError("cannot cancel")
        with pytest.raises(LLMBatchTimeout):
            client.wait_for_batch("b1")

    def test_sleep_never_overshoots_deadline(self, sdk, monkeypatch):
        fake = _FakeTime()
        monkeypatch.setattr(ac_module, "time", fake)
        c = AnthropicLLMClient(_MODELS, sdk_client=sdk, batch_poll_interval=30, batch_timeout=7)
        self._statuses(sdk, "in_progress", "ended")
        c.wait_for_batch("b1")
        assert fake.sleeps == [7]

    def test_retrieve_connection_error_becomes_transient(self, client, sdk, monkeypatch):
        monkeypatch.setattr(ac_module, "time", _FakeTime())
        sdk.messages.batches.retrieve.side_effect = _connection_error()
        with pytest.raises(LLMTransientError):
            client.wait_for_batch("b1")


# ── batch_results / parse_batch_results ──────────────────────────────────────

class TestBatchResults:
    def test_fetches_and_parses(self, client, sdk):
        sdk.messages.batches.results.return_value = iter(
            [_succeeded("a", [_tool_block({"x": 1})])]
        )
        results = client.batch_results("b1")
        sdk.messages.batches.results.assert_called_once_with("b1")
        assert len(results) == 1
        assert results[0].custom_id == "a"
        assert results[0].tool_input == {"x": 1}

    def test_results_connection_error_becomes_transient(self, client, sdk):
        sdk.messages.batches.results.side_effect = _connection_error()
        with pytest.raises(LLMTransientError):
            client.batch_results("b1")


class TestParseBatchResults:
    _TOOL_INPUT = {"decisions": [{"article_index": 0}]}

    def test_succeeded_returns_tool_input_and_tool_use(self):
        item = _succeeded("curator", [_tool_block(self._TOOL_INPUT)])
        [result] = parse_batch_results([item])
        assert result.custom_id == "curator"
        assert result.stop_reason == "tool_use"
        assert result.tool_input == self._TOOL_INPUT
        assert result.raw is item

    @pytest.mark.parametrize("status", ["errored", "expired", "canceled"])
    def test_failed_item_carries_status(self, status):
        [result] = parse_batch_results([_failed("curator", status)])
        assert result.tool_input is None
        assert result.stop_reason == status

    def test_empty_iterator_returns_empty_list(self):
        assert parse_batch_results([]) == []

    def test_succeeded_without_tool_block(self):
        item = _succeeded("curator", [SimpleNamespace(type="text", text="hi")])
        [result] = parse_batch_results([item])
        assert result.tool_input is None
        assert result.stop_reason == "no_tool_use_block"

    def test_non_dict_tool_input_treated_as_absent(self):
        item = _succeeded("s", [SimpleNamespace(type="tool_use", name="t", input="[]")])
        [result] = parse_batch_results([item])
        assert result.tool_input is None
        assert result.stop_reason == "no_tool_use_block"

    def test_preserves_order_and_all_items(self):
        results = parse_batch_results([
            _failed("unrelated", "errored"),
            _succeeded("curator", [_tool_block(self._TOOL_INPUT)]),
        ])
        assert [r.custom_id for r in results] == ["unrelated", "curator"]
        assert [r.stop_reason for r in results] == ["errored", "tool_use"]

    def test_plain_namespace_blocks_accepted(self):
        block = SimpleNamespace(type="tool_use", name="submit_decisions", input={"k": "v"})
        [result] = parse_batch_results([_succeeded("c", [block])])
        assert result.tool_input == {"k": "v"}
