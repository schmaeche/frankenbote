"""Tests for frankenbote.llm.openai_client — SDK fully mocked, no network."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from frankenbote.llm import (
    LLMBatchTimeout,
    LLMError,
    LLMTransientError,
    ModelConfig,
    OpenAILLMClient,
    ToolCallParams,
    ToolCallRequest,
)
from frankenbote.llm import openai_client as oc_module
from frankenbote.llm.openai_client import (
    REASONING_HEADROOM,
    parse_batch_results,
    result_from_response,
)
from frankenbote.llm.tasks import SUMMARIZER_TASK, WRAP_UP_TASK, curator_task
from tests.conftest import make_curator_config

# ── helpers ──────────────────────────────────────────────────────────────────

_MODELS = ModelConfig(curator="gpt-curator", summarizer="gpt-summarizer")

_SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
_TOOL = {"name": "submit_things", "description": "d", "input_schema": _SCHEMA}


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


def _call(arguments) -> dict:
    """A function_call output item; dict arguments are JSON-encoded."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"type": "function_call", "name": "submit_things", "call_id": "c1", "arguments": arguments}


def _refusal() -> dict:
    return {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}


def _body(output, status: str = "completed", reason: str | None = None) -> dict:
    return {
        "status": status,
        "output": output,
        "incomplete_details": {"reason": reason} if reason else None,
    }


class _FakeResponse:
    """Stand-in for openai.types.responses.Response on a final stream event."""

    def __init__(self, body: dict):
        self.body = body

    def model_dump(self, mode: str = "python") -> dict:
        return self.body


def _install_stream(sdk, body, *, deltas: int = 0, final: str = "response.completed"):
    response = _FakeResponse(body)
    events = [SimpleNamespace(type="response.function_call_arguments.delta")] * deltas
    events.append(SimpleNamespace(type=final, response=response))
    sdk.responses.create.return_value = iter(events)
    return response


def _batch_line(custom_id: str, body=None, *, status_code: int = 200, error=None) -> str:
    response = None if body is None else {"status_code": status_code, "body": body}
    return json.dumps({"custom_id": custom_id, "response": response, "error": error})


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://api.test"))


def _bad_request_error() -> openai.BadRequestError:
    req = httpx.Request("POST", "https://api.test")
    return openai.BadRequestError(
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
    return MagicMock(name="openai.OpenAI")


@pytest.fixture
def client(sdk) -> OpenAILLMClient:
    return OpenAILLMClient(_MODELS, sdk_client=sdk, batch_poll_interval=5, batch_timeout=100)


# ── construction / credentials ───────────────────────────────────────────────

class TestConstruction:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
            OpenAILLMClient(_MODELS)

    def test_reads_api_key_from_env(self, monkeypatch):
        created = {}

        def fake_openai(api_key):
            created["api_key"] = api_key
            return MagicMock()

        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setattr(oc_module.openai, "OpenAI", fake_openai)
        OpenAILLMClient(_MODELS)
        assert created["api_key"] == "sk-env"

    def test_explicit_key_wins_over_env(self, monkeypatch):
        created = {}
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setattr(
            oc_module.openai, "OpenAI",
            lambda api_key: created.setdefault("api_key", api_key) and MagicMock(),
        )
        OpenAILLMClient(_MODELS, api_key="sk-explicit")
        assert created["api_key"] == "sk-explicit"

    def test_injected_sdk_client_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        c = OpenAILLMClient(_MODELS, sdk_client=MagicMock())
        assert c.models is _MODELS
        assert c.use_batch is True
        assert c.max_attempts == 2
        assert c.backoff_seconds == 0
        assert c.batch_poll_interval == OpenAILLMClient.BATCH_POLL_INTERVAL
        assert c.batch_timeout == OpenAILLMClient.BATCH_TIMEOUT

    def test_settings_are_constructor_args(self):
        c = OpenAILLMClient(
            _MODELS, sdk_client=MagicMock(), use_batch=False, max_attempts=3, backoff_seconds=1.5
        )
        assert c.use_batch is False
        assert c.max_attempts == 3
        assert c.backoff_seconds == 1.5


# ── call_tool (synchronous streaming) ────────────────────────────────────────

class TestCallTool:
    def test_builds_forced_strict_tool_request(self, client, sdk):
        _install_stream(sdk, _body([_call({"a": 1})]))
        client.call_tool(_request(task="curator"))
        kwargs = sdk.responses.create.call_args.kwargs
        assert kwargs["model"] == "gpt-curator"
        assert kwargs["max_output_tokens"] == 123
        assert kwargs["instructions"] == "SYSTEM"
        assert kwargs["input"] == [{"role": "user", "content": "USER"}]
        assert kwargs["tools"] == [
            {
                "type": "function",
                "name": "submit_things",
                "description": "d",
                "parameters": _SCHEMA,
                "strict": True,
            }
        ]
        assert kwargs["tool_choice"] == {"type": "function", "name": "submit_things"}
        assert kwargs["reasoning"] == {"effort": "none"}
        assert kwargs["stream"] is True

    def test_model_resolved_per_task(self, client, sdk):
        _install_stream(sdk, _body([_call({})]))
        client.call_tool(_request(task="summarizer"))
        assert sdk.responses.create.call_args.kwargs["model"] == "gpt-summarizer"

    def test_wrap_up_falls_back_to_summarizer_model(self, client, sdk):
        _install_stream(sdk, _body([_call({})]))
        client.call_tool(_request(task="wrap_up"))
        assert sdk.responses.create.call_args.kwargs["model"] == "gpt-summarizer"

    def test_unknown_task_raises_before_calling_sdk(self, client, sdk):
        with pytest.raises(ValueError, match="Unknown LLM task"):
            client.call_tool(_request(task="headline"))
        sdk.responses.create.assert_not_called()

    def test_returns_tool_input_and_stop_reason(self, client, sdk):
        response = _install_stream(sdk, _body([_call({"a": 1})]), deltas=60)
        result = client.call_tool(_request("cid"))
        assert result.custom_id == "cid"
        assert result.tool_input == {"a": 1}
        assert result.stop_reason == "tool_use"
        assert result.raw is response

    def test_incomplete_final_event_is_translated(self, client, sdk):
        _install_stream(
            sdk,
            _body([_call('{"a": ')], status="incomplete", reason="max_output_tokens"),
            final="response.incomplete",
        )
        result = client.call_tool(_request())
        assert result.tool_input is None
        assert result.stop_reason == "max_tokens"

    def test_stream_without_final_event_is_transient(self, client, sdk):
        sdk.responses.create.return_value = iter(
            [SimpleNamespace(type="response.function_call_arguments.delta")]
        )
        with pytest.raises(LLMTransientError, match="without a final response"):
            client.call_tool(_request())

    def test_connection_error_becomes_transient(self, client, sdk):
        sdk.responses.create.side_effect = _connection_error()
        with pytest.raises(LLMTransientError):
            client.call_tool(_request())

    def test_timeout_error_becomes_transient(self, client, sdk):
        sdk.responses.create.side_effect = openai.APITimeoutError(
            request=httpx.Request("POST", "https://api.test")
        )
        with pytest.raises(LLMTransientError):
            client.call_tool(_request())

    def test_error_while_streaming_becomes_transient(self, client, sdk):
        def events():
            yield SimpleNamespace(type="response.function_call_arguments.delta")
            raise httpx.RemoteProtocolError("peer closed")

        sdk.responses.create.return_value = events()
        with pytest.raises(LLMTransientError):
            client.call_tool(_request())

    def test_status_error_becomes_llm_error_not_transient(self, client, sdk):
        sdk.responses.create.side_effect = _bad_request_error()
        with pytest.raises(LLMError) as excinfo:
            client.call_tool(_request())
        assert not isinstance(excinfo.value, LLMTransientError)
        assert isinstance(excinfo.value, RuntimeError)
        assert isinstance(excinfo.value.__cause__, openai.BadRequestError)

    def test_unrelated_exceptions_propagate_untouched(self, client, sdk):
        sdk.responses.create.side_effect = KeyError("boom")
        with pytest.raises(KeyError):
            client.call_tool(_request())


# ── result_from_response ─────────────────────────────────────────────────────

class TestResultFromResponse:
    def test_completed_with_function_call(self):
        raw = object()
        result = result_from_response("cid", _body([_call({"k": "v"})]), raw)
        assert (result.custom_id, result.tool_input, result.stop_reason) == ("cid", {"k": "v"}, "tool_use")
        assert result.raw is raw

    def test_reasoning_item_before_call_is_skipped(self):
        output = [{"type": "reasoning", "summary": []}, _call({"k": "v"})]
        assert result_from_response("c", _body(output)).tool_input == {"k": "v"}

    def test_max_output_tokens_maps_to_max_tokens(self):
        body = _body([_call('{"k": ')], status="incomplete", reason="max_output_tokens")
        result = result_from_response("c", body)
        assert result.tool_input is None
        assert result.stop_reason == "max_tokens"

    def test_other_incomplete_reason_passed_through(self):
        body = _body([], status="incomplete", reason="content_filter")
        assert result_from_response("c", body).stop_reason == "content_filter"

    def test_incomplete_without_details(self):
        assert result_from_response("c", _body([], status="incomplete")).stop_reason == "incomplete"

    def test_refusal(self):
        result = result_from_response("c", _body([_refusal()]))
        assert result.tool_input is None
        assert result.stop_reason == "refusal"

    def test_failed_status(self):
        assert result_from_response("c", _body([], status="failed")).stop_reason == "failed"

    def test_missing_status_is_unknown(self):
        assert result_from_response("c", {"output": [_call({})]}).stop_reason == "unknown"

    def test_completed_without_function_call(self):
        output = [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]
        result = result_from_response("c", _body(output))
        assert result.tool_input is None
        assert result.stop_reason == "no_tool_use_block"

    @pytest.mark.parametrize("arguments", ["{not json", "[1, 2]", ""])
    def test_unusable_arguments_treated_as_absent(self, arguments):
        result = result_from_response("c", _body([_call(arguments)]))
        assert result.tool_input is None
        assert result.stop_reason == "no_tool_use_block"


# ── submit_batch ─────────────────────────────────────────────────────────────

class TestSubmitBatch:
    def _uploaded_lines(self, sdk) -> list[dict]:
        name, content = sdk.files.create.call_args.kwargs["file"]
        assert name.endswith(".jsonl")
        return [json.loads(line) for line in content.decode("utf-8").splitlines()]

    def test_uploads_jsonl_and_creates_batch(self, client, sdk):
        sdk.files.create.return_value.id = "file_1"
        sdk.batches.create.return_value.id = "batch_1"
        assert client.submit_batch([_request("a")]) == "batch_1"
        assert sdk.files.create.call_args.kwargs["purpose"] == "batch"
        sdk.batches.create.assert_called_once_with(
            input_file_id="file_1", endpoint="/v1/responses", completion_window="24h"
        )

    def test_one_line_per_request_with_resolved_models(self, client, sdk):
        client.submit_batch([_request("a", task="curator"), _request("b", task="wrap_up")])
        lines = self._uploaded_lines(sdk)
        assert [line["custom_id"] for line in lines] == ["a", "b"]
        assert all(line["method"] == "POST" for line in lines)
        assert all(line["url"] == "/v1/responses" for line in lines)
        assert lines[0]["body"]["model"] == "gpt-curator"
        assert lines[1]["body"]["model"] == "gpt-summarizer"
        assert lines[0]["body"]["tool_choice"] == {"type": "function", "name": "submit_things"}
        assert lines[0]["body"]["tools"][0]["strict"] is True
        assert lines[0]["body"]["reasoning"] == {"effort": "none"}
        assert "stream" not in lines[0]["body"]

    def test_non_ascii_prompt_survives_upload(self, client, sdk):
        request = _request()
        request["params"]["user_prompt"] = "Fürth – Nürnberg"
        client.submit_batch([request])
        assert self._uploaded_lines(sdk)[0]["body"]["input"][0]["content"] == "Fürth – Nürnberg"

    def test_connection_error_becomes_transient(self, client, sdk):
        sdk.files.create.side_effect = _connection_error()
        with pytest.raises(LLMTransientError):
            client.submit_batch([_request()])


# ── reasoning effort (sync and batch) ────────────────────────────────────────

def _sent_body(client: OpenAILLMClient, sdk, path: str, task: str) -> dict:
    """The Responses-API parameters one request goes out with on `path`."""
    if path == "sync":
        _install_stream(sdk, _body([_call({})]))
        client.call_tool(_request(task=task))
        return sdk.responses.create.call_args.kwargs
    client.submit_batch([_request(task=task)])
    _, content = sdk.files.create.call_args.kwargs["file"]
    return json.loads(content.decode("utf-8"))["body"]


@pytest.mark.parametrize("path", ["sync", "batch"])
class TestReasoningEffort:
    def _client(self, sdk, effort: dict[str, str]) -> OpenAILLMClient:
        return OpenAILLMClient(_MODELS, sdk_client=sdk, reasoning_effort=effort)

    def test_effort_taken_from_mapping_per_task(self, sdk, path):
        client = self._client(sdk, {"curator": "high", "summarizer": "low"})
        assert _sent_body(client, sdk, path, "summarizer")["reasoning"] == {"effort": "low"}

    def test_unlisted_task_gets_none_without_headroom(self, sdk, path):
        client = self._client(sdk, {"curator": "high"})
        body = _sent_body(client, sdk, path, "wrap_up")
        assert body["reasoning"] == {"effort": "none"}
        assert body["max_output_tokens"] == 123

    @pytest.mark.parametrize(("level", "headroom"), REASONING_HEADROOM.items())
    def test_known_level_adds_headroom(self, sdk, path, level, headroom):
        client = self._client(sdk, {"curator": level})
        body = _sent_body(client, sdk, path, "curator")
        assert body["reasoning"] == {"effort": level}
        assert body["max_output_tokens"] == 123 + headroom

    @pytest.mark.parametrize("level", ["none", "ultra"])
    def test_none_and_unknown_level_add_no_headroom(self, sdk, path, level):
        client = self._client(sdk, {"curator": level})
        body = _sent_body(client, sdk, path, "curator")
        assert body["reasoning"] == {"effort": level}
        assert body["max_output_tokens"] == 123


def test_reasoning_effort_mapping_is_copied():
    effort = {"curator": "medium"}
    client = OpenAILLMClient(_MODELS, sdk_client=MagicMock(), reasoning_effort=effort)
    effort["curator"] = "high"
    assert client.reasoning_effort == {"curator": "medium"}
    assert OpenAILLMClient(_MODELS, sdk_client=MagicMock()).reasoning_effort == {}


# ── wait_for_batch ───────────────────────────────────────────────────────────

class TestWaitForBatch:
    def _statuses(self, sdk, *statuses):
        sdk.batches.retrieve.side_effect = [SimpleNamespace(status=s) for s in statuses]

    def test_returns_when_completed(self, client, sdk, monkeypatch):
        fake = _FakeTime()
        monkeypatch.setattr(oc_module, "time", fake)
        self._statuses(sdk, "validating", "in_progress", "finalizing", "completed")
        client.wait_for_batch("b1")
        assert sdk.batches.retrieve.call_count == 4
        assert fake.sleeps == [5, 5, 5]
        sdk.batches.cancel.assert_not_called()

    @pytest.mark.parametrize("status", ["expired", "cancelled"])
    def test_returns_on_other_final_statuses(self, client, sdk, monkeypatch, status):
        monkeypatch.setattr(oc_module, "time", _FakeTime())
        self._statuses(sdk, status)
        client.wait_for_batch("b1")

    def test_failed_batch_raises_llm_error_with_details(self, client, sdk, monkeypatch):
        monkeypatch.setattr(oc_module, "time", _FakeTime())
        sdk.batches.retrieve.return_value = SimpleNamespace(
            status="failed",
            errors=SimpleNamespace(data=[SimpleNamespace(message="line 1: bad model")]),
        )
        with pytest.raises(LLMError, match="line 1: bad model") as excinfo:
            client.wait_for_batch("b1")
        assert not isinstance(excinfo.value, LLMTransientError)

    def test_failed_batch_without_details(self, client, sdk, monkeypatch):
        monkeypatch.setattr(oc_module, "time", _FakeTime())
        sdk.batches.retrieve.return_value = SimpleNamespace(status="failed", errors=None)
        with pytest.raises(LLMError, match="no error details"):
            client.wait_for_batch("b1")

    def test_timeout_cancels_and_raises(self, client, sdk, monkeypatch):
        fake = _FakeTime()
        monkeypatch.setattr(oc_module, "time", fake)
        sdk.batches.retrieve.return_value = SimpleNamespace(status="in_progress")
        with pytest.raises(LLMBatchTimeout, match="timed out after 100s"):
            client.wait_for_batch("b1")
        sdk.batches.cancel.assert_called_once_with("b1")
        # 100s budget / 5s polls → deadline reached after 20 sleeps
        assert len(fake.sleeps) == 20

    def test_cancel_failure_is_swallowed(self, client, sdk, monkeypatch):
        monkeypatch.setattr(oc_module, "time", _FakeTime())
        sdk.batches.retrieve.return_value = SimpleNamespace(status="in_progress")
        sdk.batches.cancel.side_effect = RuntimeError("cannot cancel")
        with pytest.raises(LLMBatchTimeout):
            client.wait_for_batch("b1")

    def test_retrieve_connection_error_becomes_transient(self, client, sdk, monkeypatch):
        monkeypatch.setattr(oc_module, "time", _FakeTime())
        sdk.batches.retrieve.side_effect = _connection_error()
        with pytest.raises(LLMTransientError):
            client.wait_for_batch("b1")


# ── batch_results / parse_batch_results ──────────────────────────────────────

class TestBatchResults:
    def _files(self, sdk, **contents):
        sdk.files.content.side_effect = lambda file_id: SimpleNamespace(text=contents[file_id])

    def test_reads_output_then_error_file(self, client, sdk):
        sdk.batches.retrieve.return_value = SimpleNamespace(output_file_id="out", error_file_id="err")
        self._files(
            sdk,
            out=_batch_line("a", _body([_call({"x": 1})])) + "\n",
            err=_batch_line("b", error={"code": "batch_expired", "message": "expired"}) + "\n",
        )
        results = client.batch_results("b1")
        sdk.batches.retrieve.assert_called_once_with("b1")
        assert [(r.custom_id, r.stop_reason) for r in results] == [("a", "tool_use"), ("b", "expired")]
        assert results[0].tool_input == {"x": 1}

    def test_missing_files_are_skipped(self, client, sdk):
        sdk.batches.retrieve.return_value = SimpleNamespace(output_file_id="out", error_file_id=None)
        self._files(sdk, out=_batch_line("a", _body([_call({})])))
        assert [r.custom_id for r in client.batch_results("b1")] == ["a"]
        sdk.files.content.assert_called_once_with("out")

    def test_connection_error_becomes_transient(self, client, sdk):
        sdk.batches.retrieve.return_value = SimpleNamespace(output_file_id="out", error_file_id=None)
        sdk.files.content.side_effect = _connection_error()
        with pytest.raises(LLMTransientError):
            client.batch_results("b1")


class TestParseBatchResults:
    _TOOL_INPUT = {"wrap_up": "Text"}

    def test_succeeded_line(self):
        line = _batch_line("wrap_up-0", _body([_call(self._TOOL_INPUT)]))
        [result] = parse_batch_results([line])
        assert result.custom_id == "wrap_up-0"
        assert result.stop_reason == "tool_use"
        assert result.tool_input == self._TOOL_INPUT
        assert result.raw == json.loads(line)

    @pytest.mark.parametrize(
        ("code", "label"),
        [("batch_expired", "expired"), ("batch_cancelled", "canceled"), ("server_error", "errored")],
    )
    def test_error_lines_carry_labels(self, code, label):
        [result] = parse_batch_results([_batch_line("c", error={"code": code, "message": "m"})])
        assert result.tool_input is None
        assert result.stop_reason == label

    def test_non_200_response_is_errored(self):
        line = _batch_line("c", {"error": {"message": "bad"}}, status_code=400)
        assert parse_batch_results([line])[0].stop_reason == "errored"

    def test_incomplete_body_maps_to_max_tokens(self):
        body = _body([_call('{"wrap_up": "Te')], status="incomplete", reason="max_output_tokens")
        assert parse_batch_results([_batch_line("c", body)])[0].stop_reason == "max_tokens"

    def test_blank_lines_skipped_and_order_preserved(self):
        lines = [
            _batch_line("b", error={"code": "server_error", "message": "m"}),
            "",
            _batch_line("a", _body([_call(self._TOOL_INPUT)])),
            "   ",
        ]
        results = parse_batch_results(lines)
        assert [r.custom_id for r in results] == ["b", "a"]
        assert [r.stop_reason for r in results] == ["errored", "tool_use"]

    def test_empty_input_returns_empty_list(self):
        assert parse_batch_results([]) == []


# ── strict-mode compatibility of every task's schema ────────────────────────

def _strict_violations(node, path: str = "$") -> list[str]:
    """Why a JSON schema would be rejected by OpenAI strict mode, if at all."""
    problems: list[str] = []
    if isinstance(node, list):
        for i, item in enumerate(node):
            problems += _strict_violations(item, f"{path}[{i}]")
        return problems
    if not isinstance(node, dict):
        return problems
    if "default" in node:
        problems.append(f"{path}: has a default")
    if "properties" in node:
        if node.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties is not false")
        optional = set(node["properties"]) - set(node.get("required", []))
        if optional:
            problems.append(f"{path}: optional properties {sorted(optional)}")
        for name, prop in node["properties"].items():
            problems += _strict_violations(prop, f"{path}.{name}")
    for key, value in node.items():
        if key == "properties":
            continue
        if key == "$defs":
            for name, definition in value.items():
                problems += _strict_violations(definition, f"{path}.$defs.{name}")
            continue
        problems += _strict_violations(value, f"{path}.{key}")
    return problems


class TestStrictSchemas:
    @pytest.mark.parametrize(
        "task",
        [curator_task(make_curator_config()), SUMMARIZER_TASK, WRAP_UP_TASK],
        ids=lambda t: t.name,
    )
    def test_task_schema_is_strict_compatible(self, task):
        assert _strict_violations(task.tool["input_schema"]) == []

    def test_checker_flags_optional_field(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string", "default": "x"}},
            "required": [],
            "additionalProperties": False,
        }
        assert _strict_violations(schema) == [
            "$: optional properties ['a']",
            "$.a: has a default",
        ]
