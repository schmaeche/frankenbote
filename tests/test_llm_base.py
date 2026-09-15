"""Tests for frankenbote.llm.base — the retry loops, driven by a scripted client."""

import pytest
from pydantic import BaseModel

from frankenbote.llm import (
    LLMBatchTimeout,
    LLMClient,
    LLMError,
    LLMTransientError,
    ToolCallParams,
    ToolCallRequest,
    ToolCallResult,
)
from frankenbote.llm import ItemNote, ModelConfig, PerItemTask, SingleCallTask, TaskOutcome
from frankenbote.llm import base as base_module
from tests.conftest import TEST_MODELS, ScriptedLLMClient, tool_result


class _Parsed(BaseModel):
    value: int


def _parse(tool_input: dict) -> _Parsed:
    return _Parsed(**tool_input)


def _request(custom_id: str = "req", task: str = "curator") -> ToolCallRequest:
    return ToolCallRequest(
        task=task,
        custom_id=custom_id,
        params=ToolCallParams(
            system="s", user_prompt="u",
            tool={"name": "t", "input_schema": {}}, max_tokens=10,
        ),
    )


class _Doubler(SingleCallTask[int, int, _Parsed]):
    """One call for every input; the response value is doubled per input."""

    name = "summarizer"
    label = "Summarizer"
    system_prompt = "SYSTEM"
    tool_name = "submit_things"
    tool_description = "desc"
    response_model = _Parsed

    def max_tokens_for(self, n_items: int) -> int:
        return 100 + 10 * n_items

    def render(self, inputs):
        return "|".join(str(i) for i in inputs)

    def interpret(self, response, inputs):
        return TaskOutcome([response.value * i for i in inputs])


class _Echo(PerItemTask[str, "str | None", _Parsed]):
    """One call per input."""

    name = "wrap_up"
    label = "Wrap-up"
    system_prompt = "SYSTEM"
    tool_name = "submit_thing"
    tool_description = "desc"
    response_model = _Parsed

    def max_tokens_for(self, n_items: int) -> int:
        return 50

    def render(self, item):
        return f"<{item}>"

    def read(self, response):
        return str(response.value)

    def missing(self):
        return None


_SPEC = _Doubler()


_OK = tool_result("req", {"value": 42})


@pytest.fixture
def no_debug(monkeypatch):
    """Replace save_failure with a recorder; returns the list of calls."""
    calls: list[tuple] = []

    def fake_save_failure(component, attempt, error, raw):
        calls.append((component, attempt, error, raw))
        return f"data/debug/{component}-fake.txt"

    monkeypatch.setattr(base_module, "save_failure", fake_save_failure)
    return calls


# ── construction / error hierarchy ───────────────────────────────────────────

class TestBasics:
    def test_defaults(self):
        c = ScriptedLLMClient()
        assert c.max_attempts == 2
        assert c.backoff_seconds == 0

    def test_rejects_zero_attempts(self):
        with pytest.raises(ValueError):
            ScriptedLLMClient(max_attempts=0)

    def test_rejects_negative_backoff(self):
        with pytest.raises(ValueError):
            ScriptedLLMClient(backoff_seconds=-1)

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            LLMClient()  # type: ignore[abstract]

    def test_error_hierarchy(self):
        assert issubclass(LLMError, RuntimeError)
        assert issubclass(LLMTransientError, LLMError)
        assert issubclass(LLMBatchTimeout, LLMTransientError)

    def test_tool_call_result_is_frozen(self):
        r = tool_result("a", {})
        with pytest.raises(Exception):
            r.stop_reason = "x"  # type: ignore[misc]


# ── call_tool_with_retry: sync path ──────────────────────────────────────────

class TestCallToolWithRetrySync:
    def test_success_first_attempt(self, no_debug):
        c = ScriptedLLMClient([_OK])
        attempts = []
        parsed = c.call_tool_with_retry(
            _request(), _parse, component="curator", use_batch=False,
            on_attempt=attempts.append,
        )
        assert parsed == _Parsed(value=42)
        assert attempts == [1]
        assert [name for name, _ in c.calls] == ["call_tool"]
        assert no_debug == []

    def test_transient_then_success(self, no_debug):
        c = ScriptedLLMClient([LLMTransientError("net"), _OK])
        attempts = []
        parsed = c.call_tool_with_retry(
            _request(), _parse, component="curator", use_batch=False,
            on_attempt=attempts.append,
        )
        assert parsed.value == 42
        assert attempts == [1, 2]
        assert no_debug == []

    def test_transient_twice_raises_without_debug_dump(self, no_debug):
        c = ScriptedLLMClient([LLMTransientError("net1"), LLMTransientError("net2")])
        with pytest.raises(RuntimeError, match="Curator failed twice due to network errors") as ei:
            c.call_tool_with_retry(_request(), _parse, component="curator", use_batch=False)
        assert "net2" in str(ei.value)
        assert isinstance(ei.value.__cause__, LLMTransientError)
        assert no_debug == []

    def test_batch_timeout_is_retried_as_transient(self, no_debug):
        c = ScriptedLLMClient([LLMBatchTimeout("slow"), _OK])
        assert c.call_tool_with_retry(
            _request(), _parse, component="curator", use_batch=False
        ).value == 42

    def test_non_transient_llm_error_propagates_immediately(self, no_debug):
        c = ScriptedLLMClient([LLMError("400 bad request"), _OK])
        with pytest.raises(LLMError):
            c.call_tool_with_retry(_request(), _parse, component="curator", use_batch=False)
        assert len(c.calls) == 1

    def test_max_tokens_twice_dumps_and_raises(self, no_debug):
        raw = object()
        bad = tool_result("req", None, "max_tokens", raw=raw)
        c = ScriptedLLMClient([bad, bad])
        with pytest.raises(RuntimeError) as ei:
            c.call_tool_with_retry(_request(), _parse, component="curator", use_batch=False)
        msg = str(ei.value)
        assert msg.startswith("Curator failed twice. attempt 2: response truncated (max_tokens hit)")
        assert "Debug context saved to data/debug/curator-fake.txt" in msg
        assert no_debug == [("curator", 2, "attempt 2: response truncated (max_tokens hit)", raw)]

    def test_refusal_message(self, no_debug):
        bad = tool_result("req", None, "refusal")
        c = ScriptedLLMClient([bad, bad])
        with pytest.raises(RuntimeError, match="model refused on safety grounds"):
            c.call_tool_with_retry(_request(), _parse, component="summarizer", use_batch=False)
        assert no_debug[0][0] == "summarizer"

    def test_unexpected_stop_reason_message(self, no_debug):
        bad = tool_result("req", None, "end_turn")
        c = ScriptedLLMClient([bad, bad])
        with pytest.raises(RuntimeError, match="unexpected stop_reason 'end_turn'"):
            c.call_tool_with_retry(_request(), _parse, component="curator", use_batch=False)

    def test_bad_stop_then_success(self, no_debug):
        c = ScriptedLLMClient([tool_result("req", None, "max_tokens"), _OK])
        assert c.call_tool_with_retry(
            _request(), _parse, component="curator", use_batch=False
        ).value == 42
        assert no_debug == []

    def test_missing_tool_block_twice(self, no_debug):
        bad = tool_result("req", None, "tool_use", raw="RAW")
        c = ScriptedLLMClient([bad, bad])
        with pytest.raises(RuntimeError, match="no tool_use block in response"):
            c.call_tool_with_retry(_request(), _parse, component="curator", use_batch=False)
        assert no_debug[0][3] == "RAW"

    def test_validation_failure_twice_dumps_tool_input(self, no_debug):
        bad = tool_result("req", {"value": "not-an-int"})
        c = ScriptedLLMClient([bad, bad])
        with pytest.raises(RuntimeError, match="Curator tool output invalid after retry") as ei:
            c.call_tool_with_retry(_request(), _parse, component="curator", use_batch=False)
        assert "validation" in str(ei.value)
        component, attempt, error, raw = no_debug[0]
        assert (component, attempt) == ("curator", 2)
        assert raw == {"value": "not-an-int"}

    def test_validation_failure_then_success(self, no_debug):
        c = ScriptedLLMClient([tool_result("req", {"value": "x"}), _OK])
        assert c.call_tool_with_retry(
            _request(), _parse, component="curator", use_batch=False
        ).value == 42

    def test_non_validation_parse_error_propagates(self, no_debug):
        def exploding_parse(_):
            raise ValueError("not a ValidationError")

        c = ScriptedLLMClient([_OK, _OK])
        with pytest.raises(ValueError, match="not a ValidationError"):
            c.call_tool_with_retry(_request(), exploding_parse, component="curator", use_batch=False)
        assert len(c.calls) == 1

    def test_save_debug_false_skips_dump(self, no_debug):
        bad = tool_result("req", None, "max_tokens")
        c = ScriptedLLMClient([bad, bad])
        with pytest.raises(RuntimeError) as ei:
            c.call_tool_with_retry(
                _request(), _parse, component="wrap-up", use_batch=False, save_debug=False
            )
        assert "Debug context" not in str(ei.value)
        assert str(ei.value).startswith("Wrap-up failed twice.")
        assert no_debug == []

    def test_three_attempts_wording(self, no_debug):
        bad = tool_result("req", None, "max_tokens")
        c = ScriptedLLMClient([bad, bad, bad], max_attempts=3)
        with pytest.raises(RuntimeError, match="Curator failed 3 times"):
            c.call_tool_with_retry(_request(), _parse, component="curator", use_batch=False)
        assert len(c.calls) == 3

    def test_single_attempt_never_retries(self, no_debug):
        c = ScriptedLLMClient([LLMTransientError("net"), _OK], max_attempts=1)
        with pytest.raises(RuntimeError):
            c.call_tool_with_retry(_request(), _parse, component="curator", use_batch=False)
        assert len(c.calls) == 1

    def test_backoff_sleeps_between_attempts(self, no_debug, monkeypatch):
        sleeps = []
        monkeypatch.setattr(base_module.time, "sleep", sleeps.append)
        c = ScriptedLLMClient(
            [LLMTransientError("a"), tool_result("req", None, "max_tokens"), _OK],
            max_attempts=3, backoff_seconds=1.5,
        )
        c.call_tool_with_retry(_request(), _parse, component="curator", use_batch=False)
        assert sleeps == [1.5, 3.0]  # exponential: 1.5 * 2^0, 1.5 * 2^1

    def test_no_backoff_by_default(self, no_debug, monkeypatch):
        sleeps = []
        monkeypatch.setattr(base_module.time, "sleep", sleeps.append)
        c = ScriptedLLMClient([LLMTransientError("a"), _OK])
        c.call_tool_with_retry(_request(), _parse, component="curator", use_batch=False)
        assert sleeps == []


# ── call_tool_with_retry: batch path ─────────────────────────────────────────

class TestCallToolWithRetryBatch:
    def test_routes_through_batch_primitives(self, no_debug):
        c = ScriptedLLMClient([[_OK]])
        parsed = c.call_tool_with_retry(_request(), _parse, component="curator")
        assert parsed.value == 42
        names = [name for name, _ in c.calls]
        assert names == ["submit_batch", "wait_for_batch", "batch_results"]
        assert c.calls[0][1] == [_request()]

    def test_picks_matching_custom_id(self, no_debug):
        c = ScriptedLLMClient([[tool_result("other", {"value": 1}), _OK]])
        assert c.call_tool_with_retry(_request(), _parse, component="curator").value == 42

    def test_missing_custom_id_is_no_result_and_retried(self, no_debug):
        c = ScriptedLLMClient([[tool_result("other", {"value": 1})], [_OK]])
        assert c.call_tool_with_retry(_request(), _parse, component="curator").value == 42

    def test_missing_custom_id_twice_raises(self, no_debug):
        c = ScriptedLLMClient([[], []])
        with pytest.raises(RuntimeError, match="unexpected stop_reason 'no_result'"):
            c.call_tool_with_retry(_request(), _parse, component="curator")

    def test_errored_item_message(self, no_debug):
        bad = [tool_result("req", None, "errored")]
        c = ScriptedLLMClient([bad, bad])
        with pytest.raises(RuntimeError, match="unexpected stop_reason 'errored'"):
            c.call_tool_with_retry(_request(), _parse, component="curator")

    def test_submit_transient_then_success(self, no_debug):
        c = ScriptedLLMClient([LLMTransientError("net"), [_OK]])
        assert c.call_tool_with_retry(_request(), _parse, component="curator").value == 42
        names = [name for name, _ in c.calls]
        assert names == ["submit_batch", "submit_batch", "wait_for_batch", "batch_results"]


# ── run_batch_with_retry ─────────────────────────────────────────────────────

class TestRunBatchWithRetry:
    def test_returns_all_results(self):
        results = [tool_result("wrapup-0-0", {"w": 1}), tool_result("wrapup-0-1", None, "errored")]
        c = ScriptedLLMClient([results])
        attempts = []
        out = c.run_batch_with_retry(
            [_request("wrapup-0-0"), _request("wrapup-0-1")],
            component="wrap-up batch", on_attempt=attempts.append,
        )
        assert out == results
        assert attempts == [1]

    def test_per_item_failures_are_not_retried(self):
        c = ScriptedLLMClient([[tool_result("a", None, "errored")]])
        out = c.run_batch_with_retry([_request("a")], component="wrap-up batch")
        assert out[0].stop_reason == "errored"
        assert [n for n, _ in c.calls].count("submit_batch") == 1

    def test_transient_then_success(self):
        c = ScriptedLLMClient([LLMTransientError("net"), [_OK]])
        attempts = []
        out = c.run_batch_with_retry([_request()], component="wrap-up batch", on_attempt=attempts.append)
        assert out == [_OK]
        assert attempts == [1, 2]

    def test_transient_twice_raises(self):
        c = ScriptedLLMClient([LLMTransientError("net1"), LLMBatchTimeout("slow")])
        with pytest.raises(RuntimeError, match="Wrap-up batch failed twice. Last error: slow") as ei:
            c.run_batch_with_retry([_request()], component="wrap-up batch")
        assert isinstance(ei.value.__cause__, LLMBatchTimeout)

    def test_non_transient_error_propagates(self):
        c = ScriptedLLMClient([LLMError("400"), [_OK]])
        with pytest.raises(LLMError):
            c.run_batch_with_retry([_request()], component="wrap-up batch")

    def test_backoff_applies(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(base_module.time, "sleep", sleeps.append)
        c = ScriptedLLMClient([LLMTransientError("net"), [_OK]], backoff_seconds=2)
        c.run_batch_with_retry([_request()], component="wrap-up batch")
        assert sleeps == [2]


# ── composed helpers ─────────────────────────────────────────────────────────

class TestComposed:
    def test_run_batch_sequence(self):
        c = ScriptedLLMClient([[_OK]])
        assert c.run_batch([_request()]) == [_OK]
        assert [n for n, _ in c.calls] == ["submit_batch", "wait_for_batch", "batch_results"]
        assert c.calls[1][1] == c.calls[2][1]  # same batch id handed through

    def test_call_tool_batched_no_result(self):
        c = ScriptedLLMClient([[]])
        result = c.call_tool_batched(_request("x"))
        assert result == ToolCallResult("x", None, "no_result")


# ── model selection ──────────────────────────────────────────────────────────

class TestModelSelection:
    def test_resolve_model_uses_config(self):
        c = ScriptedLLMClient(models=ModelConfig(curator="cur", summarizer="sum"))
        assert c.resolve_model("curator") == "cur"
        assert c.resolve_model("summarizer") == "sum"

    def test_wrap_up_falls_back_to_summarizer(self):
        c = ScriptedLLMClient(models=ModelConfig(curator="cur", summarizer="sum"))
        assert c.resolve_model("wrap_up") == "sum"

    def test_wrap_up_override(self):
        c = ScriptedLLMClient(models=ModelConfig(curator="cur", summarizer="sum", wrap_up="w"))
        assert c.resolve_model("wrap_up") == "w"

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM task 'nope'"):
            ScriptedLLMClient().resolve_model("nope")

    def test_default_test_models(self):
        assert ScriptedLLMClient().models is TEST_MODELS

    def test_use_batch_stored(self):
        assert ScriptedLLMClient().use_batch is True
        assert ScriptedLLMClient(use_batch=False).use_batch is False


# ── build_request / run_prompt / run_prompt_batch ────────────────────────────

class TestPromptApi:
    def test_build_request_shape(self):
        c = ScriptedLLMClient()
        request = c.build_request(_SPEC, "USER", n_items=3)
        assert request["task"] == "summarizer"
        assert request["custom_id"] == "summarizer"  # defaults to the task name
        params = request["params"]
        assert params["system"] == "SYSTEM"
        assert params["user_prompt"] == "USER"
        assert params["max_tokens"] == 130
        assert params["tool"]["name"] == "submit_things"
        assert params["tool"]["description"] == "desc"
        assert params["tool"]["input_schema"]["properties"] == {"value": {"type": "integer"}}
        assert "model" not in params

    def test_build_request_custom_id(self):
        request = ScriptedLLMClient().build_request(_SPEC, "u", custom_id="item-7")
        assert request["custom_id"] == "item-7"

    def test_run_prompt_uses_client_batch_default(self, no_debug):
        c = ScriptedLLMClient([[tool_result("summarizer", {"value": 5})]])
        parsed = c.run_prompt(_SPEC, "USER")
        assert parsed == _Parsed(value=5)
        assert [n for n, _ in c.calls] == ["submit_batch", "wait_for_batch", "batch_results"]

    def test_run_prompt_sync_when_client_configured_off(self, no_debug):
        c = ScriptedLLMClient([tool_result("summarizer", {"value": 5})], use_batch=False)
        assert c.run_prompt(_SPEC, "USER").value == 5
        assert [n for n, _ in c.calls] == ["call_tool"]

    def test_run_prompt_use_batch_override(self, no_debug):
        c = ScriptedLLMClient([tool_result("summarizer", {"value": 5})])  # default batch on
        assert c.run_prompt(_SPEC, "USER", use_batch=False).value == 5
        assert [n for n, _ in c.calls] == ["call_tool"]

    def test_run_prompt_passes_request_to_primitive(self, no_debug):
        c = ScriptedLLMClient([tool_result("summarizer", {"value": 5})], use_batch=False)
        c.run_prompt(_SPEC, "USER", n_items=2)
        _, request = c.calls[0]
        assert request["task"] == "summarizer"
        assert request["params"]["max_tokens"] == 120

    def test_run_prompt_error_uses_task_label_and_name(self, no_debug):
        bad = tool_result("summarizer", None, "max_tokens")
        c = ScriptedLLMClient([bad, bad], use_batch=False)
        with pytest.raises(RuntimeError, match="Summarizer failed twice"):
            c.run_prompt(_SPEC, "USER")
        assert no_debug[0][0] == "summarizer"

    def test_run_prompt_parse_via_task(self, no_debug):
        bad = tool_result("summarizer", {"value": "x"})
        c = ScriptedLLMClient([bad, tool_result("summarizer", {"value": 1})], use_batch=False)
        assert c.run_prompt(_SPEC, "USER").value == 1

    def test_run_prompt_on_attempt_and_save_debug(self, no_debug):
        attempts = []
        bad = tool_result("summarizer", None, "refusal")
        c = ScriptedLLMClient([bad, bad], use_batch=False)
        with pytest.raises(RuntimeError) as ei:
            c.run_prompt(_SPEC, "USER", on_attempt=attempts.append, save_debug=False)
        assert attempts == [1, 2]
        assert "Debug context" not in str(ei.value)
        assert no_debug == []

    def test_run_prompt_batch_builds_one_request_per_item(self):
        results = [tool_result("a", {"value": 1}), tool_result("b", None, "errored")]
        c = ScriptedLLMClient([results])
        out = c.run_prompt_batch(_SPEC, [("a", "prompt A"), ("b", "prompt B")], n_items=1)
        assert out == results
        _, requests = c.calls[0]
        assert [r["custom_id"] for r in requests] == ["a", "b"]
        assert [r["params"]["user_prompt"] for r in requests] == ["prompt A", "prompt B"]
        assert all(r["task"] == "summarizer" for r in requests)
        assert requests[0]["params"]["max_tokens"] == 110

    def test_run_prompt_batch_error_label(self):
        c = ScriptedLLMClient([LLMTransientError("a"), LLMTransientError("b")])
        with pytest.raises(RuntimeError, match="Summarizer batch failed twice. Last error: b"):
            c.run_prompt_batch(_SPEC, [("a", "p")])

    def test_run_prompt_batch_on_attempt(self):
        attempts = []
        c = ScriptedLLMClient([LLMTransientError("a"), [tool_result("a", {"value": 1})]])
        c.run_prompt_batch(_SPEC, [("a", "p")], on_attempt=attempts.append)
        assert attempts == [1, 2]


# ── run_task: the entry point the pipeline calls ─────────────────────────────

class TestRunTaskSingleCall:
    def test_renders_inputs_and_returns_aligned_values(self, no_debug):
        c = ScriptedLLMClient([[tool_result("summarizer", {"value": 3})]])
        outcome = c.run_task(_SPEC, [1, 2, 4])
        assert outcome.values == [3, 6, 12]
        assert outcome.notes == []
        [request] = c.calls[0][1]
        assert request["params"]["user_prompt"] == "1|2|4"
        assert request["params"]["max_tokens"] == 130  # n_items == len(inputs)

    def test_no_inputs_never_reaches_the_provider(self):
        c = ScriptedLLMClient()
        outcome = c.run_task(_SPEC, [])
        assert outcome == TaskOutcome([], [])
        assert c.calls == []

    def test_on_attempt_reports_each_attempt(self, no_debug):
        attempts: list[int] = []
        c = ScriptedLLMClient(
            [LLMTransientError("net"), tool_result("summarizer", {"value": 1})],
            use_batch=False,
        )
        c.run_task(_SPEC, [1], on_attempt=attempts.append)
        assert attempts == [1, 2]

    def test_persistent_failure_still_raises(self, no_debug):
        bad = tool_result("summarizer", None, "max_tokens")
        c = ScriptedLLMClient([bad, bad], use_batch=False)
        with pytest.raises(RuntimeError, match="Summarizer failed twice"):
            c.run_task(_SPEC, [1])

    def test_misaligned_task_is_caught(self, no_debug):
        class _Broken(_Doubler):
            def interpret(self, response, inputs):
                return TaskOutcome([1])  # one value for two inputs

        c = ScriptedLLMClient([tool_result("summarizer", {"value": 1})], use_batch=False)
        with pytest.raises(RuntimeError, match="returned 1 values for 2 inputs"):
            c.run_task(_Broken(), [1, 2])

    def test_unknown_task_shape_is_rejected(self):
        class _Neither:
            name = "x"
            label = "X"

        with pytest.raises(TypeError, match="neither a SingleCallTask nor a PerItemTask"):
            ScriptedLLMClient().run_task(_Neither(), [1])  # type: ignore[arg-type]


class TestRunTaskPerItemBatched:
    def test_one_batch_with_one_request_per_input(self):
        c = ScriptedLLMClient([[
            tool_result("wrap_up-0", {"value": 1}),
            tool_result("wrap_up-1", {"value": 2}),
        ]])
        outcome = c.run_task(_Echo(), ["a", "b"])
        assert outcome.values == ["1", "2"]
        assert outcome.notes == []
        _, requests = c.calls[0]
        assert [r["custom_id"] for r in requests] == ["wrap_up-0", "wrap_up-1"]
        assert [r["params"]["user_prompt"] for r in requests] == ["<a>", "<b>"]

    def test_failed_item_becomes_a_note_and_the_missing_value(self):
        c = ScriptedLLMClient([[
            tool_result("wrap_up-0", None, "errored"),
            tool_result("wrap_up-1", {"value": 2}),
        ]])
        outcome = c.run_task(_Echo(), ["a", "b"])
        assert outcome.values == [None, "2"]
        assert outcome.notes == [ItemNote(0, "errored")]

    def test_whole_batch_is_retried_on_a_transient_error(self):
        c = ScriptedLLMClient(
            [LLMTransientError("net"), [tool_result("wrap_up-0", {"value": 1})]]
        )
        assert c.run_task(_Echo(), ["a"]).values == ["1"]


class TestRunTaskPerItemSync:
    def test_one_call_per_input(self):
        c = ScriptedLLMClient(
            [tool_result("wrap_up", {"value": 1}), tool_result("wrap_up", {"value": 2})],
            use_batch=False,
        )
        outcome = c.run_task(_Echo(), ["a", "b"])
        assert outcome.values == ["1", "2"]
        assert [n for n, _ in c.calls] == ["call_tool", "call_tool"]
        assert [r["params"]["user_prompt"] for _, r in c.calls] == ["<a>", "<b>"]

    def test_renders_the_same_prompts_as_the_batch_path(self):
        sync = ScriptedLLMClient(
            [tool_result("wrap_up", {"value": 1})], use_batch=False
        )
        sync.run_task(_Echo(), ["a"])
        batched = ScriptedLLMClient([[tool_result("wrap_up-0", {"value": 1})]])
        batched.run_task(_Echo(), ["a"])
        assert (
            sync.calls[0][1]["params"]["user_prompt"]
            == batched.calls[0][1][0]["params"]["user_prompt"]
        )

    def test_a_persistently_failing_item_is_noted_not_raised(self, no_debug):
        bad = tool_result("wrap_up", None, "max_tokens")
        c = ScriptedLLMClient(
            [bad, bad, tool_result("wrap_up", {"value": 2})], use_batch=False
        )
        outcome = c.run_task(_Echo(), ["a", "b"])
        assert outcome.values == [None, "2"]
        assert outcome.notes[0].index == 0
        assert "Wrap-up failed twice" in outcome.notes[0].reason

    def test_per_item_failures_never_write_debug_dumps(self, no_debug):
        bad = tool_result("wrap_up", None, "max_tokens")
        c = ScriptedLLMClient([bad, bad], use_batch=False)
        c.run_task(_Echo(), ["a"])
        assert no_debug == []

    def test_on_attempt_fires_once_for_the_whole_task(self):
        attempts: list[int] = []
        c = ScriptedLLMClient(
            [tool_result("wrap_up", {"value": 1}), tool_result("wrap_up", {"value": 2})],
            use_batch=False,
        )
        c.run_task(_Echo(), ["a", "b"], on_attempt=attempts.append)
        assert attempts == [1]
