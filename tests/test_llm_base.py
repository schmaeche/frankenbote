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
from frankenbote.llm import base as base_module
from tests.conftest import ScriptedLLMClient, tool_result


class _Parsed(BaseModel):
    value: int


def _parse(tool_input: dict) -> _Parsed:
    return _Parsed(**tool_input)


def _request(custom_id: str = "req") -> ToolCallRequest:
    return ToolCallRequest(
        custom_id=custom_id,
        params=ToolCallParams(
            model="m", system="s", user_prompt="u",
            tool={"name": "t", "input_schema": {}}, max_tokens=10,
        ),
    )


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
        with pytest.raises(RuntimeError, match="Claude refused on safety grounds"):
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
