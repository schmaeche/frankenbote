"""Tests for frankenbote.llm.task — the task contract itself.

The concrete tasks have their own files (test_task_curate.py,
test_task_summarize.py, test_task_wrap_up.py).
"""

import json

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from frankenbote.llm import (
    ItemNote,
    PerItemTask,
    SingleCallTask,
    TaskOutcome,
    normalize_array_field,
    tool_schema,
)
from tests.conftest import tool_result


# ── tool_schema ──────────────────────────────────────────────────────────────

class _Inner(BaseModel):
    """Inner docstring that must not reach the schema."""

    model_config = ConfigDict(extra="forbid")

    title: str  # a property literally named "title" must survive
    note: str | None = Field(None, description="kept: field-level description")


class _Outer(BaseModel):
    """Outer docstring that must not reach the schema."""

    model_config = ConfigDict(extra="forbid")

    items: list[_Inner]
    count: int = Field(..., ge=0)


class TestToolSchema:
    def test_strips_titles_everywhere(self):
        schema = tool_schema(_Outer)
        dumped = json.dumps(schema)
        # No auto-generated "title" annotations remain …
        assert '"title": "' not in dumped.replace('"title": {', "")
        # … but the property named "title" is still declared.
        assert "title" in schema["$defs"]["_Inner"]["properties"]
        assert schema["$defs"]["_Inner"]["properties"]["title"] == {"type": "string"}

    def test_strips_class_docstring_descriptions(self):
        schema = tool_schema(_Outer)
        assert "description" not in schema
        assert "description" not in schema["$defs"]["_Inner"]

    def test_keeps_field_descriptions(self):
        schema = tool_schema(_Outer)
        note = schema["$defs"]["_Inner"]["properties"]["note"]
        assert note["description"] == "kept: field-level description"

    def test_forbid_extra_emits_additional_properties_false(self):
        schema = tool_schema(_Outer)
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["_Inner"]["additionalProperties"] is False

    def test_constraints_and_required_preserved(self):
        schema = tool_schema(_Outer)
        assert schema["properties"]["count"] == {"minimum": 0, "type": "integer"}
        assert schema["required"] == ["items", "count"]
        assert schema["properties"]["items"]["items"] == {"$ref": "#/$defs/_Inner"}


# ── normalize_array_field ────────────────────────────────────────────────────

class TestNormalizeArrayField:
    @staticmethod
    def normalize(tool_input: dict) -> dict:
        return normalize_array_field(tool_input, "things")

    def test_list_passthrough(self):
        tool_input = {"things": [{"a": 1}]}
        assert self.normalize(tool_input) is tool_input

    def test_json_string_parsed(self):
        result = self.normalize({"things": json.dumps([{"a": 1}])})
        assert result == {"things": [{"a": 1}]}

    def test_does_not_mutate_input(self):
        tool_input = {"things": "[]"}
        self.normalize(tool_input)
        assert tool_input == {"things": "[]"}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="things was a string but not valid JSON"):
            self.normalize({"things": "not json"})

    def test_non_list_json_raises(self):
        with pytest.raises(ValueError, match="JSON content is dict"):
            self.normalize({"things": '{"a": 1}'})

    def test_missing_key_passthrough(self):
        assert self.normalize({"other": 1}) == {"other": 1}

    def test_other_keys_preserved(self):
        result = self.normalize({"things": "[1]", "extra": "x"})
        assert result == {"things": [1], "extra": "x"}


# ── Task: the shared half of the contract ────────────────────────────────────

class _Resp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: list[int]


class _Collect(SingleCallTask[str, int, _Resp]):
    """Minimal SingleCallTask: renders inputs joined, returns them doubled."""

    name = "curator"
    label = "Curator"
    system_prompt = "SYS"
    tool_name = "submit_values"
    tool_description = "Submit values."
    response_model = _Resp

    def max_tokens_for(self, n_items: int) -> int:
        return 10 * n_items

    def render(self, inputs):
        return "|".join(inputs)

    def interpret(self, response, inputs):
        return TaskOutcome([v * 2 for v in response.values])


class _Normalising(_Collect):
    def normalize(self, tool_input: dict) -> dict:
        return normalize_array_field(tool_input, "values")


class TestTask:
    def test_tool_definition(self):
        tool = _Collect().tool
        assert tool["name"] == "submit_values"
        assert tool["description"] == "Submit values."
        assert tool["input_schema"] == tool_schema(_Resp)

    def test_max_tokens_for(self):
        assert _Collect().max_tokens_for(4) == 40

    def test_parse_validates(self):
        assert _Collect().parse({"values": [1, 2]}) == _Resp(values=[1, 2])

    def test_parse_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _Collect().parse({"values": ["x"]})

    def test_parse_rejects_extra_keys(self):
        with pytest.raises(ValidationError):
            _Collect().parse({"values": [1], "extra": 1})

    def test_normalize_is_identity_by_default(self):
        tool_input = {"values": [1]}
        assert _Collect().normalize(tool_input) is tool_input

    def test_parse_applies_normalize(self):
        assert _Normalising().parse({"values": "[3]"}).values == [3]


class TestSingleCallTask:
    def test_render_receives_every_input(self):
        assert _Collect().render(["a", "b"]) == "a|b"

    def test_interpret_returns_aligned_values(self):
        outcome = _Collect().interpret(_Resp(values=[1, 2]), ["a", "b"])
        assert outcome.values == [2, 4]
        assert outcome.notes == []


# ── PerItemTask ──────────────────────────────────────────────────────────────

class _WrapResp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None


class _PerItem(PerItemTask[str, "str | None", _WrapResp]):
    """Minimal PerItemTask: one call per string, echoing it back."""

    name = "wrap_up"
    label = "Wrap-up"
    system_prompt = "SYS"
    tool_name = "submit_text"
    tool_description = "Submit text."
    response_model = _WrapResp

    def max_tokens_for(self, n_items: int) -> int:
        return 100

    def render(self, item):
        return f"<{item}>"

    def read(self, response):
        return response.text

    def missing(self):
        return None


class TestPerItemAddressing:
    def test_custom_id_uses_the_task_name(self):
        assert _PerItem().custom_id(3) == "wrap_up-3"

    def test_custom_id_round_trips(self):
        task = _PerItem()
        assert task.parse_custom_id(task.custom_id(17)) == 17

    def test_parse_rejects_a_foreign_prefix(self):
        assert _PerItem().parse_custom_id("summarizer-0") is None

    def test_parse_rejects_a_non_numeric_index(self):
        assert _PerItem().parse_custom_id("wrap_up-x") is None

    def test_items_pairs_each_input_with_its_id(self):
        assert _PerItem().items(["a", "b"]) == [
            ("wrap_up-0", "<a>"),
            ("wrap_up-1", "<b>"),
        ]


class TestPerItemInterpret:
    def test_results_land_on_their_own_input(self):
        outcome = _PerItem().interpret(
            [
                tool_result("wrap_up-1", {"text": "B"}),
                tool_result("wrap_up-0", {"text": "A"}),
            ],
            2,
        )
        assert outcome.values == ["A", "B"]
        assert outcome.notes == []

    def test_absent_result_falls_back_to_missing(self):
        outcome = _PerItem().interpret([tool_result("wrap_up-0", {"text": "A"})], 3)
        assert outcome.values == ["A", None, None]
        assert outcome.notes == []

    def test_null_value_is_not_an_error(self):
        outcome = _PerItem().interpret([tool_result("wrap_up-0", {"text": None})], 1)
        assert outcome.values == [None]
        assert outcome.notes == []

    def test_errored_item_is_noted(self):
        outcome = _PerItem().interpret([tool_result("wrap_up-0", None, "errored")], 1)
        assert outcome.values == [None]
        assert outcome.notes == [ItemNote(0, "errored")]

    def test_expired_item_is_noted(self):
        outcome = _PerItem().interpret([tool_result("wrap_up-0", None, "expired")], 1)
        assert outcome.notes == [ItemNote(0, "expired")]

    def test_missing_tool_block_is_noted(self):
        outcome = _PerItem().interpret(
            [tool_result("wrap_up-0", None, "no_tool_use_block")], 1
        )
        assert outcome.notes == [ItemNote(0, "no_tool_use_block")]

    def test_validation_failure_is_noted_and_does_not_raise(self):
        outcome = _PerItem().interpret([tool_result("wrap_up-0", {"bad": 1})], 1)
        assert outcome.values == [None]
        assert outcome.notes[0].index == 0
        assert outcome.notes[0].reason.startswith("validation:")

    def test_malformed_custom_id_is_noted_without_an_index(self):
        outcome = _PerItem().interpret([tool_result("wrap_up-x", {"text": "A"})], 1)
        assert outcome.values == [None]
        assert outcome.notes == [ItemNote(None, "unusable custom id 'wrap_up-x'")]

    def test_foreign_custom_id_is_noted_without_an_index(self):
        outcome = _PerItem().interpret([tool_result("summarizer", {})], 1)
        assert outcome.values == [None]
        assert outcome.notes[0].index is None

    def test_out_of_range_index_is_noted_without_an_index(self):
        outcome = _PerItem().interpret([tool_result("wrap_up-9", {"text": "A"})], 1)
        assert outcome.values == [None]
        assert outcome.notes[0].index is None

    def test_one_bad_item_does_not_cost_the_others(self):
        outcome = _PerItem().interpret(
            [
                tool_result("wrap_up-0", {"text": "A"}),
                tool_result("wrap_up-1", None, "errored"),
                tool_result("wrap_up-2", {"text": "C"}),
            ],
            3,
        )
        assert outcome.values == ["A", None, "C"]
        assert outcome.notes == [ItemNote(1, "errored")]


# ── TaskOutcome ──────────────────────────────────────────────────────────────

class TestTaskOutcome:
    def test_notes_default_to_empty(self):
        assert TaskOutcome([1, 2]).notes == []

    def test_frozen(self):
        with pytest.raises(Exception):
            TaskOutcome([1]).values = [2]  # type: ignore[misc]
