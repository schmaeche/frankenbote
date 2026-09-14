"""Tests for frankenbote.llm.task and the concrete tasks in llm/tasks/."""

import json

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from frankenbote.llm import TaskSpec, normalize_array_field, tool_schema
from frankenbote.llm.tasks import (
    SUMMARIZER_TASK,
    WRAP_UP_TASK,
    SummarizerResponse,
    WrapUpResponse,
    curator_task,
)
from frankenbote.models import CuratorDecision, CuratorResponse


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
    normalize = staticmethod(normalize_array_field("things"))

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


# ── TaskSpec ─────────────────────────────────────────────────────────────────

class _Resp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: list[int]


def _spec(**overrides) -> TaskSpec[_Resp]:
    kwargs = dict(
        name="curator",
        label="Curator",
        system_prompt="SYS",
        tool_name="submit_values",
        tool_description="Submit values.",
        response_model=_Resp,
        max_tokens=lambda n: 10 * n,
    )
    kwargs.update(overrides)
    return TaskSpec(**kwargs)


class TestTaskSpec:
    def test_tool_definition(self):
        tool = _spec().tool
        assert tool["name"] == "submit_values"
        assert tool["description"] == "Submit values."
        assert tool["input_schema"] == tool_schema(_Resp)

    def test_max_tokens_for(self):
        assert _spec().max_tokens_for(4) == 40

    def test_parse_validates(self):
        assert _spec().parse({"values": [1, 2]}) == _Resp(values=[1, 2])

    def test_parse_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _spec().parse({"values": ["x"]})

    def test_parse_rejects_extra_keys(self):
        with pytest.raises(ValidationError):
            _spec().parse({"values": [1], "extra": 1})

    def test_parse_applies_normalize(self):
        spec = _spec(normalize=normalize_array_field("values"))
        assert spec.parse({"values": "[3]"}).values == [3]

    def test_frozen(self):
        with pytest.raises(Exception):
            _spec().name = "x"  # type: ignore[misc]


# ── curator_task ─────────────────────────────────────────────────────────────

class TestCuratorTask:
    def test_identity(self):
        task = curator_task(["politik", "sport"])
        assert task.name == "curator"
        assert task.label == "Curator"
        assert task.tool_name == "submit_decisions"
        assert "submit_decisions" in task.system_prompt
        assert "UNTRUSTED INPUT" in task.system_prompt

    def test_schema_restricts_section_to_configured_ids(self):
        schema = curator_task(["politik", "sport"]).tool["input_schema"]
        section = schema["$defs"]["CuratorDecision"]["properties"]["section"]
        assert {"enum": ["politik", "sport"], "type": "string"} in section["anyOf"]
        assert {"type": "null"} in section["anyOf"]

    def test_schema_shape(self):
        schema = curator_task(["a"]).tool["input_schema"]
        assert schema["required"] == ["decisions"]
        assert schema["additionalProperties"] is False
        decision = schema["$defs"]["CuratorDecision"]
        assert decision["required"] == [
            "article_index", "section", "priority", "relevance_score", "rationale"
        ]
        assert decision["additionalProperties"] is False
        assert decision["properties"]["rationale"] == {"maxLength": 300, "type": "string"}
        assert schema["$defs"]["Priority"]["enum"] == ["P1", "P2", "P3", "P4"]
        assert "title" not in decision

    def test_parse_accepts_known_section_and_null(self):
        task = curator_task(["politik"])
        resp = task.parse({"decisions": [
            {"article_index": 0, "section": "politik", "priority": "P1",
             "relevance_score": 5.0, "rationale": "ok"},
            {"article_index": 1, "section": None, "priority": "P4",
             "relevance_score": 0.0, "rationale": "drop"},
        ]})
        assert isinstance(resp, CuratorResponse)
        assert all(isinstance(d, CuratorDecision) for d in resp.decisions)
        assert resp.decisions[0].section == "politik"
        assert resp.decisions[1].section is None

    def test_parse_rejects_unknown_section(self):
        with pytest.raises(ValidationError):
            curator_task(["politik"]).parse({"decisions": [
                {"article_index": 0, "section": "sport", "priority": "P1",
                 "relevance_score": 5.0, "rationale": "ok"},
            ]})

    def test_parse_normalises_json_string(self):
        decisions = [{"article_index": 0, "section": "politik", "priority": "P2",
                      "relevance_score": 1.0, "rationale": "r"}]
        resp = curator_task(["politik"]).parse({"decisions": json.dumps(decisions)})
        assert resp.decisions[0].priority.value == "P2"

    def test_max_tokens_formula(self):
        task = curator_task(["a"])
        assert task.max_tokens_for(1) == 650
        assert task.max_tokens_for(10) == 2000
        assert task.max_tokens_for(1_000_000) == 48000

    def test_requires_section_ids(self):
        with pytest.raises(ValueError, match="at least one section id"):
            curator_task([])


# ── SUMMARIZER_TASK ──────────────────────────────────────────────────────────

class TestSummarizerTask:
    def test_identity(self):
        assert SUMMARIZER_TASK.name == "summarizer"
        assert SUMMARIZER_TASK.label == "Summarizer"
        assert SUMMARIZER_TASK.tool_name == "submit_summaries"
        assert "submit_summaries" in SUMMARIZER_TASK.system_prompt
        assert "UNVERTRAUTE" in SUMMARIZER_TASK.system_prompt

    def test_schema(self):
        schema = SUMMARIZER_TASK.tool["input_schema"]
        assert schema["required"] == ["summaries"]
        assert schema["additionalProperties"] is False
        item = schema["$defs"]["SummaryDecision"]
        assert item["required"] == ["article_index", "summary"]
        assert item["properties"]["article_index"] == {"minimum": 0, "type": "integer"}
        assert {"type": "null"} in item["properties"]["summary"]["anyOf"]

    def test_parse(self):
        resp = SUMMARIZER_TASK.parse({"summaries": [{"article_index": 0, "summary": None}]})
        assert isinstance(resp, SummarizerResponse)
        assert resp.summaries[0].summary is None

    def test_parse_normalises_json_string(self):
        resp = SUMMARIZER_TASK.parse({"summaries": '[{"article_index": 2, "summary": "S."}]'})
        assert resp.summaries[0].article_index == 2

    def test_max_tokens_formula(self):
        assert SUMMARIZER_TASK.max_tokens_for(1) == 320
        assert SUMMARIZER_TASK.max_tokens_for(1_000_000) == 48000


# ── WRAP_UP_TASK ─────────────────────────────────────────────────────────────

class TestWrapUpTask:
    def test_identity(self):
        assert WRAP_UP_TASK.name == "wrap_up"
        assert WRAP_UP_TASK.label == "Wrap-up"
        assert WRAP_UP_TASK.tool_name == "submit_wrap_up"
        assert "German" in WRAP_UP_TASK.system_prompt
        assert "UNTRUSTED" in WRAP_UP_TASK.system_prompt

    def test_schema(self):
        schema = WRAP_UP_TASK.tool["input_schema"]
        assert schema["required"] == ["wrap_up"]
        assert schema["additionalProperties"] is False

    def test_parse_string_and_null(self):
        assert WRAP_UP_TASK.parse({"wrap_up": "Text"}).wrap_up == "Text"
        assert isinstance(WRAP_UP_TASK.parse({"wrap_up": None}), WrapUpResponse)

    def test_parse_missing_field_raises(self):
        with pytest.raises(ValidationError):
            WRAP_UP_TASK.parse({})

    def test_fixed_max_tokens(self):
        assert WRAP_UP_TASK.max_tokens_for(1) == 1200
        assert WRAP_UP_TASK.max_tokens_for(50) == 1200
