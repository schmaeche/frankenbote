"""Closed-loop tests for the wrap-up task: one article → prompt, results → wrap-ups."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from frankenbote.llm import ItemNote
from frankenbote.llm.tasks import WRAP_UP_TASK, WrapUpResponse
from tests.conftest import make_article, make_curated, tool_result

PROMPTS = Path(__file__).parent / "fixtures" / "prompts"

_ARTICLE = make_curated(
    article=make_article(
        source_name="Nordbayern",
        title="Stadtrat beschließt neuen Haushalt",
        summary="Der Nürnberger Stadtrat hat den Haushalt für 2027 verabschiedet.",
        link="https://example.com/a",
    ),
    ai_title="Nürnberg verabschiedet Haushalt für 2027",
    ai_summary="Der Stadtrat hat den Nürnberger Haushalt für 2027 beschlossen.",
)

# The same article before the summarizer ran (or where it returned nulls),
# with an empty feed summary: only the always-present fields remain.
_BARE_ARTICLE = _ARTICLE.model_copy(
    update={
        "article": _ARTICLE.article.model_copy(update={"summary": ""}),
        "ai_title": None,
        "ai_summary": None,
    }
)


# ── identity and schema ──────────────────────────────────────────────────────

class TestIdentity:
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

    def test_fixed_max_tokens(self):
        assert WRAP_UP_TASK.max_tokens_for(1) == 1200
        assert WRAP_UP_TASK.max_tokens_for(50) == 1200


# ── parse ────────────────────────────────────────────────────────────────────

class TestParse:
    def test_string_and_null(self):
        assert WRAP_UP_TASK.parse({"wrap_up": "Text"}).wrap_up == "Text"
        assert isinstance(WRAP_UP_TASK.parse({"wrap_up": None}), WrapUpResponse)

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            WRAP_UP_TASK.parse({})


# ── render: one (article, body) pair → prompt ────────────────────────────────

class TestRender:
    def test_matches_the_golden_prompt_for_a_fetched_body(self):
        rendered = WRAP_UP_TASK.render(
            (_ARTICLE, "Der vollständige Artikeltext.\n\nZweiter Absatz.")
        )
        assert rendered == (PROMPTS / "wrap_up_body.txt").read_text(encoding="utf-8")

    def test_matches_the_golden_prompt_for_the_feed_snippet_fallback(self):
        rendered = WRAP_UP_TASK.render((_ARTICLE, _ARTICLE.article.summary))
        assert rendered == (PROMPTS / "wrap_up_snippet.txt").read_text(encoding="utf-8")

    def test_matches_the_golden_prompt_without_reference_fields(self):
        rendered = WRAP_UP_TASK.render((_BARE_ARTICLE, "Der Artikeltext."))
        assert rendered == (PROMPTS / "wrap_up_bare.txt").read_text(encoding="utf-8")

    def test_reference_fields_are_included_when_set(self):
        rendered = WRAP_UP_TASK.render((_ARTICLE, "Body."))
        assert f"<feed_summary>{_ARTICLE.article.summary}</feed_summary>" in rendered
        assert f"<edition_title>{_ARTICLE.ai_title}</edition_title>" in rendered
        assert f"<edition_summary>{_ARTICLE.ai_summary}</edition_summary>" in rendered

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_unset_reference_fields_leave_no_empty_tag(self, empty):
        article = _ARTICLE.model_copy(
            update={
                "article": _ARTICLE.article.model_copy(update={"summary": empty or ""}),
                "ai_title": empty,
                "ai_summary": empty,
            }
        )
        rendered = WRAP_UP_TASK.render((article, "Body."))
        for tag in ("feed_summary", "edition_title", "edition_summary"):
            assert f"<{tag}>" not in rendered
        assert "<title>" in rendered
        assert "<body>Body.</body>" in rendered

    def test_each_reference_field_is_independent(self):
        article = _ARTICLE.model_copy(update={"ai_title": None})
        rendered = WRAP_UP_TASK.render((article, "Body."))
        assert "<edition_title>" not in rendered
        assert "<edition_summary>" in rendered

    def test_system_prompt_covers_grounding_and_mismatch(self):
        prompt = WRAP_UP_TASK.system_prompt
        assert "<edition_summary>" in prompt
        assert "different story" in prompt
        assert "Do not repeat the edition summary" in prompt

    def test_untrusted_framing_is_present(self):
        assert "untrusted data" in WRAP_UP_TASK.render((_ARTICLE, "Body."))


# ── addressing and interpretation ────────────────────────────────────────────

class TestAddressing:
    def test_items_pair_each_article_with_its_id(self):
        items = WRAP_UP_TASK.items([(_ARTICLE, "A."), (_ARTICLE, "B.")])
        assert [custom_id for custom_id, _ in items] == ["wrap_up-0", "wrap_up-1"]
        assert "A." in items[0][1]
        assert "B." in items[1][1]

    def test_custom_id_round_trips(self):
        assert WRAP_UP_TASK.parse_custom_id(WRAP_UP_TASK.custom_id(4)) == 4


class TestInterpret:
    def test_results_land_on_their_own_article(self):
        outcome = WRAP_UP_TASK.interpret(
            [
                tool_result("wrap_up-1", {"wrap_up": "Text B."}),
                tool_result("wrap_up-0", {"wrap_up": "Text A."}),
            ],
            2,
        )
        assert outcome.values == ["Text A.", "Text B."]
        assert outcome.notes == []

    def test_null_wrap_up_is_not_an_error(self):
        outcome = WRAP_UP_TASK.interpret([tool_result("wrap_up-0", {"wrap_up": None})], 1)
        assert outcome.values == [None]
        assert outcome.notes == []

    def test_errored_item_is_noted(self):
        outcome = WRAP_UP_TASK.interpret([tool_result("wrap_up-0", None, "errored")], 1)
        assert outcome.values == [None]
        assert outcome.notes == [ItemNote(0, "errored")]

    def test_malformed_custom_id_is_noted(self):
        outcome = WRAP_UP_TASK.interpret([tool_result("wrap_up-x", {"wrap_up": "T."})], 1)
        assert outcome.values == [None]
        assert outcome.notes[0].index is None

    def test_validation_failure_is_noted(self):
        outcome = WRAP_UP_TASK.interpret([tool_result("wrap_up-0", {"bad": "x"})], 1)
        assert outcome.values == [None]
        assert outcome.notes[0].reason.startswith("validation:")
