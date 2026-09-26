"""Closed-loop tests for the summarizer task: inputs → prompt, tool input → summaries."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from frankenbote.llm.tasks import SUMMARIZER_TASK, SummarizerResponse
from tests.conftest import make_article, make_curated

PROMPTS = Path(__file__).parent / "fixtures" / "prompts"


def _articles(n: int) -> list:
    return [
        make_curated(article=make_article(link=f"https://example.com/{i}"))
        for i in range(n)
    ]


# ── identity and schema ──────────────────────────────────────────────────────

class TestIdentity:
    def test_identity(self):
        assert SUMMARIZER_TASK.name == "summarizer"
        assert SUMMARIZER_TASK.label == "Summarizer"
        assert SUMMARIZER_TASK.tool_name == "submit_summaries"
        assert "submit_summaries" in SUMMARIZER_TASK.system_prompt
        assert "UNVERTRAUTE" in SUMMARIZER_TASK.system_prompt
        assert "ÜBERSCHRIFT" in SUMMARIZER_TASK.system_prompt
        assert "ai_title" in SUMMARIZER_TASK.system_prompt

    def test_schema(self):
        schema = SUMMARIZER_TASK.tool["input_schema"]
        assert schema["required"] == ["summaries"]
        assert schema["additionalProperties"] is False
        item = schema["$defs"]["SummaryDecision"]
        assert item["required"] == ["article_index", "ai_title", "summary"]
        assert item["properties"]["article_index"] == {"minimum": 0, "type": "integer"}
        assert {"type": "null"} in item["properties"]["summary"]["anyOf"]
        assert {"type": "string"} in item["properties"]["ai_title"]["anyOf"]
        assert {"type": "null"} in item["properties"]["ai_title"]["anyOf"]

    def test_max_tokens_formula(self):
        assert SUMMARIZER_TASK.max_tokens_for(1) == 350
        assert SUMMARIZER_TASK.max_tokens_for(1_000_000) == 48000


# ── parse ────────────────────────────────────────────────────────────────────

class TestParse:
    def test_null_summary_accepted(self):
        resp = SUMMARIZER_TASK.parse(
            {"summaries": [{"article_index": 0, "ai_title": None, "summary": None}]}
        )
        assert isinstance(resp, SummarizerResponse)
        assert resp.summaries[0].summary is None
        assert resp.summaries[0].ai_title is None

    def test_missing_ai_title_raises(self):
        """Required, not defaulted: the model must decide, and OpenAI strict
        mode needs every property required."""
        with pytest.raises(ValidationError):
            SUMMARIZER_TASK.parse({"summaries": [{"article_index": 0, "summary": "S."}]})

    def test_normalises_json_string(self):
        resp = SUMMARIZER_TASK.parse(
            {"summaries": '[{"article_index": 2, "ai_title": "T", "summary": "S."}]'}
        )
        assert resp.summaries[0].article_index == 2

    def test_bad_index_raises(self):
        with pytest.raises(ValidationError):
            SUMMARIZER_TASK.parse(
                {"summaries": [{"article_index": "x", "ai_title": "T", "summary": 1}]}
            )


# ── render: inputs → prompt ──────────────────────────────────────────────────

class TestRender:
    def test_matches_the_golden_prompt(self):
        """Lead, non-lead, sectionless and empty-feed-summary in one prompt.

        Byte-identical to the recorded fixture — update it deliberately.
        """
        articles = [
            make_curated(
                article=make_article(
                    source_name="Nordbayern",
                    title="Stadtrat beschließt neuen Haushalt",
                    summary="Der Nürnberger Stadtrat hat den Haushalt für 2027 verabschiedet.",
                    link="https://example.com/a",
                ),
                section="politik_verwaltung",
                is_lead=True,
            ),
            make_curated(
                article=make_article(
                    source_name="BR24",
                    title="Kein Vorspann vorhanden",
                    summary="",
                    link="https://example.com/b",
                ),
                section="wirtschaft",
                is_lead=False,
            ),
            make_curated(
                article=make_article(
                    source_name="Fränkischer Tag",
                    title="Ohne Sektion",
                    summary="Ein kurzer Vorspann.",
                    link="https://example.com/c",
                ),
                section=None,
                is_lead=False,
            ),
        ]
        rendered = SUMMARIZER_TASK.render(articles)
        assert rendered == (PROMPTS / "summarize.txt").read_text(encoding="utf-8")

    def test_expected_count_matches_the_input(self):
        assert "7 Einträge erwartet" in SUMMARIZER_TASK.render(_articles(7))

    def test_empty_feed_summary_gets_a_placeholder(self):
        rendered = SUMMARIZER_TASK.render([make_curated(article=make_article(summary=""))])
        assert "<feed_summary>(leer)</feed_summary>" in rendered

    def test_untrusted_framing_is_present(self):
        assert "unvertraute Daten" in SUMMARIZER_TASK.render(_articles(1))


# ── interpret: tool input → aligned summaries ────────────────────────────────

class TestInterpret:
    def _interpret(self, n_articles: int, summaries: list[dict]):
        outcome = SUMMARIZER_TASK.interpret(
            SUMMARIZER_TASK.parse({"summaries": summaries}), _articles(n_articles)
        )
        return [(d.ai_title, d.summary) for d in outcome.values], outcome.notes

    def test_results_land_on_their_own_article(self):
        values, notes = self._interpret(
            3,
            [
                {"article_index": 2, "ai_title": "T3", "summary": "Drei."},
                {"article_index": 0, "ai_title": "T1", "summary": "Eins."},
                {"article_index": 1, "ai_title": "T2", "summary": None},
            ],
        )
        assert values == [("T1", "Eins."), ("T2", None), ("T3", "Drei.")]
        assert notes == []

    def test_output_length_equals_input_length(self):
        values, _ = self._interpret(
            4, [{"article_index": 0, "ai_title": "T", "summary": "S."}]
        )
        assert values == [("T", "S."), (None, None), (None, None), (None, None)]

    def test_null_and_missing_are_both_no_result(self):
        values, notes = self._interpret(
            2, [{"article_index": 0, "ai_title": None, "summary": None}]
        )
        assert values == [(None, None), (None, None)]
        assert notes == []

    def test_headline_without_summary_is_kept(self):
        values, _ = self._interpret(
            1, [{"article_index": 0, "ai_title": "Nur Titel", "summary": None}]
        )
        assert values == [("Nur Titel", None)]

    def test_out_of_range_index_is_dropped(self):
        values, _ = self._interpret(
            1, [{"article_index": 9, "ai_title": "T", "summary": "S."}]
        )
        assert values == [(None, None)]
