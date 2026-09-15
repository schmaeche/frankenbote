"""Shared fixture factories for the frankenbote test suite."""

from datetime import UTC, datetime

from frankenbote.curator import CuratorConfig
from frankenbote.models import (
    Article,
    Category,
    CuratedArticle,
    Priority,
    Source,
)

# A fixed "now" used across tests — a Wednesday, well inside any rolling window.
FIXED_NOW = datetime(2026, 5, 6, 12, 0, 0, tzinfo=UTC)

# A timestamp inside a typical previous-Saturday window (published on Monday).
IN_WINDOW_DATE = datetime(2026, 5, 4, 10, 0, 0, tzinfo=UTC)


def make_article(**overrides) -> Article:
    defaults = dict(
        source_id="test_src",
        source_name="Test Source",
        title="Test Article",
        link="https://example.com/article/1",
        summary="A short test summary.",
        published=IN_WINDOW_DATE,
        fetched_at=FIXED_NOW,
    )
    defaults.update(overrides)
    return Article(**defaults)


def make_source(**overrides) -> Source:
    defaults = dict(
        id="test_src",
        name="Test Source",
        url="https://example.com/feed",
        category=Category.LOCAL,
    )
    defaults.update(overrides)
    return Source(**defaults)


def make_curated(**overrides) -> CuratedArticle:
    article = overrides.pop("article", make_article())
    defaults = dict(
        article=article,
        section="politik_verwaltung",
        priority=Priority.P1,
        relevance_score=7.0,
        rationale="Test rationale.",
        is_lead=False,
    )
    defaults.update(overrides)
    return CuratedArticle(**defaults)


def make_curator_config(**overrides) -> CuratorConfig:
    """Build a minimal CuratorConfig without loading any YAML files."""
    raw = dict(
        guidance="Prioritise local Franconian news.",
        priorities=[
            {"id": "P1", "label": "Lokal", "description": "Local Franconia news"},
            {"id": "P2", "label": "Regional", "description": "Bavaria news"},
            {"id": "P3", "label": "National", "description": "Germany news"},
            {"id": "P4", "label": "Überregional", "description": "International news"},
        ],
        sections=[
            {
                "id": "politik_verwaltung",
                "display_name": "Politik & Verwaltung",
                "description": "Local politics and administration.",
            },
            {
                "id": "wirtschaft",
                "display_name": "Wirtschaft",
                "description": "Economy and business.",
            },
            {
                "id": "kultur",
                "display_name": "Kultur",
                "description": "Culture and events.",
            },
        ],
    )
    raw.update(overrides)
    return CuratorConfig(**raw)


# ── LLM client stand-in ──────────────────────────────────────────────────────

from frankenbote.llm import (  # noqa: E402
    LLMClient,
    ModelConfig,
    ToolCallRequest,
    ToolCallResult,
)

# Model config used by every scripted client unless a test overrides it.
TEST_MODELS = ModelConfig(curator="test-curator-model", summarizer="test-summarizer-model")


class ScriptedLLMClient(LLMClient):
    """LLMClient whose primitives replay a scripted list of outcomes.

    Each entry in `outcomes` is consumed by one primitive call:
      - for the sync path (call_tool): a ToolCallResult, or an exception
        instance to raise;
      - for the batch path: an exception (raised by submit_batch) or a
        list[ToolCallResult] (returned by batch_results).

    `calls` records every primitive invocation as (name, argument).
    """

    def __init__(self, outcomes=(), models: ModelConfig | None = None, **kwargs):
        super().__init__(TEST_MODELS if models is None else models, **kwargs)
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, object]] = []

    def _pop(self):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        self.calls.append(("call_tool", request))
        return self._pop()

    def submit_batch(self, requests) -> str:
        self.calls.append(("submit_batch", list(requests)))
        if self.outcomes and isinstance(self.outcomes[0], BaseException):
            self._pop()
        return f"batch_{len(self.calls)}"

    def wait_for_batch(self, batch_id: str) -> None:
        self.calls.append(("wait_for_batch", batch_id))

    def batch_results(self, batch_id: str) -> list[ToolCallResult]:
        self.calls.append(("batch_results", batch_id))
        return self._pop()


def tool_result(custom_id: str, tool_input, stop_reason: str = "tool_use", raw=None) -> ToolCallResult:
    return ToolCallResult(custom_id, tool_input, stop_reason, raw)
