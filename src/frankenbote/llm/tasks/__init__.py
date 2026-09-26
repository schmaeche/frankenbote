"""The AI steps Frankenbote runs, one module per task.

Each module defines a Task (system prompt, forced tool, response model,
output budget, prompt rendering and response interpretation). To add a
step: create a module here, subclass SingleCallTask or PerItemTask, export
it below, and add its model to config/config.yaml (`TASK_NAMES` in
`llm/config.py`).
"""

from frankenbote.llm.tasks.curate import CurateTask, curator_task
from frankenbote.llm.tasks.summarize import (
    SUMMARIZER_TASK,
    SummarizerResponse,
    SummarizeTask,
    SummaryDecision,
)
from frankenbote.llm.tasks.wrap_up import (
    WRAP_UP_TASK,
    WrapUpItem,
    WrapUpResponse,
    WrapUpTask,
)

__all__ = [
    "SUMMARIZER_TASK",
    "WRAP_UP_TASK",
    "CurateTask",
    "SummarizeTask",
    "SummarizerResponse",
    "SummaryDecision",
    "WrapUpItem",
    "WrapUpResponse",
    "WrapUpTask",
    "curator_task",
]
