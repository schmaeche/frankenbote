"""The AI steps Frankenbote runs, one module per task.

Each module defines a TaskSpec (system prompt, forced tool, response model,
output budget). To add a step: create a module here, define its spec, export
it below, and add its model to config/config.yaml (`TASK_NAMES` in
`llm/config.py`).
"""

from frankenbote.llm.tasks.curate import curator_task
from frankenbote.llm.tasks.summarize import SUMMARIZER_TASK, SummarizerResponse
from frankenbote.llm.tasks.wrap_up import WRAP_UP_TASK, WrapUpResponse

__all__ = [
    "SUMMARIZER_TASK",
    "SummarizerResponse",
    "WRAP_UP_TASK",
    "WrapUpResponse",
    "curator_task",
]
