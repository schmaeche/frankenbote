"""LLM client abstraction.

Pipeline stages import from here and never touch a provider SDK or a model
name directly:

    from frankenbote.llm import LLMClient          # type of the injected client
    from frankenbote.llm.tasks import SUMMARIZER_TASK

and run one AI step with

    outcome = client.run_task(SUMMARIZER_TASK, articles)   # aligned with articles

The CLI builds the client once per run:

    config = load_llm_config("config/config.yaml")
    client = create_client(config, use_batch=None)
"""

from frankenbote.llm.anthropic_client import AnthropicLLMClient
from frankenbote.llm.base import (
    LLMBatchTimeout,
    LLMClient,
    LLMError,
    LLMTransientError,
    ToolCallParams,
    ToolCallRequest,
    ToolCallResult,
)
from frankenbote.llm.config import TASK_NAMES, LLMConfig, ModelConfig, load_llm_config
from frankenbote.llm.factory import api_key_env, create_client
from frankenbote.llm.openai_client import OpenAILLMClient
from frankenbote.llm.task import (
    ItemNote,
    PerItemTask,
    SingleCallTask,
    Task,
    TaskOutcome,
    normalize_array_field,
    tool_schema,
)

__all__ = [
    "AnthropicLLMClient",
    "ItemNote",
    "LLMBatchTimeout",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "LLMTransientError",
    "ModelConfig",
    "OpenAILLMClient",
    "PerItemTask",
    "SingleCallTask",
    "TASK_NAMES",
    "Task",
    "TaskOutcome",
    "ToolCallParams",
    "ToolCallRequest",
    "ToolCallResult",
    "api_key_env",
    "create_client",
    "load_llm_config",
    "normalize_array_field",
    "tool_schema",
]
