"""LLM client abstraction.

Pipeline stages import from here and never touch a provider SDK directly:

    from frankenbote.llm import AnthropicLLMClient, LLMClient, ToolCallRequest
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

__all__ = [
    "AnthropicLLMClient",
    "LLMBatchTimeout",
    "LLMClient",
    "LLMError",
    "LLMTransientError",
    "ToolCallParams",
    "ToolCallRequest",
    "ToolCallResult",
]
