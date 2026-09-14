"""Build the configured LLMClient — the only place a provider is chosen."""

from __future__ import annotations

from frankenbote.llm.anthropic_client import AnthropicLLMClient
from frankenbote.llm.base import LLMClient
from frankenbote.llm.config import LLMConfig


def create_client(
    config: LLMConfig,
    *,
    use_batch: bool | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """Instantiate the client for `config.provider`.

    use_batch overrides the configured default for this run (the CLI's
    --batch-off flag); None keeps the configured value. Credentials are
    read from the environment unless api_key is given.
    """
    effective_batch = config.use_batch if use_batch is None else use_batch
    if config.provider == "anthropic":
        return AnthropicLLMClient(config.models, use_batch=effective_batch, api_key=api_key)
    raise ValueError(f"Unsupported LLM provider {config.provider!r}")
