"""Build the configured LLMClient — the only place a provider is chosen."""

from __future__ import annotations

from frankenbote.llm.anthropic_client import AnthropicLLMClient
from frankenbote.llm.base import LLMClient
from frankenbote.llm.config import LLMConfig
from frankenbote.llm.openai_client import OpenAILLMClient

# config.yaml `llm.provider` → client implementation.
_CLIENTS: dict[str, type[AnthropicLLMClient | OpenAILLMClient]] = {
    "anthropic": AnthropicLLMClient,
    "openai": OpenAILLMClient,
}

# Task name → reasoning effort, passed to every client. Providers that
# don't support it ignore it; a task not listed here gets "none". Tune after
# real runs — see REASONING_HEADROOM in openai_client.py for the token
# headroom each level adds.
_REASONING_EFFORT: dict[str, str] = {
    "curator": "medium",
    "wrap_up": "low",
}


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
    client_class = _client_class(config.provider)
    return client_class(
        config.models,
        use_batch=effective_batch,
        api_key=api_key,
        reasoning_effort=_REASONING_EFFORT,
    )


def api_key_env(provider: str) -> str:
    """Name of the environment variable holding the provider's API key."""
    return _client_class(provider).API_KEY_ENV


def _client_class(provider: str) -> type[AnthropicLLMClient | OpenAILLMClient]:
    try:
        return _CLIENTS[provider]
    except KeyError:
        raise ValueError(f"Unsupported LLM provider {provider!r}") from None
