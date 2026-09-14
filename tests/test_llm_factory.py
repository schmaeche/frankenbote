"""Tests for frankenbote.llm.factory — provider selection from config."""

from unittest.mock import MagicMock

import pytest

from frankenbote.llm import AnthropicLLMClient, LLMConfig, ModelConfig, create_client
from frankenbote.llm import anthropic_client as ac_module


@pytest.fixture
def fake_sdk(monkeypatch):
    """Replace anthropic.Anthropic so no key is needed; records the api_key."""
    created = {}

    def fake_anthropic(api_key):
        created["api_key"] = api_key
        return MagicMock()

    monkeypatch.setattr(ac_module.anthropic, "Anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    return created


_MODELS = ModelConfig(curator="c", summarizer="s")


class TestCreateClient:
    def test_anthropic_client_from_config(self, fake_sdk):
        client = create_client(LLMConfig(models=_MODELS))
        assert isinstance(client, AnthropicLLMClient)
        assert client.models is _MODELS
        assert client.use_batch is True
        assert fake_sdk["api_key"] == "sk-env"

    def test_configured_batch_default(self, fake_sdk):
        client = create_client(LLMConfig(models=_MODELS, use_batch=False))
        assert client.use_batch is False

    def test_use_batch_override_wins(self, fake_sdk):
        assert create_client(LLMConfig(models=_MODELS, use_batch=True), use_batch=False).use_batch is False
        assert create_client(LLMConfig(models=_MODELS, use_batch=False), use_batch=True).use_batch is True

    def test_none_keeps_configured_value(self, fake_sdk):
        assert create_client(LLMConfig(models=_MODELS, use_batch=False), use_batch=None).use_batch is False

    def test_explicit_api_key_passed_through(self, fake_sdk):
        create_client(LLMConfig(models=_MODELS), api_key="sk-explicit")
        assert fake_sdk["api_key"] == "sk-explicit"

    def test_missing_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            create_client(LLMConfig(models=_MODELS))

    def test_unsupported_provider_raises(self):
        # Pydantic rejects unknown providers at validation time; bypass it to
        # exercise the factory's own guard.
        config = LLMConfig.model_construct(provider="openai", use_batch=True, models=_MODELS)
        with pytest.raises(ValueError, match="Unsupported LLM provider 'openai'"):
            create_client(config)
