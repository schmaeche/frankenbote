"""Tests for frankenbote.llm.config — config.yaml loading and model lookup."""

import textwrap

import pytest

from frankenbote.llm import TASK_NAMES, LLMConfig, ModelConfig, load_llm_config

# ── ModelConfig ──────────────────────────────────────────────────────────────


class TestModelConfig:
    def test_for_task_direct(self):
        cfg = ModelConfig(curator="c", summarizer="s")
        assert cfg.for_task("curator") == "c"
        assert cfg.for_task("summarizer") == "s"

    def test_wrap_up_falls_back_to_summarizer(self):
        assert ModelConfig(curator="c", summarizer="s").for_task("wrap_up") == "s"

    def test_wrap_up_explicit(self):
        cfg = ModelConfig(curator="c", summarizer="s", wrap_up="w")
        assert cfg.for_task("wrap_up") == "w"

    def test_every_task_name_resolves(self):
        cfg = ModelConfig(curator="c", summarizer="s")
        for task in TASK_NAMES:
            assert cfg.for_task(task)

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM task 'headline'"):
            ModelConfig(curator="c", summarizer="s").for_task("headline")

    def test_missing_required_model_raises(self):
        with pytest.raises(Exception):
            ModelConfig(curator="c")  # type: ignore[call-arg]

    def test_unknown_key_rejected(self):
        with pytest.raises(Exception):
            ModelConfig(curator="c", summarizer="s", headline="h")  # type: ignore[call-arg]


# ── LLMConfig ────────────────────────────────────────────────────────────────


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig(models={"curator": "c", "summarizer": "s"})
        assert cfg.provider == "anthropic"
        assert cfg.use_batch is True

    def test_unknown_provider_rejected(self):
        with pytest.raises(Exception):
            LLMConfig(provider="gemini", models={"curator": "c", "summarizer": "s"})

    def test_unknown_key_rejected(self):
        with pytest.raises(Exception):
            LLMConfig(models={"curator": "c", "summarizer": "s"}, retries=3)


# ── load_llm_config ──────────────────────────────────────────────────────────

_VALID = textwrap.dedent("""\
    llm:
      provider: anthropic
      use_batch: false
      models:
        curator: claude-a
        summarizer: claude-b
        wrap_up: claude-c
""")


class TestLoadLLMConfig:
    def test_loads_valid_file(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(_VALID, encoding="utf-8")
        cfg = load_llm_config(path)
        assert cfg.provider == "anthropic"
        assert cfg.use_batch is False
        assert cfg.models.for_task("curator") == "claude-a"
        assert cfg.models.for_task("wrap_up") == "claude-c"

    def test_accepts_path_as_string(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(_VALID, encoding="utf-8")
        assert load_llm_config(str(path)).models.summarizer == "claude-b"

    def test_minimal_file_uses_defaults(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "llm:\n  models:\n    curator: a\n    summarizer: b\n", encoding="utf-8"
        )
        cfg = load_llm_config(path)
        assert cfg.use_batch is True
        assert cfg.models.wrap_up is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="LLM config not found"):
            load_llm_config(tmp_path / "nope.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("llm: [unclosed\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_llm_config(path)

    def test_missing_llm_key_raises(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("models:\n  curator: a\n", encoding="utf-8")
        with pytest.raises(ValueError, match="top-level 'llm:' key"):
            load_llm_config(path)

    def test_non_dict_yaml_raises(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="top-level 'llm:' key"):
            load_llm_config(path)

    def test_missing_model_reported_as_value_error(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("llm:\n  models:\n    curator: a\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid 'llm:' block") as ei:
            load_llm_config(path)
        assert "summarizer" in str(ei.value)

    def test_openai_provider_accepted(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "llm:\n  provider: openai\n  models:\n    curator: a\n    summarizer: b\n",
            encoding="utf-8",
        )
        assert load_llm_config(path).provider == "openai"

    def test_unknown_provider_reported_as_value_error(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "llm:\n  provider: gemini\n  models:\n    curator: a\n    summarizer: b\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Invalid 'llm:' block"):
            load_llm_config(path)
