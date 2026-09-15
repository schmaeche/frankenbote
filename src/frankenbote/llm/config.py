"""LLM configuration — loads and validates config/config.yaml.

The file names the provider, the batch-mode default and one model per AI
step (task). `ModelConfig.for_task()` is the single place that answers
"which model runs this step?", including the wrap-up → summarizer fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

# The AI steps the pipeline runs. Each Task.name must be one of these,
# and each needs a model in config.yaml (wrap_up may fall back).
TASK_NAMES: tuple[str, ...] = ("curator", "summarizer", "wrap_up")


class ModelConfig(BaseModel):
    """`llm.models` block: one model id per task."""

    model_config = ConfigDict(extra="forbid")

    curator: str
    summarizer: str
    wrap_up: str | None = None  # falls back to `summarizer` when unset

    def for_task(self, task: str) -> str:
        """Return the model id for a task name; raises ValueError if unknown."""
        if task == "wrap_up":
            return self.wrap_up or self.summarizer
        if task in ("curator", "summarizer"):
            return getattr(self, task)
        raise ValueError(
            f"Unknown LLM task {task!r}; expected one of {', '.join(TASK_NAMES)}"
        )


class LLMConfig(BaseModel):
    """Validated structure of config.yaml -> llm block."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic", "openai"] = "anthropic"
    use_batch: bool = True
    models: ModelConfig


def load_llm_config(path: Path | str = "config/config.yaml") -> LLMConfig:
    """Load and validate config.yaml.

    Raises ValueError with a message meant for the CLI user.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"LLM config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e
    if not isinstance(raw, dict) or "llm" not in raw:
        raise ValueError(f"{path} must contain a top-level 'llm:' key")
    try:
        return LLMConfig(**raw["llm"])
    except (ValidationError, TypeError) as e:
        raise ValueError(f"Invalid 'llm:' block in {path}: {e}") from e
