"""Internal debug helpers — small utilities for failure diagnostics."""

from datetime import datetime
from pathlib import Path
from typing import Any


_DEBUG_DIR = Path("data/debug")


def save_failure(
    component: str,
    attempt: int,
    error: str,
    raw_output: Any,
) -> Path:
    """Save a failure context to disk for post-mortem inspection.

    Returns the file path written.
    """
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = _DEBUG_DIR / f"{component}-failure-{timestamp}.txt"

    body = (
        f"Component:  {component}\n"
        f"Attempt:    {attempt}\n"
        f"Time:       {datetime.now().isoformat()}\n"
        f"Error:      {error}\n"
        f"---\n"
        f"{raw_output!r}\n"
    )
    path.write_text(body, encoding="utf-8")
    return path