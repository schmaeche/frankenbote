"""Concrete paywall detection strategies, one module per strategy.

Each module holds one PaywallStrategy subclass plus its private helpers.
New strategies (DOM heuristics, per-publisher rules, content-length /
truncation checks, ...) get their own module here and are registered in
``detector.py``.
"""

from frankenbote.paywall.strategies.structured_metadata import (
    StructuredMetadataStrategy,
)

__all__ = ["StructuredMetadataStrategy"]
