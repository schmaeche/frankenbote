"""The detector — runs the registered strategies in order.

To add a detection approach, implement PaywallStrategy in strategies.py
and append an instance to _STRATEGIES below. Call sites stay unchanged.
"""

from frankenbote.paywall.base import PaywallResult, PaywallStrategy
from frankenbote.paywall.strategies import StructuredMetadataStrategy

# Ordered: cheapest / most reliable signals first.
_STRATEGIES: tuple[PaywallStrategy, ...] = (StructuredMetadataStrategy(),)


def is_paywalled(html: str, url: str | None = None) -> PaywallResult | None:
    """Run the registered detection strategies in order, returning the
    first definitive result.

    Returns None when no strategy decides — "couldn't tell", which is
    distinct from a PaywallResult with paywalled=False ("definitely free").
    """
    for strategy in _STRATEGIES:
        result = strategy.detect(html, url)
        if result is not None:
            return result
    return None
