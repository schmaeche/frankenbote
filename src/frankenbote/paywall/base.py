"""Strategy interface and result type for paywall detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PaywallResult:
    """A definitive verdict from one detection strategy.

    A PaywallResult always means "decided" — strategies that cannot tell
    return None instead, so the detector moves on to the next one.
    """

    paywalled: bool
    strategy: str  # which strategy decided, for logging/observability


class PaywallStrategy(ABC):
    """One way of recognising a paywall (structured metadata, DOM
    heuristics, per-publisher rules, ...).

    The three-state contract that makes strategies composable:
    a PaywallResult with paywalled=True or paywalled=False is definitive
    and stops the chain; None means "no signal here", and the detector
    tries the next strategy.
    """

    # Short identifier echoed in PaywallResult.strategy.
    name: str

    @abstractmethod
    def detect(self, html: str, url: str | None) -> PaywallResult | None:
        """Return a definitive PaywallResult, or None if this strategy has
        no signal for the given input (so the detector moves to the next).

        Must not raise on malformed input — bad HTML or JSON is "no
        signal", not an error.
        """
