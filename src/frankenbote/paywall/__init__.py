"""Paywall detection — decides whether a fetched article page is paywalled.

Different publishers expose different signals (SZ ships schema.org
structured metadata, FAZ does not), so detection runs an ordered chain of
strategies and stops at the first definitive answer. Callers use the single
entry point ``is_paywalled``; new strategies are registered in
``detector.py`` without touching any call site.
"""

from frankenbote.paywall.base import PaywallResult
from frankenbote.paywall.detector import is_paywalled

__all__ = ["PaywallResult", "is_paywalled"]
