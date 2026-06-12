"""Tests for frankenbote.paywall — strategy-based paywall detection."""

import json

import pytest

from frankenbote.paywall import PaywallResult, is_paywalled
from frankenbote.paywall import detector
from frankenbote.paywall.base import PaywallStrategy
from frankenbote.paywall.strategies import (
    ContentLengthStrategy,
    StructuredMetadataStrategy,
    content_length,
)

URL = "https://www.sueddeutsche.de/bayern/artikel-1"
FAZ_URL = "https://www.faz.net/aktuell/finanzen/artikel-1"

# Long enough that the content-length strategy stays silent — pages built
# with _page() exercise the metadata signal, not the length heuristic.
_LONG_BODY = "<p>" + (
    "Die Lage bleibt nach Einschätzung von Beobachtern angespannt, "
    "auch wenn sich einzelne Indikatoren zuletzt leicht verbessert haben. "
) * 30 + "</p>"

# FAZ-style paywall page: no usable metadata, and extraction yields only
# the teaser plus the subscription offer.
_TEASER_PAGE = (
    '<!DOCTYPE html><html lang="de"><head><title>Test</title></head>'
    "<body><article>"
    "<h1>SpaceX-Aktie: Ein Börsenprospekt wie im Science-Fiction-Film</h1>"
    "<p>Elon Musk will zum Mars. Einen Zwischenstopp aber möchte er mit seinem "
    "Unternehmen SpaceX am Freitag noch einlegen: das digitale Parkett der New "
    "Yorker Börse Nasdaq. Das nötige Kapital für die Erschließung eines neuen "
    "Planeten steuert selbst der reichste Mensch der Welt nicht einfach selbst "
    "bei, vielmehr sollen alle daran teilhaben können.</p>"
    '<div class="paywall"><p>Zugang zu allen FAZ+ Beiträgen '
    "(Originalpreis: 13,80 €) jetzt nur 0,99 €</p>"
    "<p>- Mit einem Klick online kündbar</p></div>"
    "</article></body></html>"
)


def _page(ld_json: str | None) -> str:
    script = (
        f'<script type="application/ld+json">{ld_json}</script>'
        if ld_json is not None
        else ""
    )
    return (
        '<!DOCTYPE html><html lang="de"><head><title>Test</title>'
        f"{script}</head>"
        f"<body><article><h1>Eine Schlagzeile</h1>{_LONG_BODY}</article></body></html>"
    )


def _sz_ld_json(accessible: object) -> str:
    return json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "NewsArticle",
            "headline": "Eine Schlagzeile",
            "isAccessibleForFree": accessible,
        }
    )


# FAZ-style: JSON-LD exists, but isAccessibleForFree does not.
_FAZ_LD_JSON = json.dumps(
    {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": "Eine Schlagzeile",
        "author": {"@type": "Person", "name": "A. Redakteur"},
    }
)


# ── StructuredMetadataStrategy ───────────────────────────────────────────────

class TestStructuredMetadataStrategy:
    strategy = StructuredMetadataStrategy()

    def test_accessible_false_means_paywalled(self):
        result = self.strategy.detect(_page(_sz_ld_json(False)), URL)
        assert result == PaywallResult(paywalled=True, strategy="structured_metadata")

    def test_accessible_true_means_free(self):
        result = self.strategy.detect(_page(_sz_ld_json(True)), URL)
        assert result == PaywallResult(paywalled=False, strategy="structured_metadata")

    @pytest.mark.parametrize("value", ["False", "false", "no"])
    def test_string_spellings_paywalled(self, value):
        result = self.strategy.detect(_page(_sz_ld_json(value)), URL)
        assert result is not None and result.paywalled is True

    @pytest.mark.parametrize("value", ["True", "true", "yes"])
    def test_string_spellings_free(self, value):
        result = self.strategy.detect(_page(_sz_ld_json(value)), URL)
        assert result is not None and result.paywalled is False

    def test_field_absent_is_no_signal(self):
        assert self.strategy.detect(_page(_FAZ_LD_JSON), URL) is None

    def test_no_ld_json_is_no_signal(self):
        assert self.strategy.detect(_page(None), URL) is None

    def test_malformed_json_is_no_signal(self):
        assert self.strategy.detect(_page("{not: json,,,"), URL) is None

    def test_malformed_html_is_no_signal(self):
        assert self.strategy.detect("<![unknown[ <p>corrupt</p <x><x><x", URL) is None

    def test_non_json_script_then_valid_block_still_decides(self):
        html = _page("var foo = 1;").replace(
            "</head>",
            f'<script type="application/ld+json">{_sz_ld_json(False)}</script></head>',
        )
        result = self.strategy.detect(html, URL)
        assert result is not None and result.paywalled is True

    def test_array_of_objects(self):
        blocks = json.dumps(
            [
                {"@type": "BreadcrumbList"},
                {"@type": "NewsArticle", "isAccessibleForFree": False},
            ]
        )
        result = self.strategy.detect(_page(blocks), URL)
        assert result is not None and result.paywalled is True

    def test_has_part_nesting(self):
        nested = json.dumps(
            {
                "@type": "NewsArticle",
                "hasPart": {
                    "@type": "WebPageElement",
                    "isAccessibleForFree": False,
                    "cssSelector": ".paywall",
                },
            }
        )
        result = self.strategy.detect(_page(nested), URL)
        assert result is not None and result.paywalled is True

    def test_unrecognised_value_is_no_signal(self):
        assert self.strategy.detect(_page(_sz_ld_json("vielleicht")), URL) is None

    def test_empty_html_is_no_signal(self):
        assert self.strategy.detect("", URL) is None


# ── ContentLengthStrategy ────────────────────────────────────────────────────

class TestContentLengthStrategy:
    strategy = ContentLengthStrategy()

    def test_teaser_page_is_paywalled(self):
        result = self.strategy.detect(_TEASER_PAGE, FAZ_URL)
        assert result == PaywallResult(paywalled=True, strategy="content_length")

    def test_full_article_is_no_signal(self):
        assert self.strategy.detect(_page(None), FAZ_URL) is None

    def test_empty_html_is_no_signal(self):
        assert self.strategy.detect("", FAZ_URL) is None

    def test_malformed_html_is_no_signal(self):
        assert self.strategy.detect("<![unknown[ <p>corrupt</p <x><x><x", FAZ_URL) is None

    def test_extraction_error_is_no_signal(self, monkeypatch):
        def _raise(html, **settings):
            raise RuntimeError("extraction failed")

        monkeypatch.setattr(content_length.trafilatura, "extract", _raise)
        assert self.strategy.detect(_TEASER_PAGE, FAZ_URL) is None

    def test_does_not_read_metadata(self):
        # Free per JSON-LD but teaser-sized: this strategy flags it; the
        # detector ordering is what protects labelled short articles.
        html = _TEASER_PAGE.replace(
            "</head>",
            f'<script type="application/ld+json">{_sz_ld_json(True)}</script></head>',
        )
        result = self.strategy.detect(html, FAZ_URL)
        assert result is not None and result.paywalled is True


# ── is_paywalled (detector) ──────────────────────────────────────────────────

class _StubStrategy(PaywallStrategy):
    def __init__(self, name: str, result: PaywallResult | None):
        self.name = name
        self._result = result
        self.calls = 0

    def detect(self, html: str, url: str | None) -> PaywallResult | None:
        self.calls += 1
        return self._result


class TestIsPaywalled:
    def test_sz_paywalled_article(self):
        result = is_paywalled(_page(_sz_ld_json(False)), URL)
        assert result == PaywallResult(paywalled=True, strategy="structured_metadata")

    def test_free_article(self):
        result = is_paywalled(_page(_sz_ld_json(True)), URL)
        assert result is not None and result.paywalled is False

    def test_faz_full_article_resolves_to_unknown(self):
        # No isAccessibleForFree and a full-length body: no strategy decides.
        assert is_paywalled(_page(_FAZ_LD_JSON), URL) is None

    def test_faz_teaser_resolves_to_paywalled_by_length(self):
        result = is_paywalled(_TEASER_PAGE, FAZ_URL)
        assert result == PaywallResult(paywalled=True, strategy="content_length")

    def test_metadata_free_wins_over_short_body(self):
        # A labelled-free page is free even when teaser-sized — structured
        # metadata runs before the length heuristic.
        html = _TEASER_PAGE.replace(
            "</head>",
            f'<script type="application/ld+json">{_sz_ld_json(True)}</script></head>',
        )
        result = is_paywalled(html, FAZ_URL)
        assert result == PaywallResult(paywalled=False, strategy="structured_metadata")

    def test_url_is_optional(self):
        assert is_paywalled(_page(_FAZ_LD_JSON)) is None

    def test_first_definitive_result_wins(self, monkeypatch):
        undecided = _StubStrategy("undecided", None)
        decided = _StubStrategy("decided", PaywallResult(True, "decided"))
        never_reached = _StubStrategy("never", PaywallResult(False, "never"))
        monkeypatch.setattr(
            detector, "_STRATEGIES", (undecided, decided, never_reached)
        )
        result = is_paywalled("<html></html>", URL)
        assert result == PaywallResult(paywalled=True, strategy="decided")
        assert undecided.calls == 1
        assert decided.calls == 1
        assert never_reached.calls == 0

    def test_all_strategies_undecided_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            detector,
            "_STRATEGIES",
            (_StubStrategy("a", None), _StubStrategy("b", None)),
        )
        assert is_paywalled("<html></html>", URL) is None
