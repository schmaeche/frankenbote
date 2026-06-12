"""Paywall detection via schema.org structured metadata (JSON-LD)."""

import json
from html.parser import HTMLParser
from typing import Any, Iterator

from frankenbote.paywall.base import PaywallResult, PaywallStrategy

# schema.org allows isAccessibleForFree as a real boolean or as a string;
# publishers use both spellings in the wild.
_FREE_STRINGS = {"true", "yes"}
_PAYWALLED_STRINGS = {"false", "no"}


class _LdJsonExtractor(HTMLParser):
    """Collects the text content of <script type="application/ld+json"> tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._chunks: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            script_type = (dict(attrs).get("type") or "").strip().lower()
            if script_type == "application/ld+json":
                self._chunks = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._chunks is not None:
            self.blocks.append("".join(self._chunks))
            self._chunks = None

    def handle_data(self, data: str) -> None:
        if self._chunks is not None:
            self._chunks.append(data)


def _ld_json_blocks(html: str) -> list[str]:
    extractor = _LdJsonExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:  # malformed HTML is "no signal", never an error
        pass
    return extractor.blocks


def _iter_nodes(data: Any) -> Iterator[dict]:
    """Yield every JSON-LD object, descending into arrays and the
    ``hasPart`` / ``@graph`` nestings publishers commonly use."""
    if isinstance(data, list):
        for item in data:
            yield from _iter_nodes(item)
    elif isinstance(data, dict):
        yield data
        for key in ("hasPart", "@graph"):
            if key in data:
                yield from _iter_nodes(data[key])


def _interpret(value: Any) -> bool | None:
    """Map an ``isAccessibleForFree`` value to paywalled True/False, or
    None when the value is absent or unrecognised."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _PAYWALLED_STRINGS:
            return True
        if normalized in _FREE_STRINGS:
            return False
    return None


class StructuredMetadataStrategy(PaywallStrategy):
    """Reads ``isAccessibleForFree`` from schema.org JSON-LD.

    Decides SZ articles, which carry the field; FAZ pages have JSON-LD
    without it, so this strategy yields no signal there.
    """

    name = "structured_metadata"

    def detect(self, html: str, url: str | None) -> PaywallResult | None:
        for block in _ld_json_blocks(html):
            try:
                data = json.loads(block)
            except ValueError:
                continue
            for node in _iter_nodes(data):
                verdict = _interpret(node.get("isAccessibleForFree"))
                if verdict is not None:
                    return PaywallResult(paywalled=verdict, strategy=self.name)
        return None
