"""Paywall detection via extracted-content length.

A paywalled page still renders a teaser — headline, the first paragraphs,
and the subscription offer — so boilerplate removal leaves conspicuously
little text. Observed on FAZ: a paywalled article page extracts to roughly
800 characters of teaser and offer copy ("Zugang zu allen FAZ+ Beiträgen
…"), while real article bodies land well above 2000.
"""

import trafilatura

from frankenbote.body_fetcher import EXTRACT_SETTINGS
from frankenbote.paywall.base import PaywallResult, PaywallStrategy

# Extractions below this are teaser-sized: above the ~800 characters an FAZ
# paywall teaser yields, below the length of real article bodies. Short free
# pieces (tickers, briefs) can misfire, but the structured-metadata strategy
# runs first, so publishers that label their articles are decided before
# length is consulted.
MAX_TEASER_CHARS = 1000


class ContentLengthStrategy(PaywallStrategy):
    """Flags pages whose extracted main text is teaser-sized.

    Decides FAZ articles, whose JSON-LD lacks ``isAccessibleForFree`` but
    whose paywalled pages extract to a short teaser. Only ever returns a
    paywalled=True verdict: a long body does not prove the page is free,
    and a failed extraction proves nothing, so both yield None and leave
    the decision to later strategies.
    """

    name = "content_length"

    def detect(self, html: str, url: str | None) -> PaywallResult | None:
        try:
            # Same settings as fetch_body: this strategy predicts whether
            # the body fetch would yield a real article, so both must
            # measure the same extraction.
            text = trafilatura.extract(html, **EXTRACT_SETTINGS)
        except Exception:
            return None
        if text is None or not text.strip():
            return None
        if len(text.strip()) < MAX_TEASER_CHARS:
            return PaywallResult(paywalled=True, strategy=self.name)
        return None
