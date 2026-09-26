"""Summarizer — uses the LLM to write short German digests and, optionally,
longer wrap-ups of articles.

For each kept article, the summarizer reads:
  - the article title
  - the feed-provided summary (which may be empty, HTML-laden, or junk)

and produces a clean German summary plus a headline in the same voice
(the original title stays on the Article). The summarizer NEVER fetches article
bodies from the publisher for that — it works only with what the feed
provided. Wrap-ups (opt-in) are the exception: they fetch the full body.

The prompts, tool schemas, response models and the reading of the model's
answers all live in `llm/tasks/summarize.py` and `llm/tasks/wrap_up.py`.
What is left here is I/O and domain mapping: flattening the edition,
fetching and choosing article bodies, and writing the aligned results back
onto the edition.
"""

import asyncio

import click

from frankenbote.body_fetcher import fetch_bodies
from frankenbote.llm import LLMClient
from frankenbote.llm.tasks import SUMMARIZER_TASK, WRAP_UP_TASK, WrapUpItem
from frankenbote.models import CuratedArticle, Edition

# -------- Public API --------


def summarize_edition(edition: Edition, client: LLMClient) -> Edition:
    """Run the summarizer on every article in the edition.

    Returns a new Edition with ai_title and ai_summary populated on each
    CuratedArticle. Where the LLM judged the input too thin, the field stays
    None and the renderer falls back to the feed's own title / summary.
    Raises RuntimeError on persistent failure; the client retries once and
    saves debug context to disk.
    """
    flat: list[CuratedArticle] = [
        item for section in edition.sections for item in section.articles
    ]
    if not flat:
        return edition

    call_desc = (
        "Batches API, polling until done"
        if client.use_batch
        else "tool-use API call, ~60-120 seconds"
    )

    def announce(_attempt: int) -> None:
        click.echo(f"\nSummarizing {len(flat)} articles… ({call_desc})")

    results = client.run_task(SUMMARIZER_TASK, flat, on_attempt=announce).values

    # Build a new edition with ai_title and ai_summary populated.
    new_sections = []
    flat_idx = 0
    for section in edition.sections:
        new_articles = []
        for item in section.articles:
            result = results[flat_idx]
            new_articles.append(
                item.model_copy(
                    update={"ai_title": result.ai_title, "ai_summary": result.summary}
                )
            )
            flat_idx += 1
        new_sections.append(section.model_copy(update={"articles": new_articles}))

    return edition.model_copy(update={"sections": new_sections})


# ======================================================================
# Wrap-ups — a longer, multi-paragraph digest built from the full article
# body (fetched from the source URL), not just the feed snippet.
# ======================================================================


# -------- Wrap-up helpers (pure) --------


def _select_body(article: CuratedArticle, fetched: str | None) -> str | None:
    """Pick the text to wrap up: the fetched article body if available,
    otherwise the feed snippet, otherwise None when neither has substance.
    """
    if fetched and fetched.strip():
        return fetched
    snippet = article.article.summary
    if snippet and snippet.strip():
        return snippet
    return None


def _wrap_up_inputs(
    selected: list[tuple[int, int, CuratedArticle]],
    bodies: dict[str, str | None],
) -> tuple[list[tuple[int, int]], list[WrapUpItem]]:
    """Pair each article that has usable text with the text to wrap up.

    Returns the edition positions and the task inputs as two parallel
    lists, so the results can be mapped back afterwards. Articles with no
    usable text are dropped with a log line. Pure apart from the logging —
    designed to be unit-tested.
    """
    positions: list[tuple[int, int]] = []
    items: list[WrapUpItem] = []

    for s_idx, a_idx, item in selected:
        fetched = bodies.get(item.article.link)
        body = _select_body(item, fetched)
        if body is None:
            click.echo(f"  ⚠ Skipping {item.article.title[:60]}: no usable text")
            continue

        src = "fetched body" if fetched else "feed snippet (fetch failed)"
        click.echo(f"  • {item.article.title[:60]} ({src})")
        positions.append((s_idx, a_idx))
        items.append((item, body))

    return positions, items


# -------- Public API --------


def generate_wrap_ups(  # pragma: no cover
    edition: Edition,
    client: LLMClient,
) -> Edition:
    """Generate a longer wrap-up for selected articles in the edition.

    Currently every article gets a wrap-up (see the selection filter
    below). For each, the source article body is fetched from its URL
    (falling back to the feed snippet when the fetch fails) and the LLM
    writes a 2-3 paragraph German digest into the article's wrap_up field.

    Batch or single calls are the client's business: the task renders and
    reads one article the same way either way. Per-article failures come
    back as notes and are logged; they never abort the run.
    """
    # Selection filter — uncomment is_lead to use lead articles only
    selected: list[tuple[int, int, CuratedArticle]] = [
        (s_idx, a_idx, item)
        for s_idx, section in enumerate(edition.sections)
        for a_idx, item in enumerate(section.articles)
        # if item.is_lead
    ]
    if not selected:
        return edition

    # Fetch every source body in parallel up front.
    click.echo(f"\nFetching {len(selected)} article bodies for wrap-ups…")
    bodies = asyncio.run(fetch_bodies([item.article.link for _, _, item in selected]))

    positions, items = _wrap_up_inputs(selected, bodies)
    if not items:
        return edition

    call_desc = (
        "Batches API, polling until done"
        if client.use_batch
        else "one tool-use call each"
    )

    def announce(attempt: int) -> None:
        click.echo(
            f"\nGenerating {len(items)} wrap-up(s) (attempt {attempt})… ({call_desc})"
        )

    outcome = client.run_task(WRAP_UP_TASK, items, on_attempt=announce)
    for note in outcome.notes:
        title = (
            items[note.index][0].article.title[:60] if note.index is not None else "?"
        )
        click.echo(f"  ⚠ Wrap-up for {title}: {note.reason}", err=True)

    results = {
        position: wrap_up
        for position, wrap_up in zip(positions, outcome.values, strict=True)
        if wrap_up
    }

    # Rebuild the edition with wrap_up populated where we have one.
    new_sections = []
    for s_idx, section in enumerate(edition.sections):
        new_articles = []
        for a_idx, item in enumerate(section.articles):
            wrap_up = results.get((s_idx, a_idx))
            if wrap_up is not None:
                item = item.model_copy(update={"wrap_up": wrap_up})
            new_articles.append(item)
        new_sections.append(section.model_copy(update={"articles": new_articles}))

    return edition.model_copy(update={"sections": new_sections})
