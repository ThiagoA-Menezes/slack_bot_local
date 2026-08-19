"""
CLI entry point for the Slack Knowledge Base tool.

Commands
--------
  ingest    Pull messages from Slack and build the knowledge base.
  ask       Query the knowledge base with a natural language question.
  search    Semantic search without LLM generation.
  channels  Interactive channel browser — pick channels to ingest or query.
"""
from __future__ import annotations

import logging
import os
import sys

import click
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


@click.group()
def cli():
    """Slack Knowledge Base — ingest, search, and query your Slack history."""


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--channels", "-c",
    multiple=True,
    help="Channel names to ingest (repeat for multiple). Omit for ALL channels.",
)
@click.option(
    "--full", "full_sync",
    is_flag=True,
    default=False,
    help="Force a full re-sync ignoring previously stored timestamps.",
)
@click.option(
    "--batch-size", "-b",
    default=int(os.environ.get("BATCH_SIZE", 50)),
    show_default=True,
    help="Number of messages to accumulate per summary chunk.",
)
def ingest(channels, full_sync, batch_size):
    """Ingest Slack messages and store summaries in the knowledge base."""
    from slack_kb.ingestion.slack_client import SlackIngestionClient
    from slack_kb.pipeline import run_pipeline
    from slack_kb.processing.summarizer import OllamaSummarizer
    from slack_kb.storage.knowledge_store import KnowledgeStore

    _require_env("SLACK_BOT_TOKEN")

    slack = SlackIngestionClient(token=os.environ["SLACK_BOT_TOKEN"])
    summarizer = OllamaSummarizer.from_env()
    store = KnowledgeStore.from_env()

    run_pipeline(
        slack_client=slack,
        summarizer=summarizer,
        store=store,
        channel_filter=list(channels) or None,
        batch_size=batch_size,
        incremental=not full_sync,
    )
    click.echo("✓ Ingestion complete.")


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("question")
@click.option(
    "--channel", "-c",
    default=None,
    help="Restrict search to a specific channel name.",
)
def ask(question, channel):
    """Ask a question and get an answer grounded in the knowledge base."""
    from slack_kb.query.qa import KnowledgeBaseQA
    from slack_kb.storage.knowledge_store import KnowledgeStore
    from slack_kb.storage.response_cache import ResponseCache

    store = KnowledgeStore.from_env()
    cache = ResponseCache.from_env()
    qa = KnowledgeBaseQA.from_env(store, cache)
    result = qa.ask(question, channel_name=channel)

    if result["cache_hit"]:
        click.echo(
            "\n" + click.style("⚡ Resposta em cache", fg="yellow", bold=True)
            + f"  (pergunta similar feita {result['hit_count']}x)"
        )
        click.echo(click.style(f"  Pergunta original: ", dim=True) + result["similar_query"])
    click.echo("\n" + click.style("Answer:", bold=True))
    click.echo(result["answer"])

    if result["sources"]:
        click.echo("\n" + click.style("Sources:", bold=True))
        for src in result["sources"]:
            authors = ", ".join(src.get("authors", [])) or "unknown"
            click.echo(
                f"  • #{src['channel_name']}  {src.get('oldest_dt','')} – {src.get('newest_dt','')}  ({authors})"
            )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--top", "-n", default=5, show_default=True, help="Number of results.")
@click.option("--channel", "-c", default=None, help="Filter by channel name.")
def search(query, top, channel):
    """Semantic search over knowledge base summaries (no LLM generation)."""
    from slack_kb.storage.knowledge_store import KnowledgeStore

    store = KnowledgeStore.from_env()
    hits = store.search(query, n_results=top, channel_name=channel)

    if not hits:
        click.echo("No results found.")
        return

    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        authors = ", ".join(meta.get("authors", [])) or "unknown"
        click.echo(
            f"\n[{i}] #{meta['channel_name']}  score={hit['score']:.4f}\n"
            f"    📅 {meta.get('oldest_dt','')} – {meta.get('newest_dt','')}\n"
            f"    👥 {authors}"
        )
        click.echo(hit["summary"])
        click.echo("─" * 60)


# ---------------------------------------------------------------------------
# channels  (interactive browser)
# ---------------------------------------------------------------------------

@cli.command("top-queries")
@click.option("--type", "query_type", default=None,
              type=click.Choice(["ask_question", "get_channel_summary"]),
              help="Filter by query type (default: all).")
@click.option("--top", "-n", default=20, show_default=True,
              help="Number of entries to display.")
def top_queries(query_type, top):
    """Show the most-queried topics — useful for opportunity mapping."""
    from slack_kb.storage.response_cache import ResponseCache

    cache = ResponseCache.from_env()
    rows = cache.top_queries(query_type=query_type, size=top)

    if not rows:
        click.echo("No cached queries yet.")
        return

    click.echo(
        "\n" + click.style(f"{'#':>4}  {'Hits':>5}  {'Type':<20}  {'Channel':<12}  Query", bold=True)
    )
    click.echo("─" * 80)
    for row in rows:
        ch = row["channel"] or "—"
        click.echo(
            f"{row['rank']:>4}  {row['hit_count']:>5}  {row['query_type']:<20}  {ch:<12}  {row['query_text'][:60]}"
        )


@cli.command()
def channels():
    """
    Interactive channel browser.

    Lists every channel that has been ingested into the knowledge base
    and lets you pick one or more to:

      • View their stored summaries
      • Run a semantic search scoped to the selected channels
      • Re-ingest (incremental) the selected channels
    """
    import questionary
    from slack_kb.storage.knowledge_store import KnowledgeStore

    store = KnowledgeStore.from_env()
    available = store.list_channels()

    if not available:
        click.echo(
            "No channels in the knowledge base yet.\n"
            "Run:  python cli.py ingest"
        )
        return

    # ── Step 1: channel multi-select ──────────────────────────────────
    selected: list[str] = questionary.checkbox(
        "Select channels  (Space to toggle, Enter to confirm):",
        choices=available,
    ).ask()

    if not selected:
        click.echo("No channels selected — nothing to do.")
        return

    # ── Step 2: action menu ───────────────────────────────────────────
    action: str = questionary.select(
        f"What do you want to do with {len(selected)} channel(s)?",
        choices=[
            questionary.Choice("📖  View summaries", value="view"),
            questionary.Choice("🔍  Semantic search", value="search"),
            questionary.Choice("🔄  Re-ingest (incremental)", value="ingest"),
            questionary.Choice("🔄  Re-ingest (full re-sync)", value="ingest_full"),
        ],
    ).ask()

    if action is None:
        return

    # ── view ──────────────────────────────────────────────────────────
    if action == "view":
        limit: int = questionary.text(
            "How many recent entries per channel?", default="5"
        ).ask()
        limit_int = int(limit) if limit and limit.isdigit() else 5

        for ch in selected:
            click.echo(f"\n{'═' * 60}")
            click.echo(click.style(f"  #{ch}", bold=True))
            click.echo('═' * 60)
            entries = store.get_channel_summary(ch, limit=limit_int)
            if not entries:
                click.echo("  (no entries)")
                continue
            for i, e in enumerate(entries, 1):
                authors = ", ".join(e.get("authors", [])) or "unknown"
                click.echo(f"\n  [{i}] 📅 {e.get('oldest_dt','')} – {e.get('newest_dt','')}")
                click.echo(f"       👥 {authors}")
                click.echo(f"  {e['summary']}")
                click.echo("  " + "─" * 56)

    # ── semantic search ───────────────────────────────────────────────
    elif action == "search":
        query: str = questionary.text("Enter your search query:").ask()
        if not query:
            return
        top_raw: str = questionary.text("How many results per channel?", default="3").ask()
        top_int = int(top_raw) if top_raw and top_raw.isdigit() else 3

        for ch in selected:
            click.echo(f"\n{'═' * 60}")
            click.echo(click.style(f"  #{ch}", bold=True))
            click.echo('═' * 60)
            hits = store.search(query, n_results=top_int, channel_name=ch)
            if not hits:
                click.echo("  No results.")
                continue
            for i, hit in enumerate(hits, 1):
                meta = hit["metadata"]
                authors = ", ".join(meta.get("authors", [])) or "unknown"
                click.echo(
                    f"\n  [{i}] score={hit['score']:.4f}  "
                    f"📅 {meta.get('oldest_dt','')} – {meta.get('newest_dt','')}\n"
                    f"       👥 {authors}"
                )
                click.echo(f"  {hit['summary']}")
                click.echo("  " + "─" * 56)

    # ── ingest ────────────────────────────────────────────────────────
    elif action in ("ingest", "ingest_full"):
        from slack_kb.ingestion.slack_client import SlackIngestionClient
        from slack_kb.pipeline import run_pipeline
        from slack_kb.processing.summarizer import OllamaSummarizer

        _require_env("SLACK_BOT_TOKEN")

        full_sync = action == "ingest_full"
        click.echo(
            f"\nIngesting {len(selected)} channel(s): {', '.join('#' + c for c in selected)}"
            + (" [full re-sync]" if full_sync else " [incremental]")
        )

        slack = SlackIngestionClient(token=os.environ["SLACK_BOT_TOKEN"])
        summarizer = OllamaSummarizer.from_env()
        run_pipeline(
            slack_client=slack,
            summarizer=summarizer,
            store=store,
            channel_filter=selected,
            incremental=not full_sync,
        )
        click.echo("✓ Ingestion complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(*keys: str) -> None:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        click.echo(
            f"Error: missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in the values.",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    cli()
