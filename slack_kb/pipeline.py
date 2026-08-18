"""
Ingestion pipeline.

Ties together: Slack → chunker → watsonx summarizer → OpenSearch.
Supports both full and incremental (timestamp-based) runs.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from slack_kb.ingestion.slack_client import SlackIngestionClient, SlackMessage
from slack_kb.processing.chunker import MessageChunk, chunk_messages
from slack_kb.processing.summarizer import OllamaSummarizer
from slack_kb.storage.knowledge_store import KnowledgeEntry, KnowledgeStore

logger = logging.getLogger(__name__)


def run_pipeline(
    slack_client: SlackIngestionClient,
    summarizer: OllamaSummarizer,
    store: KnowledgeStore,
    channel_filter: list[str] | None = None,
    batch_size: int = 50,
    incremental: bool = True,
) -> None:
    """
    Main ingestion pipeline.

    Parameters
    ----------
    channel_filter:
        If provided, only process channels whose names are in this list.
    batch_size:
        Maximum number of messages to accumulate before creating a chunk.
    incremental:
        If True, only fetch messages newer than the last stored timestamp
        for each channel (uses ``KnowledgeStore.get_newest_ts``).
    """
    channels = slack_client.list_channels()

    if channel_filter:
        channels = [c for c in channels if c["name"] in channel_filter]
        logger.info("Filtered to %d channel(s): %s", len(channels), channel_filter)

    for channel in channels:
        cid = channel["id"]
        cname = channel["name"]

        oldest: str | None = None
        if incremental:
            oldest = store.get_newest_ts(cid)
            if oldest:
                logger.info("Incremental sync for #%s from ts=%s", cname, oldest)
            else:
                logger.info("Full sync for #%s (no prior data)", cname)

        try:
            messages: list[SlackMessage] = list(
                slack_client.iter_channel_messages(
                    channel_id=cid,
                    channel_name=cname,
                    oldest=oldest,
                )
            )
        except Exception as exc:
            # Unwrap tenacity RetryError to show the real Slack error
            cause = getattr(exc, "last_attempt", None)
            real = cause.exception() if cause else exc
            logger.warning("#%s: skipping — %s", cname, real)
            continue

        if not messages:
            logger.info("#%s: no new messages", cname)
            continue

        logger.info("#%s: %d messages to process", cname, len(messages))
        chunks = chunk_messages(messages, cid, cname, max_tokens=batch_size * 40)
        entries: list[KnowledgeEntry] = []

        for i, chunk in enumerate(chunks, 1):
            logger.info(
                "#%s chunk %d/%d — summarizing…", cname, i, len(chunks)
            )
            summary = summarizer.summarize(chunk)
            entries.append(
                KnowledgeEntry(
                    channel_id=cid,
                    channel_name=cname,
                    oldest_ts=chunk.oldest_ts,
                    newest_ts=chunk.newest_ts,
                    oldest_dt=chunk.oldest_dt,
                    newest_dt=chunk.newest_dt,
                    authors=chunk.authors,
                    summary=summary,
                )
            )

        store.upsert_many(entries)
        logger.info("#%s: stored %d knowledge entries", cname, len(entries))
