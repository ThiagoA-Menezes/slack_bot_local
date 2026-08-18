"""
Message chunker / preprocessor.

Groups raw SlackMessages into text chunks that fit within the LLM's
context window, preserving channel, author, and datetime context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from slack_kb.ingestion.slack_client import SlackMessage

# Encoder used only for token counting; model does not have to match watsonx
_ENC = tiktoken.get_encoding("cl100k_base")

# Leave headroom for the summarization prompt itself
MAX_CHUNK_TOKENS = 2_000

_DT_FMT = "%Y-%m-%d %H:%M UTC"


@dataclass
class MessageChunk:
    channel_id: str
    channel_name: str
    # Slack timestamps of the first and last message in this chunk
    oldest_ts: str
    newest_ts: str
    # ISO date strings for the first and last message
    oldest_dt: str
    newest_dt: str
    # Sorted unique list of author display names in this chunk
    authors: list[str]
    text: str          # Human-readable block ready to be fed to the LLM


def _clean(text: str) -> str:
    """Strip Slack formatting noise that adds no semantic value."""
    # Preserve user mentions with the resolved name already in the text
    text = re.sub(r"<@[A-Z0-9]+>", "@user", text)
    text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", text)
    text = re.sub(r"<(https?://[^|>]+)(?:\|[^>]*)?>", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text


def _token_count(text: str) -> int:
    return len(_ENC.encode(text))


def build_message_text(msg: SlackMessage) -> str:
    """
    Format a single message (and its replies) as a readable block.

    Each line carries the author's display name and a human-readable
    UTC datetime so the LLM can reference who said what and when.
    """
    dt_str = msg.datetime.strftime(_DT_FMT)
    lines = [f"[{dt_str}] {msg.user}: {_clean(msg.text)}"]
    for reply in msg.replies:
        reply_dt = reply.datetime.strftime(_DT_FMT)
        lines.append(f"  ↳ [{reply_dt}] {reply.user}: {_clean(reply.text)}")
    return "\n".join(lines)


def chunk_messages(
    messages: list[SlackMessage],
    channel_id: str,
    channel_name: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[MessageChunk]:
    """
    Split *messages* into token-bounded chunks.

    Each chunk is prefixed with a header so the LLM knows its context
    and carries the full list of participating authors.
    """
    chunks: list[MessageChunk] = []
    current_lines: list[str] = []
    current_tokens = 0
    oldest_ts: str | None = None
    newest_ts: str | None = None
    oldest_dt: str | None = None
    newest_dt: str | None = None
    author_set: set[str] = set()

    def flush():
        nonlocal current_lines, current_tokens, oldest_ts, newest_ts, oldest_dt, newest_dt, author_set
        if not current_lines:
            return
        header = f"=== Channel: #{channel_name} | {oldest_dt} – {newest_dt} ===\n"
        body = "\n".join(current_lines)
        chunks.append(
            MessageChunk(
                channel_id=channel_id,
                channel_name=channel_name,
                oldest_ts=oldest_ts,
                newest_ts=newest_ts,
                oldest_dt=oldest_dt,
                newest_dt=newest_dt,
                authors=sorted(author_set),
                text=header + body,
            )
        )
        current_lines = []
        current_tokens = 0
        oldest_ts = None
        newest_ts = None
        oldest_dt = None
        newest_dt = None
        author_set = set()

    for msg in messages:
        block = build_message_text(msg)
        block_tokens = _token_count(block)

        if current_tokens + block_tokens > max_tokens:
            flush()

        current_lines.append(block)
        current_tokens += block_tokens
        if oldest_ts is None:
            oldest_ts = msg.ts
            oldest_dt = msg.datetime.strftime(_DT_FMT)
        newest_ts = msg.ts
        newest_dt = msg.datetime.strftime(_DT_FMT)

        # Collect author from message and all its replies
        author_set.add(msg.user)
        for reply in msg.replies:
            author_set.add(reply.user)

    flush()
    return chunks
