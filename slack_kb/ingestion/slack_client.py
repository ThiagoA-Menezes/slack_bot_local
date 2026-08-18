"""
Slack ingestion module.

Fetches messages (including thread replies) from all joined channels,
with cursor-based pagination and Slack API rate-limit backoff.

User IDs (e.g. U0123ABC) are resolved to display names via a cached
call to users.info so that author information is human-readable before
being passed downstream.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generator

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class SlackMessage:
    channel_id: str
    channel_name: str
    ts: str                    # Slack timestamp — unique message ID
    user: str                  # resolved display name (e.g. "Jane Smith")
    user_id: str               # raw Slack user ID (e.g. "U0123ABC")
    text: str
    datetime: datetime         # UTC datetime derived from the Slack ts float
    thread_ts: str | None = None
    replies: list["SlackMessage"] = field(default_factory=list)


def _ts_to_datetime(ts: str) -> datetime:
    """Convert a Slack timestamp string (e.g. '1700000000.123456') to UTC datetime."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


class SlackIngestionClient:
    """Wraps the Slack WebClient to iterate messages across all channels."""

    def __init__(self, token: str) -> None:
        self._client = WebClient(token=token)
        # Cache: user_id → display name
        self._user_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_channels(self) -> list[dict]:
        """Return all public + private channels the bot has joined."""
        channels: list[dict] = []
        cursor: str | None = None

        while True:
            resp = self._call(
                self._client.conversations_list,
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=200,
                **({"cursor": cursor} if cursor else {}),
            )
            channels.extend(resp["channels"])
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        logger.info("Found %d channels", len(channels))
        return channels

    def iter_channel_messages(
        self,
        channel_id: str,
        channel_name: str,
        oldest: str | None = None,
        latest: str | None = None,
        fetch_replies: bool = True,
    ) -> Generator[SlackMessage, None, None]:
        """
        Yield SlackMessage objects for every message in *channel_id*.

        Parameters
        ----------
        oldest:
            Only return messages after this Slack timestamp (exclusive).
            Pass the last-synced timestamp for incremental runs.
        latest:
            Only return messages before this Slack timestamp (inclusive).
        fetch_replies:
            Whether to recursively fetch threaded replies.
        """
        cursor: str | None = None

        while True:
            params: dict = dict(
                channel=channel_id,
                limit=200,
                **({"oldest": oldest} if oldest else {}),
                **({"latest": latest} if latest else {}),
                **({"cursor": cursor} if cursor else {}),
            )
            resp = self._call(self._client.conversations_history, **params)

            for raw in resp.get("messages", []):
                msg = self._parse_message(raw, channel_id, channel_name)
                if msg is None:
                    continue

                if fetch_replies and raw.get("reply_count", 0) > 0:
                    msg.replies = list(
                        self._fetch_replies(channel_id, channel_name, raw["ts"])
                    )

                yield msg

            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

    def resolve_user(self, user_id: str) -> str:
        """Return the display name for *user_id*, using a local cache."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        try:
            resp = self._call(self._client.users_info, user=user_id)
            profile = resp["user"].get("profile", {})
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or resp["user"].get("name")
                or user_id
            )
        except SlackApiError:
            name = user_id
        self._user_cache[user_id] = name
        return name

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_replies(
        self, channel_id: str, channel_name: str, thread_ts: str
    ) -> Generator[SlackMessage, None, None]:
        cursor: str | None = None
        while True:
            resp = self._call(
                self._client.conversations_replies,
                channel=channel_id,
                ts=thread_ts,
                limit=200,
                **({"cursor": cursor} if cursor else {}),
            )
            # First message is the parent — skip it
            for raw in resp.get("messages", [])[1:]:
                msg = self._parse_message(raw, channel_id, channel_name, thread_ts)
                if msg:
                    yield msg

            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

    def _parse_message(
        self, raw: dict, channel_id: str, channel_name: str, thread_ts: str | None = None
    ) -> SlackMessage | None:
        text = raw.get("text", "").strip()
        if not text or raw.get("subtype") in {"bot_message", "channel_join", "channel_leave"}:
            return None
        user_id = raw.get("user", "unknown")
        return SlackMessage(
            channel_id=channel_id,
            channel_name=channel_name,
            ts=raw["ts"],
            user=self.resolve_user(user_id) if user_id != "unknown" else "unknown",
            user_id=user_id,
            text=text,
            datetime=_ts_to_datetime(raw["ts"]),
            thread_ts=thread_ts or raw.get("thread_ts"),
        )

    @retry(
        retry=retry_if_exception_type(SlackApiError),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    def _call(self, method, **kwargs):
        try:
            return method(**kwargs)
        except SlackApiError as exc:
            if exc.response.get("error") == "ratelimited":
                retry_after = int(exc.response.headers.get("Retry-After", 10))
                logger.warning("Rate limited. Sleeping %ds", retry_after)
                time.sleep(retry_after)
            raise
