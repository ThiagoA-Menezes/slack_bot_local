"""
Ollama summarization module.

Calls a locally-running Ollama instance to produce a structured summary
of a message chunk, extracting topics, decisions, and action items.

Default model: llama3.2  (pull with: ollama pull llama3.2)
Any chat model available in your Ollama installation can be used by
setting OLLAMA_GEN_MODEL in the environment.
"""
from __future__ import annotations

import logging
import os

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from slack_kb.processing.chunker import MessageChunk

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT_TEMPLATE = """\
You are a technical knowledge base assistant. Read the following Slack \
conversation excerpt and produce a concise, structured summary.

Format your response exactly as:
**Topics:** <comma-separated list of main topics discussed>
**Key Points:** <bullet list of important information, decisions, or answers>
**Action Items:** <bullet list of tasks or follow-ups, or "None">

--- Conversation ---
{conversation}
--- End ---

Structured Summary:"""


class OllamaSummarizer:
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._client = ollama.Client(host=host)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
    )
    def summarize(self, chunk: MessageChunk) -> str:
        """Return a structured summary string for *chunk*."""
        prompt = _SUMMARY_PROMPT_TEMPLATE.format(conversation=chunk.text)
        logger.debug("Summarizing chunk from #%s (%s)", chunk.channel_name, chunk.oldest_ts)
        resp = self._client.generate(
            model=self._model,
            prompt=prompt,
            options={"temperature": 0.2, "stop": ["---"]},
        )
        return resp.response.strip()

    @classmethod
    def from_env(cls) -> "OllamaSummarizer":
        return cls(
            model=os.environ.get("OLLAMA_GEN_MODEL", "llama3.2"),
            host=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        )
