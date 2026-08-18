"""
RAG (Retrieval-Augmented Generation) query layer.

Combines OpenSearch k-NN vector search with a local Ollama generation
call to answer questions grounded in the stored knowledge base.
"""
from __future__ import annotations

import logging
import os

import ollama

from slack_kb.storage.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

_QA_PROMPT_TEMPLATE = """\
You are a helpful knowledge base assistant. Answer the question below \
using ONLY the context provided. If the answer is not in the context, \
say "I don't have enough information to answer that."

Context (summaries of Slack conversations):
{context}

Question: {question}

Answer:"""


class KnowledgeBaseQA:
    def __init__(
        self,
        store: KnowledgeStore,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        n_context_entries: int = 5,
    ) -> None:
        self._store = store
        self._n = n_context_entries
        self._model = model
        self._client = ollama.Client(host=host)

    def ask(self, question: str, channel_name: str | None = None) -> dict:
        """
        Answer *question* using RAG over the knowledge base.

        Returns a dict with keys:
          - answer: the generated answer string
          - sources: list of metadata dicts for the retrieved context entries
        """
        hits = self._store.search(question, n_results=self._n, channel_name=channel_name)
        if not hits:
            return {"answer": "The knowledge base is empty. Run the ingest pipeline first.", "sources": []}

        context = "\n\n---\n\n".join(
            f"[#{h['metadata']['channel_name']} | {h['metadata'].get('oldest_dt','')} – {h['metadata'].get('newest_dt','')} | by {', '.join(h['metadata'].get('authors', []))}]\n{h['summary']}"
            for h in hits
        )
        prompt = _QA_PROMPT_TEMPLATE.format(context=context, question=question)
        resp = self._client.generate(
            model=self._model,
            prompt=prompt,
            options={"temperature": 0.1},
        )
        answer = resp.response.strip()

        return {
            "answer": answer,
            "sources": [h["metadata"] for h in hits],
        }

    @classmethod
    def from_env(cls, store: KnowledgeStore) -> "KnowledgeBaseQA":
        return cls(
            store=store,
            model=os.environ.get("OLLAMA_GEN_MODEL", "llama3.2"),
            host=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            n_context_entries=int(os.environ.get("N_CONTEXT_ENTRIES", 5)),
        )
