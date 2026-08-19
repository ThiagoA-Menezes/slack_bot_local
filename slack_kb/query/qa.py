"""
RAG (Retrieval-Augmented Generation) query layer with semantic response cache.

Flow
----
1. Embed the user question with Ollama.
2. Look up the response_cache index for a semantically similar previous answer
   (cosine similarity >= CACHE_SIMILARITY_THRESHOLD).
   → HIT:  return the cached answer immediately (zero LLM tokens spent).
           The hit_count for the entry is incremented automatically.
   → MISS: run full RAG (k-NN search over slack_knowledge + Ollama generation),
           store the result in the cache, then return.
"""
from __future__ import annotations

import logging
import os

import ollama

from slack_kb.storage.knowledge_store import KnowledgeStore
from slack_kb.storage.response_cache import CacheHit, ResponseCache

logger = logging.getLogger(__name__)

_QUERY_TYPE = "ask_question"

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
        cache: ResponseCache,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        n_context_entries: int = 5,
    ) -> None:
        self._store = store
        self._cache = cache
        self._n = n_context_entries
        self._model = model
        self._client = ollama.Client(host=host)

    def ask(self, question: str, channel_name: str | None = None) -> dict:
        """
        Answer *question* using the cache first, then RAG if needed.

        Returns a dict with keys:
          - answer:      the answer string
          - sources:     list of metadata dicts for the context entries
          - cache_hit:   True if answered from cache (no LLM tokens spent)
          - hit_count:   how many times this (or a similar) question was asked
          - similar_query: the original cached question text (when cache_hit=True)
        """
        # ── 1. Cache lookup ───────────────────────────────────────────
        hit: CacheHit | None = self._cache.lookup(
            query=question,
            query_type=_QUERY_TYPE,
            channel=channel_name,
        )
        if hit:
            return {
                "answer":        hit.answer_text,
                "sources":       hit.sources,
                "cache_hit":     True,
                "hit_count":     hit.hit_count,
                "similar_query": hit.query_text,
            }

        # ── 2. RAG: retrieve + generate ───────────────────────────────
        hits = self._store.search(question, n_results=self._n, channel_name=channel_name)
        if not hits:
            return {
                "answer":    "The knowledge base is empty. Run the ingest pipeline first.",
                "sources":   [],
                "cache_hit": False,
                "hit_count": 0,
                "similar_query": None,
            }

        context = "\n\n---\n\n".join(
            f"[#{h['metadata']['channel_name']} | {h['metadata'].get('oldest_dt','')} – "
            f"{h['metadata'].get('newest_dt','')} | by {', '.join(h['metadata'].get('authors', []))}]\n"
            f"{h['summary']}"
            for h in hits
        )
        prompt = _QA_PROMPT_TEMPLATE.format(context=context, question=question)
        resp = self._client.generate(
            model=self._model,
            prompt=prompt,
            options={"temperature": 0.1},
        )
        answer = resp.response.strip()
        sources = [h["metadata"] for h in hits]

        # ── 3. Store in cache ─────────────────────────────────────────
        self._cache.store(
            query=question,
            query_type=_QUERY_TYPE,
            answer=answer,
            sources=sources,
            channel=channel_name,
        )

        return {
            "answer":        answer,
            "sources":       sources,
            "cache_hit":     False,
            "hit_count":     0,
            "similar_query": None,
        }

    @classmethod
    def from_env(cls, store: KnowledgeStore, cache: ResponseCache) -> "KnowledgeBaseQA":
        return cls(
            store=store,
            cache=cache,
            model=os.environ.get("OLLAMA_GEN_MODEL", "llama3.2"),
            host=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            n_context_entries=int(os.environ.get("N_CONTEXT_ENTRIES", 5)),
        )
