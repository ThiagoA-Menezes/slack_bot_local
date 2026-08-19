"""
Response cache with semantic deduplication.

Every time a question is answered (ask_question) or a channel summary is
retrieved (get_channel_summary), the result is stored here alongside its
query embedding.  On subsequent calls the embedding of the new query is
compared against stored ones; if the cosine similarity exceeds the
configured threshold the cached answer is returned immediately —
no LLM generation required.

Hit counts are tracked per entry so the most-queried topics can be
surfaced as engagement/opportunity signals.

Index: response_cache
Schema
------
{
  "cache_id":    "<sha256[:24] of canonical query>",
  "query_type":  "ask_question" | "get_channel_summary",
  "channel":     "general" | null,        # null for cross-channel asks
  "query_text":  "<original query string>",
  "answer_text": "<full generated answer>",
  "sources":     [...],                   # list of source metadata dicts
  "hit_count":   <int>,                   # how many times this was reused
  "embedding":   [0.12, -0.34, ...],      # 768-dim nomic-embed-text vector
  "created_at":  "2024-01-15T11:30:00Z",
  "last_hit_at": "2024-01-15T12:00:00Z"
}
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import ollama
from opensearchpy import OpenSearch, RequestsHttpConnection
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_INDEX = "response_cache"
_VECTOR_DIMS = 768
_DEFAULT_THRESHOLD = 0.92   # cosine similarity — tune as needed

_INDEX_MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100,
        }
    },
    "mappings": {
        "properties": {
            "cache_id":    {"type": "keyword"},
            "query_type":  {"type": "keyword"},
            "channel":     {"type": "keyword"},
            "query_text":  {"type": "text"},
            "answer_text": {"type": "text"},
            "sources":     {"type": "object", "enabled": False},
            "hit_count":   {"type": "integer"},
            "embedding": {
                "type":      "knn_vector",
                "dimension": _VECTOR_DIMS,
                "method": {
                    "name":       "hnsw",
                    "engine":     "lucene",
                    "space_type": "cosinesimil",
                    "parameters": {"ef_construction": 128, "m": 16},
                },
            },
            "created_at":  {"type": "date"},
            "last_hit_at": {"type": "date"},
        }
    },
}


@dataclass
class CacheHit:
    cache_id: str
    query_type: str
    channel: str | None
    query_text: str
    answer_text: str
    sources: list[dict]
    hit_count: int
    score: float


class ResponseCache:
    """
    Semantic response cache backed by OpenSearch k-NN.

    Usage
    -----
    cache = ResponseCache.from_env()

    # Try cache first
    hit = cache.lookup("what is an API?", query_type="ask_question")
    if hit:
        return hit.answer_text   # free — no LLM call needed

    # Generate answer, then persist
    answer = llm.generate(...)
    cache.store("what is an API?", "ask_question", answer, sources=[...])
    """

    def __init__(
        self,
        os_url: str,
        os_user: str | None,
        os_password: str | None,
        ollama_host: str,
        embed_model: str,
        similarity_threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        use_ssl = os_url.startswith("https")
        auth = (os_user, os_password) if os_user and os_password else None
        self._os = OpenSearch(
            hosts=[os_url],
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=use_ssl,
            connection_class=RequestsHttpConnection,
            timeout=30,
        )
        self._ollama = ollama.Client(host=ollama_host)
        self._embed_model = embed_model
        self._threshold = similarity_threshold
        self._ensure_index()

        count = self._os.count(index=_INDEX)["count"]
        logger.info(
            "ResponseCache ready — %d entries in index '%s' (threshold=%.2f)",
            count, _INDEX, similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        query: str,
        query_type: str,
        channel: str | None = None,
    ) -> CacheHit | None:
        """
        Return a cached answer if a semantically similar query exists.

        Parameters
        ----------
        query:
            The user's natural-language question or channel name.
        query_type:
            ``"ask_question"`` or ``"get_channel_summary"``.
        channel:
            Optional channel filter — cache entries for different channels
            are kept separate even if queries look similar.
        """
        vector = self._embed(query)

        body: dict[str, Any] = {
            "size": 1,
            "query": {
                "knn": {
                    "embedding": {"vector": vector, "k": 1},
                }
            },
            "post_filter": {
                "bool": {
                    "must": [{"term": {"query_type": query_type}}],
                }
            },
            "_source": [
                "cache_id", "query_type", "channel",
                "query_text", "answer_text", "sources", "hit_count",
            ],
        }

        if channel:
            body["post_filter"]["bool"]["must"].append(
                {"term": {"channel": channel}}
            )
        else:
            # Only match entries that are also cross-channel
            body["post_filter"]["bool"]["must"].append(
                {"bool": {"must_not": {"exists": {"field": "channel"}}}}
            )

        try:
            resp = self._os.search(index=_INDEX, body=body)
        except Exception as exc:
            logger.warning("Cache lookup failed: %s", exc)
            return None

        hits = resp["hits"]["hits"]
        if not hits:
            return None

        hit = hits[0]
        score: float = hit["_score"]

        if score < self._threshold:
            logger.debug(
                "Cache miss — best score %.4f < threshold %.2f for %r",
                score, self._threshold, query[:80],
            )
            return None

        src = hit["_source"]
        logger.info(
            "Cache HIT (score=%.4f, hits=%d) for %r",
            score, src["hit_count"], query[:80],
        )
        self._increment_hit(hit["_id"], src["hit_count"])

        return CacheHit(
            cache_id=src["cache_id"],
            query_type=src["query_type"],
            channel=src.get("channel"),
            query_text=src["query_text"],
            answer_text=src["answer_text"],
            sources=src.get("sources", []),
            hit_count=src["hit_count"] + 1,   # reflects increment
            score=score,
        )

    def store(
        self,
        query: str,
        query_type: str,
        answer: str,
        sources: list[dict] | None = None,
        channel: str | None = None,
    ) -> str:
        """
        Persist a new query/answer pair.

        Returns the cache_id of the stored entry.
        """
        vector = self._embed(query)
        cache_id = _make_id(query_type, channel or "", query)
        now = datetime.now(timezone.utc).isoformat()

        doc: dict[str, Any] = {
            "cache_id":    cache_id,
            "query_type":  query_type,
            "query_text":  query,
            "answer_text": answer,
            "sources":     sources or [],
            "hit_count":   0,
            "embedding":   vector,
            "created_at":  now,
            "last_hit_at": now,
        }
        if channel:
            doc["channel"] = channel

        self._os.index(index=_INDEX, id=cache_id, body=doc)
        logger.info("Cached new entry [%s] for %r", cache_id, query[:80])
        return cache_id

    def top_queries(
        self,
        query_type: str | None = None,
        size: int = 20,
    ) -> list[dict]:
        """
        Return the most-queried topics, sorted by hit_count descending.

        Useful for opportunity mapping — shows which subjects users ask
        about the most.
        """
        body: dict[str, Any] = {
            "size": size,
            "sort": [{"hit_count": {"order": "desc"}}],
            "_source": [
                "cache_id", "query_type", "channel",
                "query_text", "hit_count", "created_at", "last_hit_at",
            ],
        }
        if query_type:
            body["query"] = {"term": {"query_type": query_type}}
        else:
            body["query"] = {"match_all": {}}

        try:
            resp = self._os.search(index=_INDEX, body=body)
        except Exception as exc:
            logger.warning("top_queries failed: %s", exc)
            return []

        return [
            {
                "rank":        i + 1,
                "query_text":  h["_source"]["query_text"],
                "query_type":  h["_source"]["query_type"],
                "channel":     h["_source"].get("channel"),
                "hit_count":   h["_source"]["hit_count"],
                "created_at":  h["_source"]["created_at"],
                "last_hit_at": h["_source"].get("last_hit_at"),
            }
            for i, h in enumerate(resp["hits"]["hits"])
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _increment_hit(self, doc_id: str, current_count: int) -> None:
        """Atomically increment hit_count and update last_hit_at."""
        try:
            self._os.update(
                index=_INDEX,
                id=doc_id,
                body={
                    "doc": {
                        "hit_count":   current_count + 1,
                        "last_hit_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )
        except Exception as exc:
            logger.warning("Failed to increment hit count for %s: %s", doc_id, exc)

    @retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(4))
    def _embed(self, text: str) -> list[float]:
        resp = self._ollama.embed(model=self._embed_model, input=text)
        return resp.embeddings[0]

    def _ensure_index(self) -> None:
        if not self._os.indices.exists(index=_INDEX):
            self._os.indices.create(index=_INDEX, body=_INDEX_MAPPING)
            logger.info("Created OpenSearch index '%s'", _INDEX)

    @classmethod
    def from_env(cls) -> "ResponseCache":
        return cls(
            os_url=os.environ.get("OPENSEARCH_URL", "http://localhost:9200"),
            os_user=os.environ.get("OPENSEARCH_USER") or None,
            os_password=os.environ.get("OPENSEARCH_PASSWORD") or None,
            ollama_host=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            embed_model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            similarity_threshold=float(
                os.environ.get("CACHE_SIMILARITY_THRESHOLD", _DEFAULT_THRESHOLD)
            ),
        )


def _make_id(query_type: str, channel: str, query: str) -> str:
    raw = f"{query_type}:{channel}:{query.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
