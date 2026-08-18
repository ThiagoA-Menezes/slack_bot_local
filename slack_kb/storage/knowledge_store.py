"""
OpenSearch knowledge store with k-NN vector search.

Every knowledge entry is stored as an OpenSearch document with:
  - All metadata fields (channel_id, channel_name, oldest_ts, newest_ts, summary)
  - authors: sorted list of display names who contributed to this chunk
  - oldest_dt / newest_dt: human-readable UTC date range strings
  - A dense_vector field ``embedding`` produced by the local Ollama embedding
    model nomic-embed-text (768 dimensions)

Document schema
---------------
{
  "doc_id":       "<sha256[:24]>",
  "channel_id":   "C0123ABC",
  "channel_name": "general",
  "oldest_ts":    "1700000000.000000",
  "newest_ts":    "1700001000.000000",
  "oldest_dt":    "2024-01-15 09:00 UTC",
  "newest_dt":    "2024-01-15 11:30 UTC",
  "authors":      ["Alice Smith", "Bob Jones"],
  "summary":      "**Topics:** …",
  "embedding":    [0.12, -0.34, …],   # 768-dim float vector
  "ingested_at":  "2024-01-15T11:30:00Z"
}
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import ollama
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# OpenSearch index name
_INDEX = "slack_knowledge"

# Ollama embedding model — 768 dimensions
# Pull with: ollama pull nomic-embed-text
_DEFAULT_EMBED_MODEL = "nomic-embed-text"
_VECTOR_DIMS = 768

# Index mapping: knn_vector field + BM25-friendly keyword/text fields
_INDEX_MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100,
        }
    },
    "mappings": {
        "properties": {
            "doc_id":       {"type": "keyword"},
            "channel_id":   {"type": "keyword"},
            "channel_name": {"type": "keyword"},
            "oldest_ts":    {"type": "keyword"},
            "newest_ts":    {"type": "keyword"},
            "oldest_dt":    {"type": "keyword"},
            "newest_dt":    {"type": "keyword"},
            "authors":      {"type": "keyword"},
            "summary":      {"type": "text"},
            "ingested_at":  {"type": "date"},
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
        }
    },
}


@dataclass
class KnowledgeEntry:
    channel_id: str
    channel_name: str
    oldest_ts: str
    newest_ts: str
    oldest_dt: str
    newest_dt: str
    authors: list[str]
    summary: str


class KnowledgeStore:
    """Persists summarised entries in OpenSearch with k-NN vector search."""

    def __init__(
        self,
        os_url: str,
        os_user: str | None,
        os_password: str | None,
        ollama_host: str,
        embed_model: str,
    ) -> None:
        # ── OpenSearch client ─────────────────────────────────────────
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
        self._ensure_index()

        # ── Ollama embedding client ───────────────────────────────────
        self._ollama = ollama.Client(host=ollama_host)
        self._embed_model = embed_model

        count = self._os.count(index=_INDEX)["count"]
        logger.info(
            "KnowledgeStore ready — %d entries in index '%s' (embed: %s)",
            count, _INDEX, embed_model,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, entry: KnowledgeEntry) -> None:
        vector = self._embed(entry.summary)
        doc_id = _make_id(entry.channel_id, entry.oldest_ts, entry.newest_ts)
        self._os.index(
            index=_INDEX,
            id=doc_id,
            body=self._to_doc(entry, vector, doc_id),
        )

    def upsert_many(self, entries: list[KnowledgeEntry]) -> None:
        if not entries:
            return

        # Embed all summaries — one call per entry (Ollama is local, no quota)
        actions = []
        for entry in entries:
            vector = self._embed(entry.summary)
            doc_id = _make_id(entry.channel_id, entry.oldest_ts, entry.newest_ts)
            actions.append(
                {
                    "_op_type": "index",
                    "_index": _INDEX,
                    "_id": doc_id,
                    **self._to_doc(entry, vector, doc_id),
                }
            )

        success, errors = bulk(self._os, actions, raise_on_error=False)
        if errors:
            logger.warning("Bulk upsert had %d errors: %s", len(errors), errors[:3])
        logger.info("Upserted %d entries (%d succeeded)", len(entries), success)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        channel_name: str | None = None,
    ) -> list[dict]:
        """
        k-NN vector search over stored summaries.

        Embeds *query* with Ollama, then finds the *n_results* nearest
        documents by cosine similarity.
        """
        query_vector = self._embed(query)

        knn_query: dict = {
            "size": n_results,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": n_results,
                    }
                }
            },
            "_source": ["channel_id", "channel_name", "oldest_ts", "newest_ts",
                        "oldest_dt", "newest_dt", "authors", "summary"],
        }

        if channel_name:
            knn_query["post_filter"] = {"term": {"channel_name": channel_name}}

        resp = self._os.search(index=_INDEX, body=knn_query)
        hits = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            hits.append(
                {
                    "summary": src["summary"],
                    "score":   hit["_score"],
                    "metadata": {
                        "channel_id":   src["channel_id"],
                        "channel_name": src["channel_name"],
                        "oldest_ts":    src["oldest_ts"],
                        "newest_ts":    src["newest_ts"],
                        "oldest_dt":    src.get("oldest_dt", ""),
                        "newest_dt":    src.get("newest_dt", ""),
                        "authors":      src.get("authors", []),
                    },
                }
            )
        return hits

    def list_channels(self) -> list[str]:
        """Return a sorted list of channel names that have at least one entry."""
        resp = self._os.search(
            index=_INDEX,
            body={
                "size": 0,
                "aggs": {"channels": {"terms": {"field": "channel_name", "size": 500}}},
            },
        )
        buckets = resp["aggregations"]["channels"]["buckets"]
        return sorted(b["key"] for b in buckets)

    def get_channel_summary(self, channel_name: str, limit: int = 50) -> list[dict]:
        """Return stored entries for *channel_name*, sorted newest-first."""
        resp = self._os.search(
            index=_INDEX,
            body={
                "size": limit,
                "query": {"term": {"channel_name": channel_name}},
                "sort":  [{"newest_ts": {"order": "desc"}}],
                "_source": ["summary", "oldest_ts", "newest_ts", "oldest_dt", "newest_dt", "authors"],
            },
        )
        return [
            {
                "summary":   h["_source"]["summary"],
                "oldest_ts": h["_source"]["oldest_ts"],
                "newest_ts": h["_source"]["newest_ts"],
                "oldest_dt": h["_source"].get("oldest_dt", ""),
                "newest_dt": h["_source"].get("newest_dt", ""),
                "authors":   h["_source"].get("authors", []),
            }
            for h in resp["hits"]["hits"]
        ]

    def get_newest_ts(self, channel_id: str) -> str | None:
        """Return the newest_ts of the most recently ingested chunk for *channel_id*."""
        resp = self._os.search(
            index=_INDEX,
            body={
                "size": 1,
                "query": {"term": {"channel_id": channel_id}},
                "sort":  [{"newest_ts": {"order": "desc"}}],
                "_source": ["newest_ts"],
            },
        )
        hits = resp["hits"]["hits"]
        return hits[0]["_source"]["newest_ts"] if hits else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_doc(entry: KnowledgeEntry, vector: list[float], doc_id: str) -> dict:
        return {
            "doc_id":       doc_id,
            "channel_id":   entry.channel_id,
            "channel_name": entry.channel_name,
            "oldest_ts":    entry.oldest_ts,
            "newest_ts":    entry.newest_ts,
            "oldest_dt":    entry.oldest_dt,
            "newest_dt":    entry.newest_dt,
            "authors":      entry.authors,
            "summary":      entry.summary,
            "embedding":    vector,
            "ingested_at":  datetime.now(timezone.utc).isoformat(),
        }

    @retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(4))
    def _embed(self, text: str) -> list[float]:
        resp = self._ollama.embed(model=self._embed_model, input=text)
        return resp.embeddings[0]

    def _ensure_index(self) -> None:
        if not self._os.indices.exists(index=_INDEX):
            self._os.indices.create(index=_INDEX, body=_INDEX_MAPPING)
            logger.info("Created OpenSearch index '%s'", _INDEX)

    @classmethod
    def from_env(cls) -> "KnowledgeStore":
        return cls(
            os_url=os.environ.get("OPENSEARCH_URL", "http://localhost:9200"),
            os_user=os.environ.get("OPENSEARCH_USER") or None,
            os_password=os.environ.get("OPENSEARCH_PASSWORD") or None,
            ollama_host=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            embed_model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        )


def _make_id(channel_id: str, oldest_ts: str, newest_ts: str) -> str:
    raw = f"{channel_id}:{oldest_ts}:{newest_ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
