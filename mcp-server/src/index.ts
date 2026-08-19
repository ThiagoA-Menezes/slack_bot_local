#!/usr/bin/env node
/**
 * Slack Knowledge Base — MCP Server (Ollama + OpenSearch edition)
 *
 * Tools exposed to AI assistants:
 *
 *   list_channels         — list channels present in the knowledge base
 *   search_knowledge      — k-NN vector search; results include authors + dates
 *   get_channel_summary   — all summaries for a channel, with authors + dates
 *   ask_question          — RAG with semantic cache; returns cached answer when
 *                           a similar question was already answered before
 *   post_message          — post a reply back into a Slack channel or thread
 *   top_queries           — opportunity mapping: most-asked topics by hit count
 *
 * All AI inference is local — no cloud credentials required.
 *
 * Configuration (environment variables):
 *   OPENSEARCH_URL              — OpenSearch base URL (default: http://localhost:9200)
 *   OPENSEARCH_USER             — Basic-auth username (optional)
 *   OPENSEARCH_PASSWORD         — Basic-auth password (optional)
 *   OLLAMA_URL                  — Ollama base URL (default: http://localhost:11434)
 *   OLLAMA_EMBED_MODEL          — Embedding model (default: nomic-embed-text)
 *   OLLAMA_GEN_MODEL            — Generation model (default: llama3.2)
 *   SLACK_BOT_TOKEN             — Slack bot token (required only for post_message)
 *   CACHE_SIMILARITY_THRESHOLD  — cosine similarity cutoff for cache hits (default: 0.92)
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Client as OpenSearchClient } from "@opensearch-project/opensearch";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const OS_INDEX        = "slack_knowledge";
const CACHE_INDEX     = "response_cache";
const OLLAMA_URL      = process.env.OLLAMA_URL        ?? "http://localhost:11434";
const EMBED_MODEL     = process.env.OLLAMA_EMBED_MODEL ?? "nomic-embed-text";
const GEN_MODEL       = process.env.OLLAMA_GEN_MODEL   ?? "llama3.2";
const SLACK_BOT_TOKEN = process.env.SLACK_BOT_TOKEN    ?? "";
const CACHE_THRESHOLD = parseFloat(process.env.CACHE_SIMILARITY_THRESHOLD ?? "0.92");

// ---------------------------------------------------------------------------
// OpenSearch client
// ---------------------------------------------------------------------------

const OPENSEARCH_URL      = process.env.OPENSEARCH_URL      ?? "http://localhost:9200";
const OPENSEARCH_USER     = process.env.OPENSEARCH_USER;
const OPENSEARCH_PASSWORD = process.env.OPENSEARCH_PASSWORD;

const osAuth =
  OPENSEARCH_USER && OPENSEARCH_PASSWORD
    ? `Basic ${Buffer.from(`${OPENSEARCH_USER}:${OPENSEARCH_PASSWORD}`).toString("base64")}`
    : undefined;

const osClient = new OpenSearchClient({
  node: OPENSEARCH_URL,
  ...(osAuth ? { headers: { Authorization: osAuth } } : {}),
  ssl: { rejectUnauthorized: OPENSEARCH_URL.startsWith("https") },
});

// ---------------------------------------------------------------------------
// Ollama helpers
// ---------------------------------------------------------------------------

async function embedText(text: string): Promise<number[]> {
  const resp = await fetch(`${OLLAMA_URL}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: EMBED_MODEL, input: text }),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Ollama embed error ${resp.status}: ${txt}`);
  }
  const json = (await resp.json()) as { embeddings: number[][] };
  return json.embeddings[0];
}

async function ollamaGenerate(prompt: string): Promise<string> {
  const resp = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: GEN_MODEL,
      prompt,
      stream: false,
      options: { temperature: 0.1 },
    }),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Ollama generate error ${resp.status}: ${txt}`);
  }
  const json = (await resp.json()) as { response: string };
  return json.response.trim();
}

const QA_PROMPT = (context: string, question: string) => `\
You are a helpful knowledge base assistant. Answer the question below \
using ONLY the context provided. If the answer is not in the context, \
say "I don't have enough information to answer that."

Context (summaries of Slack conversations):
${context}

Question: ${question}

Answer:`;

// ---------------------------------------------------------------------------
// Shared types + helpers
// ---------------------------------------------------------------------------

interface KnowledgeSource {
  _score: number;
  _source: {
    channel_name: string;
    oldest_ts:    string;
    newest_ts:    string;
    oldest_dt:    string;
    newest_dt:    string;
    authors:      string[];
    summary:      string;
  };
}

function formatHit(h: KnowledgeSource, idx: number): string {
  const authors = h._source.authors?.length ? h._source.authors.join(", ") : "unknown";
  return (
    `[${idx}] #${h._source.channel_name}  score=${h._score.toFixed(4)}\n` +
    `    📅 ${h._source.oldest_dt} – ${h._source.newest_dt}\n` +
    `    👥 ${authors}\n` +
    h._source.summary
  );
}

async function knnSearch(vector: number[], k: number, channel?: string): Promise<KnowledgeSource[]> {
  const body: Record<string, unknown> = {
    size: k,
    query: { knn: { embedding: { vector, k } } },
    _source: ["channel_name", "oldest_ts", "newest_ts", "oldest_dt", "newest_dt", "authors", "summary"],
  };
  if (channel) body["post_filter"] = { term: { channel_name: channel } };
  const resp = await osClient.search({ index: OS_INDEX, body });
  return (resp.body.hits?.hits ?? []) as unknown as KnowledgeSource[];
}

// ---------------------------------------------------------------------------
// Response cache helpers
// ---------------------------------------------------------------------------

interface CacheDoc {
  cache_id:    string;
  query_type:  string;
  channel?:    string;
  query_text:  string;
  answer_text: string;
  sources:     unknown[];
  hit_count:   number;
}

async function cacheLookup(
  vector: number[],
  queryType: string,
  channel?: string,
): Promise<{ hit: CacheDoc; score: number; docId: string } | null> {
  const mustClauses: unknown[] = [{ term: { query_type: queryType } }];
  if (channel) {
    mustClauses.push({ term: { channel } });
  } else {
    mustClauses.push({ bool: { must_not: { exists: { field: "channel" } } } });
  }

  try {
    const resp = await (osClient.search as (p: Record<string, unknown>) => Promise<{ body: Record<string, unknown> }>)({
      index: CACHE_INDEX,
      body: {
        size: 1,
        query: { knn: { embedding: { vector, k: 1 } } },
        post_filter: { bool: { must: mustClauses } },
        _source: ["cache_id", "query_type", "channel", "query_text", "answer_text", "sources", "hit_count"],
      },
    });
    const rawHits = (resp.body as Record<string, unknown>);
    const hits = ((rawHits["hits"] as Record<string, unknown>)?.["hits"] ?? []) as Array<{ _id: string; _score: number; _source: CacheDoc }>;
    if (!hits.length) return null;
    const top = hits[0];
    if (top._score < CACHE_THRESHOLD) return null;
    return { hit: top._source, score: top._score, docId: top._id };
  } catch {
    return null;
  }
}

async function cacheIncrement(docId: string, currentCount: number): Promise<void> {
  try {
    await osClient.update({
      index: CACHE_INDEX,
      id: docId,
      body: { doc: { hit_count: currentCount + 1, last_hit_at: new Date().toISOString() } },
    });
  } catch { /* best-effort */ }
}

async function cacheStore(
  query: string,
  queryType: string,
  answer: string,
  sources: unknown[],
  vector: number[],
  channel?: string,
): Promise<void> {
  const raw = `${queryType}:${channel ?? ""}:${query.toLowerCase().trim()}`;
  const cacheId = await sha256Prefix(raw);
  const now = new Date().toISOString();
  const doc: Record<string, unknown> = {
    cache_id:    cacheId,
    query_type:  queryType,
    query_text:  query,
    answer_text: answer,
    sources,
    hit_count:   0,
    embedding:   vector,
    created_at:  now,
    last_hit_at: now,
  };
  if (channel) doc["channel"] = channel;
  try {
    await osClient.index({ index: CACHE_INDEX, id: cacheId, body: doc });
  } catch { /* best-effort */ }
}

async function sha256Prefix(text: string): Promise<string> {
  const { createHash } = await import("node:crypto");
  return createHash("sha256").update(text).digest("hex").slice(0, 24);
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

const server = new McpServer({ name: "slack-kb", version: "0.5.0" });

// ── Tool 1: list_channels ───────────────────────────────────────────────────

server.registerTool(
  "list_channels",
  {
    description: "List all Slack channels ingested into the knowledge base.",
    inputSchema: z.object({}),
  },
  async () => {
    try {
      const resp = await osClient.search({
        index: OS_INDEX,
        body: {
          size: 0,
          aggs: { channels: { terms: { field: "channel_name", size: 500 } } },
        },
      });
      const buckets = ((resp.body.aggregations?.channels as unknown as {
        buckets?: Array<{ key: string; doc_count: number }>;
      })?.buckets ?? []);

      if (buckets.length === 0) {
        return { content: [{ type: "text" as const, text: "No channels found. Run the ingest pipeline first." }] };
      }
      const lines = buckets
        .sort((a, b) => a.key.localeCompare(b.key))
        .map((b) => `• #${b.key}  (${b.doc_count} knowledge entries)`)
        .join("\n");
      return { content: [{ type: "text" as const, text: `Channels in knowledge base (${buckets.length}):\n\n${lines}` }] };
    } catch (err) {
      return { content: [{ type: "text" as const, text: `Error: ${String(err)}` }], isError: true };
    }
  }
);

// ── Tool 2: search_knowledge ────────────────────────────────────────────────

server.registerTool(
  "search_knowledge",
  {
    description:
      "Semantic k-NN vector search over the knowledge base using local Ollama embeddings. " +
      "Each result shows the channel, date range, authors, and summary.",
    inputSchema: z.object({
      query:   z.string().describe("Natural language search query"),
      channel: z.string().optional().describe("Restrict to a specific channel (without #)"),
      top_k:   z.number().int().min(1).max(20).default(5).describe("Max results"),
    }),
  },
  async ({ query, channel, top_k }) => {
    try {
      const vector = await embedText(query);
      const hits   = await knnSearch(vector, top_k, channel);
      if (hits.length === 0) return { content: [{ type: "text" as const, text: "No results found." }] };
      return { content: [{ type: "text" as const, text: hits.map(formatHit).join("\n\n---\n\n") }] };
    } catch (err) {
      return { content: [{ type: "text" as const, text: `Error: ${String(err)}` }], isError: true };
    }
  }
);

// ── Tool 3: get_channel_summary ─────────────────────────────────────────────

server.registerTool(
  "get_channel_summary",
  {
    description:
      "Return stored knowledge entries for a Slack channel, newest-first. " +
      "Each entry shows authors and date range. " +
      "Results are cached semantically — repeated or similar requests are served instantly.",
    inputSchema: z.object({
      channel: z.string().describe("Slack channel name (without #)"),
      limit:   z.number().int().min(1).max(50).default(10).describe("Max entries"),
    }),
  },
  async ({ channel, limit }) => {
    try {
      // ── Cache lookup ─────────────────────────────────────────────
      const vector    = await embedText(channel);
      const cacheResult = await cacheLookup(vector, "get_channel_summary", channel);
      if (cacheResult) {
        const { hit, score } = cacheResult;
        await cacheIncrement(cacheResult.docId, hit.hit_count);
        return {
          content: [{
            type: "text" as const,
            text:
              `⚡ *Resposta em cache* (similaridade=${score.toFixed(4)}, consultado ${hit.hit_count + 1}x)\n\n` +
              hit.answer_text,
          }],
        };
      }

      // ── Fetch from OpenSearch ────────────────────────────────────
      const resp = await osClient.search({
        index: OS_INDEX,
        body: {
          size: limit,
          query: { term: { channel_name: channel } },
          sort:  [{ newest_ts: { order: "desc" } }],
          _source: ["summary", "oldest_ts", "newest_ts", "oldest_dt", "newest_dt", "authors"],
        },
      });
      const hits = (resp.body.hits?.hits ?? []) as unknown as Array<{ _source: KnowledgeSource["_source"] }>;
      if (hits.length === 0) return { content: [{ type: "text" as const, text: `No entries found for #${channel}.` }] };

      const parts = hits.map((h, i) => {
        const authors = h._source.authors?.length ? h._source.authors.join(", ") : "unknown";
        return (
          `[${i + 1}] 📅 ${h._source.oldest_dt} – ${h._source.newest_dt}\n` +
          `     👥 ${authors}\n` +
          h._source.summary
        );
      });
      const answer = `Entries for #${channel} (${hits.length}):\n\n` + parts.join("\n\n---\n\n");

      // ── Store in cache ───────────────────────────────────────────
      await cacheStore(channel, "get_channel_summary", answer, [], vector, channel);

      return { content: [{ type: "text" as const, text: answer }] };
    } catch (err) {
      return { content: [{ type: "text" as const, text: `Error: ${String(err)}` }], isError: true };
    }
  }
);

// ── Tool 4: ask_question ────────────────────────────────────────────────────

server.registerTool(
  "ask_question",
  {
    description:
      "Answer a question using RAG with semantic cache: embeds with Ollama, checks " +
      "the response cache first (returns instantly if a similar question was already " +
      "answered), otherwise retrieves relevant knowledge entries and generates a grounded " +
      "answer using a local Ollama model. 100% local — no cloud credentials needed.",
    inputSchema: z.object({
      question:        z.string().describe("Natural language question"),
      channel:         z.string().optional().describe("Restrict to one channel (optional)"),
      context_entries: z.number().int().min(1).max(10).default(5)
        .describe("Number of knowledge entries to use as context"),
    }),
  },
  async ({ question, channel, context_entries }) => {
    try {
      const vector = await embedText(question);

      // ── 1. Cache lookup ──────────────────────────────────────────
      const cacheResult = await cacheLookup(vector, "ask_question", channel);
      if (cacheResult) {
        const { hit, score } = cacheResult;
        await cacheIncrement(cacheResult.docId, hit.hit_count);
        const sourcesText = Array.isArray(hit.sources) && hit.sources.length
          ? "\n\n**Sources (cached):**\n" + (hit.sources as Array<Record<string, unknown>>).map(
              (s) => `• #${s["channel_name"]}  ${s["oldest_dt"] ?? ""} – ${s["newest_dt"] ?? ""}  (${(s["authors"] as string[] | undefined)?.join(", ") ?? "unknown"})`
            ).join("\n")
          : "";
        return {
          content: [{
            type: "text" as const,
            text:
              `⚡ *Resposta em cache* (similaridade=${score.toFixed(4)}, consultado ${hit.hit_count + 1}x)\n` +
              `*Pergunta original:* ${hit.query_text}\n\n` +
              `**Answer:**\n${hit.answer_text}` +
              sourcesText,
          }],
        };
      }

      // ── 2. RAG: retrieve + generate ──────────────────────────────
      const hits = await knnSearch(vector, context_entries, channel);
      if (hits.length === 0) {
        return { content: [{ type: "text" as const, text: "No results. Run the ingest pipeline first." }] };
      }

      const context = hits.map((h) => {
        const authors = h._source.authors?.length ? h._source.authors.join(", ") : "unknown";
        return `[#${h._source.channel_name} | ${h._source.oldest_dt} – ${h._source.newest_dt} | by ${authors}]\n${h._source.summary}`;
      }).join("\n\n---\n\n");

      const answer  = await ollamaGenerate(QA_PROMPT(context, question));
      const sources = hits.map((h) => ({
        channel_name: h._source.channel_name,
        oldest_dt:    h._source.oldest_dt,
        newest_dt:    h._source.newest_dt,
        authors:      h._source.authors,
      }));
      const sourcesText = sources.map(
        (s) => `• #${s.channel_name}  ${s.oldest_dt} – ${s.newest_dt}  (${s.authors?.join(", ") ?? "unknown"})`
      ).join("\n");

      // ── 3. Store in cache ────────────────────────────────────────
      await cacheStore(question, "ask_question", answer, sources, vector, channel);

      return { content: [{ type: "text" as const, text: `**Answer:**\n${answer}\n\n**Sources:**\n${sourcesText}` }] };
    } catch (err) {
      return { content: [{ type: "text" as const, text: `Error: ${String(err)}` }], isError: true };
    }
  }
);

// ── Tool 5: post_message ────────────────────────────────────────────────────

server.registerTool(
  "post_message",
  {
    description:
      "Post a message to a Slack channel or reply to a thread. " +
      "Requires SLACK_BOT_TOKEN and the chat:write OAuth scope.",
    inputSchema: z.object({
      channel:   z.string().describe("Channel name (e.g. 'general') or channel ID"),
      text:      z.string().describe("Message text (Slack markdown supported)"),
      thread_ts: z.string().optional().describe("Reply in this thread (parent message ts)"),
    }),
  },
  async ({ channel, text, thread_ts }) => {
    if (!SLACK_BOT_TOKEN) {
      return { content: [{ type: "text" as const, text: "SLACK_BOT_TOKEN must be set to use post_message." }], isError: true };
    }
    try {
      const body: Record<string, string> = { channel, text };
      if (thread_ts) body["thread_ts"] = thread_ts;

      const resp = await fetch("https://slack.com/api/chat.postMessage", {
        method: "POST",
        headers: {
          "Content-Type": "application/json; charset=utf-8",
          Authorization:  `Bearer ${SLACK_BOT_TOKEN}`,
        },
        body: JSON.stringify(body),
      });
      const json = (await resp.json()) as { ok: boolean; error?: string; ts?: string };
      if (!json.ok) return { content: [{ type: "text" as const, text: `Slack API error: ${json.error}` }], isError: true };

      return {
        content: [{ type: "text" as const, text: `✓ Message posted to #${channel}${thread_ts ? " (thread reply)" : ""}\n  ts: ${json.ts}` }],
      };
    } catch (err) {
      return { content: [{ type: "text" as const, text: `Error: ${String(err)}` }], isError: true };
    }
  }
);

// ── Tool 6: top_queries ─────────────────────────────────────────────────────

server.registerTool(
  "top_queries",
  {
    description:
      "Return the most-queried topics from the response cache, sorted by hit count. " +
      "Useful for opportunity mapping — shows which subjects users ask about most.",
    inputSchema: z.object({
      query_type: z.enum(["ask_question", "get_channel_summary"]).optional()
        .describe("Filter by query type (omit for all types)"),
      size: z.number().int().min(1).max(100).default(20)
        .describe("Number of entries to return"),
    }),
  },
  async ({ query_type, size }) => {
    try {
      const body: Record<string, unknown> = {
        size,
        sort: [{ hit_count: { order: "desc" } }],
        _source: ["cache_id", "query_type", "channel", "query_text", "hit_count", "created_at", "last_hit_at"],
      };
      body["query"] = query_type ? { term: { query_type } } : { match_all: {} };

      const resp = await osClient.search({ index: CACHE_INDEX, body });
      const hits = (resp.body.hits?.hits ?? []) as unknown as Array<{
        _source: {
          query_type: string; channel?: string; query_text: string;
          hit_count: number; created_at: string; last_hit_at?: string;
        };
      }>;

      if (!hits.length) {
        return { content: [{ type: "text" as const, text: "No cached queries yet." }] };
      }

      const header = `${"#".padStart(4)}  ${"Hits".padStart(5)}  ${"Type".padEnd(20)}  ${"Channel".padEnd(12)}  Query`;
      const sep    = "─".repeat(80);
      const rows   = hits.map((h, i) => {
        const ch = h._source.channel ?? "—";
        const q  = h._source.query_text.length > 60
          ? h._source.query_text.slice(0, 57) + "..."
          : h._source.query_text;
        return `${String(i + 1).padStart(4)}  ${String(h._source.hit_count).padStart(5)}  ${h._source.query_type.padEnd(20)}  ${ch.padEnd(12)}  ${q}`;
      });

      return {
        content: [{
          type: "text" as const,
          text: `Most-queried topics (${hits.length}):\n\n${header}\n${sep}\n${rows.join("\n")}`,
        }],
      };
    } catch (err) {
      return { content: [{ type: "text" as const, text: `Error: ${String(err)}` }], isError: true };
    }
  }
);

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function main() {
  try {
    await osClient.cluster.health({});
    console.error("[slack-kb-mcp] Connected to OpenSearch at", OPENSEARCH_URL);
  } catch (err) {
    console.error("[slack-kb-mcp] WARNING: Could not reach OpenSearch:", err);
  }
  try {
    const r = await fetch(`${OLLAMA_URL}/api/tags`);
    if (r.ok) console.error(`[slack-kb-mcp] Connected to Ollama at ${OLLAMA_URL}`);
  } catch {
    console.error(`[slack-kb-mcp] WARNING: Could not reach Ollama at ${OLLAMA_URL}`);
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[slack-kb-mcp] MCP server running on stdio");
}

main().catch((err) => {
  console.error("[slack-kb-mcp] Fatal error:", err);
  process.exit(1);
});
