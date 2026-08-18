# Slack Knowledge Base

Reads messages from your Slack workspace, summarises them with **IBM watsonx.ai**, vectorises the summaries with `ibm/slate-30m-english-rtrvr` (384 dims), and stores them in **OpenSearch** with a k-NN HNSW index.  
An **MCP server** then exposes the knowledge base as tools that AI assistants (Bob, Claude, etc.) can call directly with true semantic search.

---

## How it works

```
Slack API
  └── SlackIngestionClient       cursor pagination · thread replies · rate-limit backoff
        │
        ▼
  MessageChunker                 token-bounded chunks (~2 000 tokens)
        │
        ▼
  WatsonxSummarizer              ibm/granite-13b-instruct-v2
        │  structured summary per chunk
        ▼
  KnowledgeStore                 OpenSearch  k-NN HNSW index (384 dims)
        │
        ├── cli.py               ingest · ask · search · channels (interactive)
        │
        └── mcp-server/          MCP tools available to any AI assistant
              list_channels · search_knowledge · get_channel_summary · ask_question
```

---

## Step-by-step setup

### Step 1 — Install system prerequisites

| Tool | Minimum version | How to get it |
|---|---|---|
| Python | 3.10 | [python.org](https://www.python.org/downloads/) |
| Node.js | 18 | [nodejs.org](https://nodejs.org/) |
| Podman | any | [podman.io](https://podman.io/getting-started/installation) — already installed ✓ |

---

### Step 2 — Start OpenSearch

Run a single-node OpenSearch container with the k-NN plugin enabled (included by default):

```bash
podman run -d --name opensearch-kb -p 9200:9200 -e discovery.type=single-node -e plugins.security.disabled=true -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=Sl@ckKB2024! opensearchproject/opensearch:2.18.0
```

> **Tip:** Paste the command above as a single line — multi-line `\` continuations can break in some terminals when pasted.

Verify it is running (wait ~20 seconds for OpenSearch to boot):

```bash
curl -s http://localhost:9200
# You should see JSON with "name", "cluster_name", "version" …
```

**Useful Podman commands:**

```bash
podman ps                        # list running containers
podman stop opensearch-kb        # stop the container
podman start opensearch-kb       # restart it later (data is preserved)
podman rm opensearch-kb          # remove container (deletes data)
podman logs opensearch-kb        # view logs
```

> **Managed OpenSearch?**
> Skip this step and use your cluster URL in `OPENSEARCH_URL`.
> AWS OpenSearch Serverless, Aiven, and Elastic Cloud are all supported.

---

### Step 3 — Create a Slack app and get a bot token

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From scratch**.
2. Give it a name (e.g. `knowledge-base-bot`) and pick your workspace.
3. In the left sidebar click **OAuth & Permissions**.
4. Under **Bot Token Scopes** add all six scopes:

   | Scope | Purpose |
   |---|---|
   | `channels:read` | List public channels |
   | `groups:read` | List private channels the bot is in |
   | `channels:history` | Read public channel messages |
   | `groups:history` | Read private channel messages |
   | `im:history` | Read direct messages |
   | `mpim:history` | Read group direct messages |

5. Click **Install to Workspace** → **Allow**.
6. Copy the **Bot User OAuth Token** — it starts with `xoxb-`.
7. **Add the bot to every channel you want to ingest:**  
   In each Slack channel, type `/invite @knowledge-base-bot`.

---

### Step 4 — Get IBM watsonx.ai credentials

1. Log in to [cloud.ibm.com](https://cloud.ibm.com) (free Lite tier works).
2. **API Key:** Menu → **Manage** → **Access (IAM)** → **API keys** → **Create** → copy the key.
3. **Project ID:** Open [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → your project → **Manage** tab → copy the **Project ID** GUID.
4. **Service URL:** use `https://us-south.ml.cloud.ibm.com` (Dallas) unless your account is in a different region.

---

### Step 5 — Configure the project

```bash
cd slack-kb
cp .env.example .env
```

Open `.env` and fill in every value:

```dotenv
SLACK_BOT_TOKEN=xoxb-your-token-here

WATSONX_API_KEY=your-ibm-api-key
WATSONX_PROJECT_ID=your-project-id-guid
WATSONX_URL=https://us-south.ml.cloud.ibm.com

OPENSEARCH_URL=http://localhost:9200
# Leave USER/PASSWORD empty if security is disabled (local Podman setup)
OPENSEARCH_USER=
OPENSEARCH_PASSWORD=
```

---

### Step 6 — Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 7 — Ingest your Slack channels

**All channels the bot is a member of:**

```bash
python cli.py ingest
```

**Specific channels only:**

```bash
python cli.py ingest -c general -c engineering -c product
```

The first run does a **full sync** of all history.  
Subsequent runs are **incremental** — only new messages since the last run are fetched.

Force a full re-sync at any time:

```bash
python cli.py ingest --full
```

---

### Step 8 — Query the knowledge base (CLI)

**Ask a question (RAG — LLM-generated answer):**

```bash
python cli.py ask "What did the team decide about the new API design?"
python cli.py ask "What are the open action items?" --channel engineering
```

**Semantic search (no LLM, raw similarity results):**

```bash
python cli.py search "database migration" --top 5
python cli.py search "deployment rollout" --channel devops
```

**Interactive channel browser:**

```bash
python cli.py channels
```

This launches a terminal UI where you can:

```
? Select channels  (Space to toggle, Enter to confirm):
 ❯ ○ #engineering
   ○ #general
   ○ #incident-response
   ○ #product
   ○ #releases

? What do you want to do with 2 channel(s)?
 ❯ 📖  View summaries
   🔍  Semantic search
   🔄  Re-ingest (incremental)
   🔄  Re-ingest (full re-sync)
```

Use **Space** to toggle channels, **Arrow keys** to navigate, and **Enter** to confirm.

---

### Step 9 — Build the MCP server

```bash
cd mcp-server
npm install
npm run build
cd ..
```

---

### Step 10 — Register the MCP server with Bob

Open `~/.bob/settings/mcp.json` and add the `slack-kb` entry (already done if you used automated setup):

```json
{
  "mcpServers": {
    "slack-kb": {
      "command": "node",
      "args": ["/absolute/path/to/slack-kb/mcp-server/build/index.js"],
      "env": {
        "OPENSEARCH_URL": "http://localhost:9200",
        "OPENSEARCH_USER": "",
        "OPENSEARCH_PASSWORD": "",
        "WATSONX_API_KEY": "your-ibm-api-key",
        "WATSONX_PROJECT_ID": "your-project-id",
        "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
        "WATSONX_MODEL_ID": "ibm/granite-13b-instruct-v2"
      }
    }
  }
}
```

Replace `/absolute/path/to/slack-kb` with the real path, e.g. `/Users/you/workspace/slack-kb`.

After saving, **reload Bob** — the `slack-kb` tools appear automatically.

---

### Step 11 — Use the MCP tools in Bob

Once registered, you can ask Bob things like:

> *"List all the Slack channels in the knowledge base."*  
> *"What did the engineering team discuss about the API migration?"*  
> *"Summarise everything posted in #incident-response."*  
> *"Were there any action items assigned in #product?"*

Bob will automatically call the right tool (`list_channels`, `ask_question`, `get_channel_summary`, or `search_knowledge`) and return a grounded answer.

---

## MCP tools reference

| Tool | Input | What it does |
|---|---|---|
| `list_channels` | *(none)* | Lists every channel present in the knowledge base |
| `search_knowledge` | `query`, `channel?`, `top_k?` | k-NN semantic search — returns raw summary hits, no LLM |
| `get_channel_summary` | `channel`, `limit?` | All summaries for one channel, newest-first |
| `ask_question` | `question`, `channel?`, `context_entries?` | Vector RAG: retrieves context then calls watsonx.ai |

---

## Configuration reference

| Variable | Description | Default |
|---|---|---|
| `SLACK_BOT_TOKEN` | Slack bot OAuth token (`xoxb-…`) | — |
| `WATSONX_API_KEY` | IBM Cloud API key | — |
| `WATSONX_PROJECT_ID` | watsonx.ai project ID | — |
| `WATSONX_URL` | watsonx.ai service URL | `https://us-south.ml.cloud.ibm.com` |
| `WATSONX_MODEL_ID` | Generation model | `ibm/granite-13b-instruct-v2` |
| `OPENSEARCH_URL` | OpenSearch base URL | `http://localhost:9200` |
| `OPENSEARCH_USER` | Basic-auth username | *(empty)* |
| `OPENSEARCH_PASSWORD` | Basic-auth password | *(empty)* |
| `BATCH_SIZE` | Messages accumulated per summary chunk | `50` |
| `N_CONTEXT_ENTRIES` | Context entries used in RAG Q&A | `5` |

---

## Incremental sync

Every `ingest` run records the newest Slack timestamp per channel in OpenSearch.  
The next run fetches **only messages newer than that timestamp** — keeping API usage and cost minimal.

Use `--full` to force a complete re-read of all history for any channel.

---

## Security notes

- All credentials are read from environment variables — never hardcoded.
- `.env` is excluded from version control.
- Slack user/bot mentions are anonymised to `@user` before being sent to the LLM.
- The OpenSearch index and the MCP server both run locally — no data leaves your machine except to the Slack API and watsonx.ai.
- For production deployments, enable OpenSearch TLS and use strong credentials; set `OPENSEARCH_URL` to `https://…` and provide `OPENSEARCH_USER` / `OPENSEARCH_PASSWORD`.
