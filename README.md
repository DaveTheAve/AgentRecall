# AgentRecall

> Durable, local-first memory for AI agents across native Hermes, OpenClaw, and MCP integrations.

AgentRecall is a host-agnostic Python memory engine with first-class adapters. Hermes keeps its original direct, in-process memory-provider integration. OpenClaw uses its native memory slot, prompt and session lifecycle hooks, standard memory runtime, and a persistent local stdio bridge to the same Python core. Other hosts can use the optional MCP server. Profiles and hosts can safely share one SQLite workspace while retaining distinct agent/session identities.

The memories live in SQLite. You can inspect them, back them up, delete them, and glare at them when an agent remembers the exact wrong thing. Nothing vanishes into an opaque vendor-side memory swamp.

Status: alpha, production-oriented. The project has tests, access controls, import restrictions, configuration validation, and a healthy distrust of magic.

## The short version

AgentRecall combines:

- SQLite for durable memory rows and ACL metadata.
- Deterministic FTS5/BM25 and OpenAI-compatible embedding fusion, plus embedding-free lexical prefix fallback.
- Canonical keys for updating stable durable facts in place instead of accumulating duplicates.
- Optional expiration and explicit physical cleanup while permanent memory remains the default.
- An optional chat-model curation backend that turns a wall of messy text into candidate memories instead of turning it into a new religion.
- Native host tools for storing, searching, reviewing, and curating memories; Hermes additionally exposes controlled Markdown import.

It is designed for people who want several agents to share useful context without making every agent's private notes public. A daring concept.

## What it is for

Use AgentRecall when you want to:

- Share durable facts across Hermes profiles, OpenClaw agents, and other hosts in one workspace.
- Keep agent-specific memories private while allowing intentionally shared facts.
- Store user preferences, environment conventions, project facts, decisions, and source-backed conclusions.
- Retrieve relevant context automatically through Hermes prefetch or OpenClaw's `before_prompt_build` hook.
- Curate a long conversation or note into reviewable durable-memory candidates.
- Import Markdown notes from explicitly allowed locations.
- Back up your memory database without asking an AI vendor for a boat and a search party.

It is not intended to be a secret store, a complete transcript archive, or an excuse to save every thought an agent has at 3:17 a.m.

## Design rules

- Local-first storage with explicit network endpoints.
- Clear workspace, agent, session, and shared-memory isolation.
- Configurable curation. No hard-coded loyalty to a particular model, provider, or mystical moon phase.
- Native plugin installation for Hermes and OpenClaw, with MCP as an optional interface.
- Conservative import and auto-capture defaults.
- Public-repo hygiene: MIT license, security policy, contributing guide, CI, tests, and no committed credentials. Because leaking an API key is a memorable event in the bad sense.

## Architecture

```text
                         AgentRecallCore
                 (policy, retrieval, ACL, curation)
                    /          |           \
                   /           |            \
      Hermes native adapter  OpenClaw adapter    MCP adapter
       direct / in-process   native hooks+stdio  stdio / HTTP
                   \           |            /
                    +---- AgentRecallStore -+
                           SQLite WAL
```

`AgentRecallCore` owns host-independent behavior. Adapters translate host lifecycles and tool protocols without reimplementing memory semantics. Hermes does not use MCP internally and requires no daemon or network service. OpenClaw uses a long-lived local child process rather than MCP internally, preserving MCP as an independent public interface. See [Architecture](docs/ARCHITECTURE.md), [MCP](docs/MCP.md), and [OpenClaw integration](docs/OPENCLAW.md).

Semantic retrieval uses stored embeddings and lexical scoring. Curation is separate: a chat model proposes candidate memories, then AgentRecall applies the same visibility, metadata, exclusion, embedding, and ACL rules used by explicit memory writes. The robot still has to fill out the paperwork.

## Installation

From a repository checkout:

```bash
cd /path/to/AgentRecall
python scripts/install_user_plugin.py
```

The installer creates this symlink:

```text
$HERMES_HOME/plugins/agent-recall -> /path/to/AgentRecall
```

Then restart Hermes and enable the provider:

```bash
hermes memory setup agent-recall
```

For a named profile:

```bash
hermes -p coding-agent memory setup agent-recall
```

Installing the files does not activate the provider. Hermes only uses AgentRecall after `memory.provider` is set to `agent-recall`. This is intentional. Surprise memory systems have never been a universally beloved product category.

### OpenClaw

OpenClaw has a first-class exclusive memory slot. Create the AgentRecall JSON config first; explicit adapter config paths fail closed if missing or malformed. The installer then links this tree, explicitly acknowledges OpenClaw's child-process security scan, enables the conversation/prompt hook permissions required for automatic recall and capture, and selects AgentRecall as the memory plugin:

```bash
python3 scripts/install_openclaw_plugin.py \
  --config-path ~/.agent-recall/agent-recall.json \
  --workspace-id shared-workspace
```

Restart the OpenClaw gateway after installation. OpenClaw agents default to separate identities such as `openclaw:main`; set a fixed `agentId` only when intentional. See [docs/OPENCLAW.md](docs/OPENCLAW.md).

For a registry-installed npm package, run the same packaged installer from the consuming project and copy it into OpenClaw's managed plugin area:

```bash
python3 node_modules/agent-recall/scripts/install_openclaw_plugin.py --copy \
  --config-path ~/.agent-recall/agent-recall.json \
  --workspace-id shared-workspace
```

### MCP

MCP is optional and does not affect Hermes:

```bash
# Local stdio server (installs the optional MCP SDK and console entry point)
pip install '.[mcp]'
agent-recall-mcp --config /path/to/agent-recall.json --transport stdio
```

MCP can also be configured additively in OpenClaw for external interoperability or administrative tools; it does not replace the native OpenClaw memory plugin. Never expose the HTTP server without authentication.

## Configuration

AgentRecall reads configuration from:

```text
$HERMES_HOME/agent-recall.json
```

A practical local configuration:

```json
{
  "db_path": "$HERMES_HOME/shared-memory/agent-recall.db",
  "workspace_id": "shared-workspace",
  "agent_id": "hermes",
  "embedding_base_url": "http://127.0.0.1:8000/v1",
  "embedding_model": "your-embedding-model",
  "embedding_api_key_env": "LLM_OPENAI_API_KEY",
  "embedding_dimensions": 0,
  "sqlite_busy_timeout_ms": 5000,
  "default_visibility": "agent",
  "shared_recall": true,
  "raw_memories_enabled": true,
  "curated_memories_enabled": true,
  "conclusions_enabled": false,
  "peer_profiles_enabled": false,
  "workspace_profiles_enabled": false,
  "agent_profiles_enabled": false,
  "dialectic_review_enabled": false,
  "conflict_detection_enabled": false,
  "staleness_detection_enabled": false,
  "promotion_rules_enabled": false,
  "demotion_rules_enabled": false,
  "llm_curator_enabled": true,
  "llm_curator_backend": "codex-cli",
  "llm_curator_model": "gpt-5.3-mini",
  "llm_curator_command": "codex",
  "llm_curator_base_url": "",
  "llm_curator_api_key_env": "OPENAI_API_KEY",
  "llm_curator_timeout": 120,
  "auto_capture_turns": false,
  "auto_capture_compression_checkpoints": false,
  "allow_any_agent_to_mutate_shared": false,
  "mcp_access": "read-write",
  "excluded_terms": ["example-sensitive-project"]
}
```

Replace `embedding_base_url` and `embedding_model` with the endpoint and model exposed by your embedding service.

Profiles that should share the same database and workspace use the same `db_path` and `workspace_id`, but each profile gets its own `agent_id`:

```json
{
  "agent_id": "coding-agent"
}
```

### Important settings

| Setting | What it controls |
| --- | --- |
| `db_path` | SQLite database location. Back it up if your memories are important. Future You has enough enemies. |
| `workspace_id` | Memory namespace. Agents only see rows in their current workspace. |
| `agent_id` | Identity used for private and session memory ownership. |
| `embedding_base_url` | OpenAI-compatible embeddings endpoint. |
| `embedding_model` | Embedding model to call. |
| `embedding_dimensions` | Positive values send `dimensions`; `0` omits it. |
| `sqlite_busy_timeout_ms` | SQLite lock wait used by every host process; default 5000 ms (below Hermes's prefetch deadline). |
| `raw_memories_enabled` | Enables direct `agent_recall_remember` storage. |
| `curated_memories_enabled` | Enables the curation layer. |
| `conclusions_enabled` | Enables source-backed conclusions with provenance. |
| `peer_profiles_enabled` | Enables peer-profile synthesis. |
| `workspace_profiles_enabled` | Enables workspace-profile synthesis. |
| `agent_profiles_enabled` | Enables agent-profile synthesis. |
| `dialectic_review_enabled` | Enables bounded memory-review runs. |
| `conflict_detection_enabled` | Adds conflict-detection instructions to reviews. |
| `staleness_detection_enabled` | Adds stale-memory detection to reviews. |
| `promotion_rules_enabled` | Adds promotion recommendations to reviews. |
| `demotion_rules_enabled` | Adds demotion/archive recommendations to reviews. |
| `llm_curator_enabled` | Turns model-backed curation and synthesis on or off. |
| `llm_curator_backend` | `codex-cli` or `openai-compatible` (also accepts `chat-completions`). |
| `llm_curator_model` | Model passed to the configured curation backend. |
| `llm_curator_command` | Command used by the `codex-cli` backend. |
| `llm_curator_base_url` | Base URL for an OpenAI-compatible chat endpoint. |
| `llm_curator_api_key_env` | Environment variable containing the API key for that endpoint. |
| `auto_capture_turns` | Stores completed turns as session-scoped transcripts. Defaults to `false`, because databases deserve mercy. |
| `auto_capture_compression_checkpoints` | Stores pre-compression checkpoints. Defaults to `false`. |
| `allow_any_agent_to_mutate_shared` | Lets any trusted agent edit shared facts. Default: owner-only. |
| `mcp_access` | MCP policy: `read-only` or `read-write`; server identity remains fixed at startup. |
| `excluded_terms` | Case-insensitive terms blocked from storing, importing, or curation. |

Common environment overrides are also supported:

```text
AGENT_RECALL_WORKSPACE
AGENT_RECALL_AGENT
AGENT_RECALL_EMBEDDING_BASE_URL
AGENT_RECALL_EMBEDDING_MODEL
AGENT_RECALL_EMBEDDING_API_KEY_ENV
AGENT_RECALL_EMBEDDING_DIMENSIONS
AGENT_RECALL_LLM_CURATOR_BACKEND
AGENT_RECALL_LLM_CURATOR_MODEL
AGENT_RECALL_LLM_CURATOR_COMMAND
AGENT_RECALL_LLM_CURATOR_BASE_URL
AGENT_RECALL_LLM_CURATOR_API_KEY_ENV
```

## Memory visibility

Every row records a workspace, agent, source agent, user, session, and visibility level.

| Visibility | Who can read it |
| --- | --- |
| `agent` | The owning agent in that workspace. |
| `shared` | Every agent in the workspace. Use this for facts that should travel, not for your secret lasagna recipe. |
| `session` | The owning agent during that specific session. |

Automatic recall allows only:

```text
workspace_id == current workspace
AND (
  visibility == 'shared'
  OR (visibility == 'agent' AND agent_id == current agent)
  OR (visibility == 'session' AND agent_id == current agent AND session_id == current session)
)
```

By default, private and session memories are owner-only. Shared memories are readable cross-agent but can be edited or deleted only by their owner. Set `allow_any_agent_to_mutate_shared` to `true` only when every participating agent is trusted not to turn a typo into workplace folklore.

## Canonical facts, expiration, and cleanup

AgentRecall v0.3.0 adds two optional lifecycle fields:

- `canonical_key` identifies a stable fact that should update in place. Durable `agent` and `shared` keys remain stable across sessions; `session` keys remain isolated to their session.
- `expires_at` is a Unix timestamp. Omit it or set it to `0` for a permanent memory.

For example, a host can remember a current preference with `canonical_key="preference.response_style"` and later write the same key to update the existing row instead of creating a second durable fact. Set `expires_at` only for facts with a real lifetime; expiration is never mandatory.

Expired rows are excluded from recall immediately. Physical cleanup is an explicit maintenance operation on `AgentRecallStore`, not an LLM-callable Hermes, MCP, or OpenClaw tool, because it hard-deletes rows and can optionally compact SQLite. Upgrades do not run physical cleanup automatically.

Retrieval remains bounded and deterministic: public search returns at most 50 rows, internal candidate processing is capped at 500 rows, ACL/tag/scalar filters are applied before candidate limiting, and exact score ties use memory ID as the secondary order.

The bound applies to Python candidate materialization and scoring, not SQLite's filtering/sorting work or end-to-end latency. Semantic retrieval is approximate: the vector source window is selected by importance and recency, so older or lower-importance semantic matches outside that window may be missed.

## Upgrading from v0.2.0

The v0.3.0 schema migration is additive. On first open, AgentRecall adds `canonical_key`, `expires_at`, and their supporting indexes. Existing rows receive permanent defaults, and v0.2.0 code can still open the migrated database because it ignores the added columns and indexes.

Before upgrading a production database, create a SQLite-consistent online backup rather than copying only the main `.db` file while WAL writers may be active:

```python
import sqlite3

source = sqlite3.connect("file:/path/to/agent-recall.db?mode=ro", uri=True)
backup = sqlite3.connect("/path/to/agent-recall-before-v0.3.0.db")
source.backup(backup)
assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
backup.close()
source.close()
```

Concurrent first-open migration is supported, but a controlled canary restart is easier to observe: start one upgraded host, verify health/search/write behavior, then restart the remaining hosts. Do not run physical cleanup as part of the migration.

Code rollback to v0.2.0 is supported with the additive schema left in place. If data restoration is required instead, stop every writer before replacing the database with the online backup. See [CHANGELOG.md](CHANGELOG.md) and [Architecture](docs/ARCHITECTURE.md) for details.

Schema compatibility does not preserve new feature semantics in old hosts: v0.2.0 ignores expiration and does not perform canonical upserts. Avoid mixed-version operation when expiration filtering matters.

## Curation

`agent_recall_curate` sends text to the configured chat-model backend, receives JSON memory candidates, and then runs normal AgentRecall storage rules before anything is saved.

```text
agent_recall_curate(
  text="Long conversation or notes...",
  default_visibility="agent",
  dry_run=true
)
```

The built-in backends are:

- `codex-cli`: runs the configured command with `codex exec`-compatible arguments.
- `openai-compatible`: calls `<llm_curator_base_url>/chat/completions` using the configured model and API-key environment variable.

Example OpenAI-compatible configuration:

```json
{
  "llm_curator_backend": "openai-compatible",
  "llm_curator_base_url": "https://api.openai.com/v1",
  "llm_curator_model": "gpt-5.3-mini",
  "llm_curator_api_key_env": "OPENAI_API_KEY"
}
```

Turn curation off when you prefer the agents to remember only what you explicitly save:

```json
{
  "llm_curator_enabled": false
}
```

Curation is useful for distilling noisy text. It is not a substitute for judgment. Ask any attic full of mystery boxes.

## Hermes tools

| Tool | What it does |
| --- | --- |
| `agent_recall_remember` | Stores an explicit durable memory. |
| `agent_recall_search` | Uses semantic and lexical retrieval across visible memories. |
| `agent_recall_profile` | Shows workspace/agent information and relevant memories. |
| `agent_recall_update` | Updates, retags, promotes/demotes, or archives a visible memory. |
| `agent_recall_forget` | Deletes a visible memory. Sometimes wisdom is knowing when to hit delete. |
| `agent_recall_curate` | Uses the configured chat model to extract durable candidates from text. |
| `agent_recall_conclude` | Stores source-backed conclusions with provenance. |
| `agent_recall_profile_synthesize` | Produces peer, workspace, or agent profile candidates. |
| `agent_recall_review` | Runs a bounded review for conflicts, staleness, and promotion/demotion recommendations. |
| `agent_recall_stats` | Reports visible-memory counts without exposing other agents' private rows. |
| `agent_recall_import_markdown` | Imports approved Markdown notes in chunks. |

Optional intelligence modules are explicit and configurable. Nothing quietly changes the main Hermes model, forces a local model, or starts writing a memoir about your shell history.

## Auto-capture

Auto-capture defaults to off:

```json
{
  "auto_capture_turns": false,
  "auto_capture_compression_checkpoints": false
}
```

When enabled, completed turns and compression checkpoints are stored as session-scoped transcript memories. That can be useful for debugging or transcript recall, but it can grow a database with the confidence of a sourdough starter. For durable memory hygiene, prefer explicit `agent_recall_remember` calls or curated candidates you have reviewed.

## Markdown imports

Markdown imports are deliberately constrained:

- Only `.md` files are accepted.
- Symlinks are rejected.
- Paths must sit under configured `import_roots`.
- Files larger than `max_import_bytes` are rejected.
- `excluded_terms` are enforced before storage.

Use `excluded_terms` for retired projects, sensitive domains, or any phrase you never want in AgentRecall:

```json
{
  "excluded_terms": ["example-sensitive-project", "legacy-sensitive-domain"]
}
```

The point is to import a knowledge base, not accidentally hoover up your entire home directory and give every agent a detailed opinion about `Downloads/final_final_really_final.zip`.

## Development

AgentRecall requires Python 3.10 or later.

Run tests:

```bash
uv run --with pytest python -m pytest tests -q
```

Run lint:

```bash
uv run --with ruff ruff check .
```

Run a compile check:

```bash
python -m py_compile __init__.py hermes_plugin/__init__.py agent_recall_core.py agent_recall_store.py agent_recall_curator.py agent_recall_schemas.py agent_recall_mcp.py agent_recall_bridge.py cli.py scripts/*.py tests/*.py
npm test
```

The test suite covers workspace isolation, private/shared/session visibility, shared-mutation policy, Hermes tool and lifecycle compatibility, curation configuration, config normalization, checkpoints, built-in-memory mirroring, Markdown import restrictions, embedding fallback, excluded terms, core/adapter boundaries, the OpenClaw memory-slot package and persistent bridge, MCP stdio and authenticated Streamable HTTP, and concurrent SQLite writers.

## Security and production notes

Before relying on AgentRecall heavily:

1. Put `db_path` somewhere backed up.
2. Use stable `workspace_id` and `agent_id` values.
3. Keep `excluded_terms` current.
4. Decide whether shared memories are owner-only or cross-agent mutable.
5. Test embeddings and curation with `dry_run=true`.
6. Monitor database size if auto-capture is enabled.
7. Keep runtime credentials in `.env` or your process manager, never in `agent-recall.json`.
8. Treat shared memory as shared operational context, not a vault for secrets.

SQLite JSON-vector storage is intentionally portable and easy to inspect. Very large memory banks may eventually benefit from an optional backend such as sqlite-vec, LanceDB, or pgvector. Until then, SQLite is doing what SQLite does best: quietly carrying more of civilization than anyone gives it credit for.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome. Please bring tests, avoid committing secrets, and remember that a memory system should be slightly boring in production. The funny part should stay in the README.
