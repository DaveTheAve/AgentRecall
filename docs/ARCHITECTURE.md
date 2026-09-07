# Architecture

## Layers

```text
Host lifecycle/tool protocol
        |
        +-- Hermes: AgentRecallProvider (native, in-process)
        |      +-- optional SessionArchive adapter
        |              -> current-profile state.db FTS5 (read-only, host-owned)
        +-- OpenClaw: native memory plugin + lifecycle hooks
        +-- MCP: MCPAdapter + optional FastMCP transport
        |
        +-- OpenClaw local JSONL bridge (persistent stdio, not MCP)
        |
AgentRecallCore
  identity, policy, exclusions, memory operations, prefetch,
  curation, conclusions, synthesis, review, capture, health
        |
AgentRecallStore + EmbeddingClient + Curator
        |
SQLite WAL / OpenAI-compatible embedding endpoint / configured chat backend
```

The host-neutral `agent_recall_core.py` has no Hermes, OpenClaw, or MCP imports. `hermes_plugin/` contains the importable native Hermes adapter and packaged manifest; the root `__init__.py` is a thin source-checkout compatibility wrapper over that same implementation. `agent_recall_mcp.py` imports the MCP SDK only inside `build_fastmcp`, so normal core and Hermes use do not install or load MCP. `openclaw_plugin/` is a native OpenClaw `kind: "memory"` plugin. It keeps one local Python child alive and sends bounded JSONL requests to `agent_recall_bridge.py`, which owns a bounded LRU of core instances keyed by fixed workspace/agent/session identity.

The bridge is deliberately not MCP. MCP stays an optional public API for independent clients; the native OpenClaw path uses OpenClaw's memory capability, standard memory runtime, agent tools, and lifecycle hooks directly.

SessionArchive is deliberately outside `AgentRecallCore` and `AgentRecallStore`. The Hermes-only adapter delegates to the host's existing session-search API, preserves the active profile boundary, and never duplicates, migrates, or writes the host transcript index. Raw archive results remain untrusted historical data. This keeps AgentRecall as durable distilled knowledge while Hermes remains the owner of conversation history.

## Compatibility contract

The Hermes adapter preserves:

- direct in-process initialization and storage;
- the existing eleven Hermes tool names and JSON result shapes, plus an opt-in optional twelfth tool named `agent_recall_session_archive`;
- exact automatic prefetch block formatting;
- synchronous prefetch and lexical fallback after embedding failures;
- completed-turn capture, pre-compression checkpoints, and built-in-memory mirroring;
- configuration defaults and `$HERMES_HOME` expansion;
- workspace/agent/session ACLs and owner-only shared mutation by default;
- configured curation backend/model behavior.

Hermes never calls MCP and needs no sidecar, daemon, socket, or new runtime dependency.

## Identity and ACLs

Every `AgentRecallCore` instance has a fixed `AgentIdentity(workspace_id, agent_id, session_id, user_id)`. Adapters establish that identity at startup; individual tool calls cannot impersonate another agent.

OpenClaw maps each host agent to `openclaw:<agentId>` by default and uses the canonical `sessionKey` when available. A fixed `agentId` can be configured for intentional identity sharing. The bridge rejects missing, non-string, or oversized identity fields and evicts stale contexts when its configured bound is reached.

Visible rows satisfy:

```text
same workspace
AND not archived
AND (
  shared
  OR agent-owned by current agent
  OR session-owned by current agent and current session
)
```

Shared rows are cross-agent readable but owner-mutated unless `allow_any_agent_to_mutate_shared=true`.

## Database compatibility and concurrency

AgentRecall applies an additive, backward-compatible migration when an existing database first opens:

- `canonical_key TEXT NOT NULL DEFAULT ''` supports in-place durable fact updates;
- `expires_at REAL NOT NULL DEFAULT 0` keeps memories permanent unless an explicit Unix expiration is supplied;
- partial unique indexes keep durable canonical facts unique across sessions while retaining per-session uniqueness for `session` visibility;
- conflicting durable canonical rows are reconciled by retaining the newest row and deleting older duplicates plus orphan links.

Existing databases therefore open without an offline migration step. Test upgrades only against verified SQLite online-backup copies because opening the database applies the migration.

Each process owns a separate SQLite connection. Connections use:

- WAL journal mode;
- the existing SQLite synchronous/durability setting;
- foreign keys enabled;
- configurable `busy_timeout` (default 5 seconds, below Hermes's prefetch deadline);
- short, committed write transactions;
- an in-process re-entrant lock for shared connection/thread safety.

This supports several host processes sharing one database while SQLite serializes writes. SQLite remains a single-node/local-filesystem design; do not place the database on an unsafe network filesystem. Back up a live database with SQLite's online backup API rather than copying only the main `.db` file while WAL is active.

## Search and recall

Hybrid ranking now combines FTS5/BM25 and semantic signals using the tuned deterministic blend:

```text
0.25 BM25 rank + 0.15 lexical overlap + 0.55 vector similarity
+ 0.03 importance + 0.02 recency
+ 0.16 exact-all-query-terms lexical boost
```

The final memory ID is the deterministic secondary order for exact score ties. Embedding-free FTS candidate terms use safely quoted prefixes (for example, `postgres` matches `PostgreSQL`); this is prefix matching, not arbitrary substring matching. Embedding-backed queries use whole terms. Retrieval hard-caps the combined Python fusion set at 500 rows after ACL, scalar, and tag filters are applied in SQL. Hybrid search assigns half of that window to BM25 candidates and fills the remainder from a deterministic importance/recency-ordered vector source window; lexical-only or vector-only search can use the full window. Cosine scoring covers the bounded union of lexical and vector-source candidates, so this is approximate semantic retrieval and can miss relevant older/lower-importance memories outside both windows. SQLite may still scan or sort more than 500 rows: the cap is not a database-work or latency guarantee. Explicit search returns at most 50 results. `prefetch_context` may consume the full internal window, deduplicates canonical projections with scope preference `session > agent > shared`, records access only for rows actually appended or truncated into the final character-budgeted context, and returns a ready-to-inject context block plus a count. MCP callers must explicitly set `include_results=true` when they also need the underlying rows/provenance.

## Extension boundary

Add host-independent memory behavior to `AgentRecallCore`. Add lifecycle translation, host-specific tool schemas, commands, and installation logic to adapters. Do not put OpenClaw/Hermes/MCP names or transport assumptions into the store or core policy.
