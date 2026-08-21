# Architecture

## Layers

```text
Host lifecycle/tool protocol
        |
        +-- Hermes: AgentRecallProvider (native, in-process)
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

`agent_recall_core.py` has no Hermes, OpenClaw, or MCP imports. `__init__.py` is the Hermes compatibility adapter. `agent_recall_mcp.py` imports the MCP SDK only inside `build_fastmcp`, so normal core and Hermes use do not install or load MCP. `openclaw_plugin/` is a native OpenClaw `kind: "memory"` plugin. It keeps one local Python child alive and sends bounded JSONL requests to `agent_recall_bridge.py`, which owns a bounded LRU of core instances keyed by fixed workspace/agent/session identity.

The bridge is deliberately not MCP. MCP stays an optional public API for independent clients; the native OpenClaw path uses OpenClaw's memory capability, standard memory runtime, agent tools, and lifecycle hooks directly.

## Compatibility contract

The Hermes adapter preserves:

- direct in-process initialization and storage;
- the existing eleven Hermes tool names and JSON result shapes;
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

No schema migration is introduced by the multi-host architecture. Existing AgentRecall databases open unchanged.

Each process owns a separate SQLite connection. Connections use:

- WAL journal mode;
- the existing SQLite synchronous/durability setting;
- foreign keys enabled;
- configurable `busy_timeout` (default 5 seconds, below Hermes's prefetch deadline);
- short, committed write transactions;
- an in-process re-entrant lock for shared connection/thread safety.

This supports several host processes sharing one database while SQLite serializes writes. SQLite remains a single-node/local-filesystem design; do not place the database on an unsafe network filesystem. Back up a live database with SQLite's online backup API rather than copying only the main `.db` file while WAL is active.

## Search and recall

Hybrid ranking is unchanged for normal Hermes calls:

```text
0.58 vector + 0.25 lexical + 0.10 importance + 0.07 recency
```

Adapters may request score explanations and richer ACL-safe filters without changing default result ranking. `prefetch_context` adds a character budget and returns a ready-to-inject context block plus a count; MCP callers must explicitly set `include_results=true` when they also need the underlying rows/provenance.

## Extension boundary

Add host-independent memory behavior to `AgentRecallCore`. Add lifecycle translation, host-specific tool schemas, commands, and installation logic to adapters. Do not put OpenClaw/Hermes/MCP names or transport assumptions into the store or core policy.
