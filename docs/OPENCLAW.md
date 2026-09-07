# OpenClaw Integration

AgentRecall is a native OpenClaw memory plugin. It uses OpenClaw's exclusive memory slot and lifecycle APIs directly; MCP is optional and is not the adapter's internal transport.

## Verified OpenClaw surfaces

This adapter was loader-tested against the locally installed OpenClaw `2026.5.22` and the current npm release `2026.7.1-2`, while checking the current OpenClaw documentation on August 18, 2026. It uses these public surfaces:

- native `openclaw.plugin.json` manifest with `kind: "memory"`;
- `registerMemoryCapability` for memory prompt guidance and the standard memory runtime;
- `memory_search` and `memory_get`, plus native write/review tools;
- `before_prompt_build` for automatic recalled-context injection;
- `agent_end` for optional completed-turn capture;
- `before_compaction` and `before_reset` for optional checkpoints;
- `session_start` / `session_end` for identity lifecycle and bridge cleanup;
- `gateway_stop` for deterministic child-process shutdown.

Authoritative references:

- <https://docs.openclaw.ai/plugins/manifest>
- <https://docs.openclaw.ai/plugins/hooks>
- <https://docs.openclaw.ai/plugins/building-plugins>
- <https://docs.openclaw.ai/plugins/sdk-testing>
- <https://docs.openclaw.ai/concepts/memory>

## Architecture

OpenClaw is Node.js and AgentRecall Core is Python. Reimplementing storage, ranking, ACLs, curation, or mutation policy in JavaScript would create two memory engines. Instead, the native plugin keeps one local Python child process alive and exchanges bounded newline-delimited JSON over stdio:

```text
OpenClaw memory slot / hooks / tools
                |
      openclaw_plugin/index.js
                |
 persistent local stdio JSONL bridge
                |
     agent_recall_bridge.py
                |
        AgentRecallCore / Store
```

This bridge is not MCP, does not listen on a socket, and does not expose a network service. It maintains a bounded LRU of core instances keyed by workspace/agent/session identity and closes them on session release or gateway shutdown. The plugin package includes the Python modules and sets `PYTHONPATH` only for its child process, so a managed or linked OpenClaw installation does not require a separately published Python package.

## Install

Prerequisites: OpenClaw `>=2026.5.22`, Node.js `>=22.19`, and Python `>=3.10` available as `python3` (or set plugin `pythonCommand`). Create the JSON file passed to `--config-path` before installation; explicit adapter config paths fail closed when missing or malformed.

From this repository:

```bash
python3 scripts/install_openclaw_plugin.py \
  --config-path ~/.agent-recall/agent-recall.json \
  --workspace-id shared-workspace
```

Then restart the OpenClaw gateway. Pass `--copy` for a self-contained managed copy instead of a development symlink; the npm package includes the Python core and bridge modules needed by that copy.

If the npm package was installed into another project with `npm install agent-recall`, invoke the bundled installer directly and request a managed copy:

```bash
python3 node_modules/agent-recall/scripts/install_openclaw_plugin.py --copy \
  --config-path ~/.agent-recall/agent-recall.json \
  --workspace-id shared-workspace
```

The plugin intentionally starts a local Python child process. OpenClaw's static installer therefore reports `node:child_process` and requires `--dangerously-force-unsafe-install`. The installer supplies that explicit operator acknowledgement and does not hide or bypass the scan. Review `openclaw_plugin/bridge-client.js` before installation if this checkout is not trusted.

The installer also configures:

```text
plugins.entries.agent-recall.hooks.allowConversationAccess = true
plugins.entries.agent-recall.hooks.allowPromptInjection = true
plugins.entries.agent-recall.config.configPath = <path>
plugins.entries.agent-recall.config.workspaceId = <workspace>
plugins.entries.agent-recall.config.agentIdPrefix = openclaw
plugins.slots.memory = agent-recall  # selected automatically by install
```

Verify the live loader after restarting:

```bash
openclaw plugins inspect agent-recall --runtime --json
openclaw plugins doctor
openclaw memory status
```

The runtime inspection should report `status: "loaded"`, `memorySlotSelected: true`, ten tools, and typed hooks including `before_prompt_build`, `agent_end`, and `before_compaction` with no diagnostics.

## AgentRecall configuration

The Python config controls the database, embeddings, curation, ACLs, and capture policy. Example shared with Hermes:

```json
{
  "db_path": "~/.hermes/shared-memory/agent-recall.db",
  "workspace_id": "shared-workspace",
  "embedding_base_url": "http://127.0.0.1:8000/v1",
  "embedding_model": "your-embedding-model",
  "shared_recall": true,
  "raw_memories_enabled": true,
  "auto_capture_turns": true,
  "auto_capture_compression_checkpoints": true,
  "sqlite_busy_timeout_ms": 5000
}
```

The OpenClaw plugin's `workspaceId` is host-authoritative and overrides the config's identity for bridge calls. The database and memory policy still come from the JSON file.

## Identity mapping

By default:

```text
workspace_id = plugin config workspaceId
agent_id     = "openclaw:" + OpenClaw ctx.agentId
session_id   = OpenClaw ctx.sessionKey, falling back to ctx.sessionId
```

This means `openclaw:main` and `hermes` can read shared rows in the same workspace while retaining separate private rows. Set plugin `agentId` only when all OpenClaw agents should deliberately share one private identity.

The bridge rejects missing, non-string, or oversized identities. Tool arguments cannot override identity fields.

## Native behavior

### Automatic recall

`before_prompt_build` sends the current prompt to `prefetch_context` and returns AgentRecall's bounded context as `prependContext`. Failure is logged and skipped so memory outages do not block the model turn. The character budget is capped by both `maxContextChars` and OpenClaw's resolved context-token budget.

### Completed turns

`agent_end` extracts the last user and assistant text and calls the core capture path. AgentRecall's own `auto_capture_turns` setting remains authoritative; disabling it in the Python config prevents storage even if the OpenClaw hook remains enabled.

### Compaction and reset

`before_compaction` and `before_reset` pass the prepared messages to the existing core checkpoint operation. AgentRecall's `auto_capture_compression_checkpoints` setting remains authoritative.

### Memory runtime

The registered memory runtime powers OpenClaw's standard memory host surface:

- hybrid `search` maps agent/shared rows to `agent-recall://memory/<id>` citations;
- the host-wide runtime intentionally excludes session-scoped rows because OpenClaw's shared `readFile` manager receives no caller-session context;
- session memory remains available through automatic lifecycle recall and context-bound native tools;
- `readFile` resolves only that virtual URI scheme using an agent-level identity;
- `status` and embedding probes expose health without exposing the server database path;
- no filesystem memory corpus is invented or duplicated.

## Native tools

Required tools:

```text
memory_search
memory_get
memory_store
memory_forget
agent_recall_update
agent_recall_curate
agent_recall_conclude
agent_recall_profile
agent_recall_review
agent_recall_stats
```

In v0.3, `memory_store` accepts optional `canonicalKey` and `expiresAt` fields. Reusing a canonical key updates the stable fact in place; durable agent/shared keys span sessions while session-visible keys remain session-specific. Omit `expiresAt` or pass `0` for a permanent memory. `agent_recall_update` also accepts `expiresAt`, including `0` to clear a prior expiration.

Physical cleanup remains a trusted direct-store maintenance operation rather than an OpenClaw tool because it performs destructive hard deletion and optional database compaction.

All operations delegate to AgentRecall Core. Workspace/agent/session ACLs, exclusions, shared-owner mutation rules, embedding fallback, and curation feature flags therefore match Hermes and MCP. OpenClaw curation defaults to `dry_run=true`; pass `dry_run=false` explicitly to store candidates.

## Optional MCP

OpenClaw can additionally configure the AgentRecall MCP server for a separate read-only administrative identity or external interoperability. This is additive and normally unnecessary for recall because the native plugin already owns the memory slot and lifecycle hooks. See [MCP.md](MCP.md).

## Security and reliability

- The bridge uses argument-vector process spawning with `shell: false`.
- The child receives no network listener, runs the canonical packaged bridge script from the plugin root, and gets a replaced plugin-only `PYTHONPATH`.
- Requests are bounded to 1 MB; identity fields and live core contexts are bounded.
- Result rendering removes session identifiers and database/server filesystem paths while preserving virtual `agent-recall://` citations.
- Prompt recall failures fail open; write tools still return explicit errors.
- Conversation access is required only for `agent_end` turn capture and is explicitly enabled during installation.
- SQLite WAL and `busy_timeout` support concurrent Hermes/OpenClaw/MCP processes sharing one local database.

## Disable or remove

Switch back to OpenClaw's built-in memory slot before uninstalling:

```bash
openclaw config set plugins.slots.memory '"memory-core"' --json
openclaw plugins uninstall agent-recall
```

This does not delete the AgentRecall SQLite database.
