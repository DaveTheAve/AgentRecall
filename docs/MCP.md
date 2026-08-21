# MCP Adapter

MCP is an optional public interface over `AgentRecallCore`; it is not the internal architecture and does not replace the native Hermes or OpenClaw adapters.

## Install and run

```bash
pip install '.[mcp]'

agent-recall-mcp \
  --config /path/to/agent-recall.json \
  --transport stdio \
  --workspace shared-workspace \
  --agent external-agent \
  --access read-write
```

The server identity is fixed at startup. Tool arguments cannot override `workspace_id`, `agent_id`, or `session_id`. An explicitly supplied config path must exist and contain a JSON object; startup fails closed otherwise. Session-scoped writes require a non-empty server `session_id`; an identity without one can access only agent/shared rows.

## Optional use alongside Hermes

Hermes continues to use the native provider. An MCP server can be started separately against the same config/database for external IDEs, automation, or read-only diagnostics:

```bash
agent-recall-mcp \
  --config "$HOME/.hermes/agent-recall.json" \
  --transport stdio \
  --workspace shared-workspace \
  --agent external-ide \
  --access read-only
```

Use a distinct agent identity when the client should see only workspace-shared rows. Reuse the Hermes `agent_id` only for a trusted operator client that is intentionally allowed to inspect that agent's private rows. This sidecar is additive and is not on Hermes's recall path.

## Public tools

| Tool | Access | Purpose |
| --- | --- | --- |
| `search` | read | Hybrid search with category/tag/visibility/source/importance/time filters and optional score explanations. |
| `prefetch_context` | read | Bounded explained context plus a count; ranked rows are opt-in with `include_results=true`. |
| `get_memory` | read | Inspect one visible row with provenance and ACL metadata. |
| `profile` | read | Focused identity/profile context. |
| `stats` | read | ACL-filtered buckets. |
| `health` | read | SQLite quick check, WAL/busy-timeout state, and embedding configuration status. |
| `capabilities` | read | Fixed identity, policy, enabled features, and public tools. |
| `remember` | write | Store a durable memory under the fixed identity. |
| `update` | write | Correct, retag, promote/demote, or archive an owned visible memory. |
| `forget` | write | Delete an owned visible memory. |
| `curate` | write | Extract candidates; dry-run is recommended before storage. |
| `conclude` | write | Store source-linked conclusions when enabled. |
| `profile_synthesize` | write | Synthesize configured profile scopes; dry-run by default. |
| `review` | write | Run configured conflict/staleness/promotion review; dry-run by default. |

Filesystem import is intentionally not exposed through MCP. Remote clients should not receive a general server-local file-reading primitive.

## Read-only clients

Set either:

```json
{"mcp_access": "read-only"}
```

or pass `--access read-only`. Mutation tools remain discoverable for protocol stability but fail closed with an explicit read-only error; searches also leave `access_count` and `last_accessed_at` unchanged.

## Optional OpenClaw MCP configuration

The native OpenClaw memory plugin already provides automatic recall, lifecycle capture, standard memory tools, and the extended AgentRecall tools. Add MCP only when a separately namespaced public interface is useful for administration, external automation, or interoperability:

```bash
openclaw mcp set agent-recall '{
  "command": "agent-recall-mcp",
  "args": [
    "--config", "/absolute/path/to/agent-recall.json",
    "--transport", "stdio",
    "--workspace", "shared-workspace",
    "--agent", "openclaw-mcp",
    "--access", "read-only"
  ]
}'
```

Use a distinct MCP `agent_id` unless that client is intentionally trusted to share a native OpenClaw agent's private identity. The native plugin does not call this MCP server internally.

## Authenticated Streamable HTTP

HTTP fails closed unless a bearer token is configured or the unsafe override is explicit.

```bash
export AGENT_RECALL_MCP_TOKEN='generate-a-long-random-token'
agent-recall-mcp \
  --config /path/to/agent-recall.json \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8765 \
  --auth-token-env AGENT_RECALL_MCP_TOKEN
```

Endpoint: `http://127.0.0.1:8765/mcp`

Clients send:

```text
Authorization: Bearer <token>
```

For a public tunnel, keep AgentRecall bound to loopback, tunnel only that port, configure the client with the public `/mcp` URL, and set the same bearer credential. Do not pass `--allow-unauthenticated-http` on a public or shared interface.

## Security notes

- Use a separate AgentRecall config/identity per MCP client trust boundary.
- Prefer `read-only` for IDEs, analytics, or untrusted agents.
- The token protects transport access; AgentRecall ACLs still protect row visibility and mutation.
- Public results omit session identifiers and redact absolute server-path metadata.
- Keep secrets in environment variables, not `agent-recall.json`.
- Public HTTP exposes local memory contents to whoever has the token; rotate it after accidental disclosure.
