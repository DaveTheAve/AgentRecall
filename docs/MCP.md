# AgentRecall MCP boundary

The optional MCP adapter exposes a fixed server identity, not arbitrary workspace,
agent, or session routing. Install the `mcp` extra using the installation guide.
Run `python -m agent_recall_mcp --config /path/to/agent-recall.json` for stdio.
Stdout is reserved for newline-delimited JSON-RPC; diagnostics belong on stderr.

## Permissions and discovery

The default is **read-only**, including when no access setting is supplied.
`--access read-write` (or config `mcp_access: "read-write"`) explicitly enables
writes. Config `mcp_tools` is an optional list of public operation names; its
intersection with the access policy is both the discovery list and execution
allowlist. An empty list exposes nothing. Unknown names are configuration errors.
Permissions are checked before admission and again when queued work executes.
No request may override workspace/agent/session identity. Nested metadata identity
keys are rejected too. Startup `--workspace`, `--agent`, and `--session` select the
fixed identity; they are administrator settings, never tool arguments.

Read tools: `search`, `prefetch_context`, `get_memory`, `profile`, `stats`, `health`,
`capabilities`. Read-only recall does not update access counts or last-access time.
Write tools: `remember`, `update`, `forget`, `curate`, `conclude`,
`profile_synthesize`, `review`. Even model-operation dry runs require write access:
they can invoke a configured external model and perform contextual recall.
Optional operations still require their respective core feature settings.
`curate`, `profile_synthesize`, and `review` default to dry run.

Discovery includes input/output JSON schemas and read-only, destructive,
idempotent, and open-world annotations. Capabilities report the same permitted
operations and do not advertise disabled-by-policy model features.
`profile_synthesize` and `review` advertise `destructiveHint: true`: persisted
curator candidates can canonical-upsert an existing row. The dry-run default does
not remove that possibility, and these annotations do not change core semantics.

## Contracts and public output

Inputs are validated without coercion: booleans are not integers, numeric strings
are not numbers, and NaN/infinity are invalid. Unknown top-level tool fields are
rejected. Limits include 65,536 serialized argument bytes, depth 8, 1,024 JSON
nodes, 64 elements per collection, text up to 12,000 characters, positive memory
IDs, search limits 1–50, and bounded numeric weights. Individual schemas can impose
smaller limits. Explicit nulls are not substitutes for omitted optional fields.
The authoritative machine-readable contracts are in
`agent_recall_mcp_contracts.py` and returned by `tools/list`.

Responses use explicit allowlisted projection, not heuristic secret redaction.
Memory content/title/summary remain opaque user data; deliberately stored text is
not rewritten. Arbitrary metadata, session IDs, source paths, embedding warnings,
private diagnostic state, and undeclared fields are not exposed. Approved metadata
contains only typed module/scope/source-ID/provenance fields. Approved output that
exceeds its schema or the 1 MiB response budget fails safely instead of being
silently clipped. Read APIs can therefore fail safely on oversized legacy data.

| Operation | Public success payload beyond `success` |
| --- | --- |
| remember, conclude | id, action, visibility |
| update / forget | updated / deleted |
| get_memory | memory |
| search | results, count |
| prefetch_context | context, results, count, identity |
| profile | workspace_id, agent_id, recall |
| stats | total and typed visibility/category/agent buckets |
| health | sanitized startup SQLite/embedding snapshot and runtime status |
| capabilities | engine, identity, access, tools, operations, features, visibility |
| curate | stored plus candidates (dry run) or write results |
| profile_synthesize | stored, scope, subject, candidates or write results |
| review | stored, enabled_modules, recommendations or write results |

MCP tool failures have `isError: true` and matching text/structured content:
`{"success":false,"error":{"code":"backend_error","message":"Operation failed."}}`.
Stable codes are `invalid_arguments`, `forbidden`, `backend_error`, `not_found`,
`busy`, `timeout`, and `closed`. Exceptions and unknown tool names are never echoed.
Every registered SDK request handler is guarded, including prompts, resources,
subscriptions, and completion. RPC failures retain existing MCP error codes but
replace messages with `Operation failed.` and discard error data; unexpected
exceptions use the protocol internal-error code. Tool error text and structured
content are reconstructed from public constants. Valid results, initialization,
ping, and discovery retain their protocol shapes. This protection also applies to
RPC errors carried inside HTTP 200 SSE responses, not just HTTP error statuses.

## HTTP transport

Use `--transport streamable-http --host 127.0.0.1 --port 8765`. HTTP requires a
bearer token in `AGENT_RECALL_MCP_TOKEN`; `--auth-token-env NAME` chooses another
variable. The explicit `--allow-unauthenticated-http` escape hatch disables only
authentication, not body limits or Host/Origin validation. Do not expose it publicly.
Tokens are compared in constant time. Duplicate authorization headers fail closed.

Both Streamable HTTP and legacy SSE applications are wrapped in the same pure-ASGI
boundary. Authentication precedes body parsing. Host and Origin use the SDK's
loopback allowlist with additional authority syntax and duplicate-header checks;
userinfo, path-bearing origins, and wildcard-port prefix tricks are rejected.
Non-loopback virtual hosts are not configurable through this CLI. A trusted reverse
proxy must preserve a permitted loopback authority, validate its public Origin,
and provide TLS; this is not a multi-tenant OAuth authorization service.

Bodies are counted incrementally, including chunked requests: maximum 131,072
bytes, with a 10-second body-read deadline. Duplicate JSON keys, invalid envelopes,
malformed client requests/notifications, excessive nesting, and invalid JSON are
rejected before SDK parsing can expose Pydantic input values. HTTP errors use stable
public codes such as `unauthorized`, `invalid_host`, `invalid_origin`,
`invalid_request`, `request_too_large`, and `request_timeout`. SDK HTTP error bodies
are replaced rather than passed through. Successful SSE/protocol responses remain
unmodified. The stdio reader applies the same frame size and raw-message validation
and recovers at the next newline after an oversized frame.

## Runtime bounds and cancellation

`build_fastmcp` accepts `workers` (default 2, range 1–32), `queue_capacity`
(default 2, range 0–128), `operation_timeout` (default 30 seconds), and
`shutdown_timeout` (default 1 second). Deadlines must be positive and at most
300 seconds. These are Python API settings, not CLI flags.

Admission is bounded by workers plus queued slots. A deadline includes queue time.
A timed-out or cancelled queued operation never executes. **An already-running
operation retains its slot until the actual backend work finishes**, even after
its caller disconnects, cancels, or receives a timeout. Saturated admission returns
`busy`. `health` and `capabilities` bypass worker admission; health is explicitly a
startup snapshot rather than a fresh database lock acquisition.

Shutdown rejects admission, cancels queued work, and waits only its configured
budget. Core cleanup runs after workers actually finish. Daemon workers avoid an
unbounded interpreter-exit join. Python cannot forcibly abort a running database
write or network operation: timeout/cancellation is not rollback, and an in-flight
write may complete. Confirm state before retrying non-idempotent operations. Hard
process termination may interrupt work; use database recovery and idempotency
rather than assuming the client received a definitive outcome.

## v0.3.0 upgrade and memory lifecycle

Back up existing stores using SQLite online backup before testing a new release;
restore and verify the backup on a disposable database first. Do not point a
version comparison at a live store. Existing clients that wrote through MCP
must now explicitly select `--access read-write` and fit the strict request and
public response schemas; arbitrary response metadata is no longer returned.

`remember` accepts an optional `canonical_key` for a scoped canonical upsert and
an optional `expires_at` timestamp. `update` can change `expires_at` or clear it
with `0`; expired memories are excluded from recall. Expiration is logical until
an authorized core maintenance operation performs physical cleanup. MCP does not
expose arbitrary administrative maintenance or raw session archive routing.
