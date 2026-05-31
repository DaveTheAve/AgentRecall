# AgentRecall

AgentRecall is a durable semantic memory provider for Hermes Agent. It gives Hermes profiles and agent processes a local, auditable memory store with explicit workspace, agent, session, and shared-memory boundaries.

It combines three pieces:

- SQLite storage for durable memory rows and access-control metadata.
- OpenAI-compatible embeddings for semantic retrieval, with lexical scoring as a companion signal.
- A configurable chat-model curation backend for turning messy text into clean durable memory candidates.

The default curation backend is `codex-cli` with model `gpt-5.3-mini`, because that works well with an existing Codex OAuth login. AgentRecall also supports normal OpenAI-compatible `/chat/completions` endpoints through the `openai-compatible` backend. These are configuration choices, not code assumptions: change the backend, command/base URL, model, API-key environment variable, or disable curation entirely in `agent-recall.json`.

Status: alpha, production-oriented. The storage layer, ACL behavior, provider tools, curation path, config normalization, import restrictions, and endpoint behavior are covered by unit tests.

## What AgentRecall is for

AgentRecall is for Hermes users who want memory they can inspect, back up, share across profiles, and reason about operationally.

Typical uses:

- Keep one shared memory workspace across several Hermes profiles.
- Let each agent keep private memories while still participating in shared facts.
- Store durable user preferences, environment conventions, project facts, and decisions.
- Retrieve relevant memory automatically through Hermes prefetch.
- Curate long text into memory candidates with a configurable chat model.
- Import markdown notes into memory under explicit import roots.

AgentRecall is intentionally transparent: memories are SQLite rows with metadata, not opaque provider-side state.

## Design goals

- Clear isolation between workspace, agent, session, and shared facts.
- Configurable curation instead of hardcoded model/provider behavior.
- Local-first storage with explicit network endpoints.
- Simple installation as a Hermes user plugin.
- Public-repo hygiene: license, security policy, contributing guide, CI, tests, and no committed secrets.
- Safe defaults for multi-agent use: private memories are private, shared memories are owner-mutable by default, and markdown imports are constrained.

## Architecture

```text
Hermes memory provider API
        |
        v
AgentRecallProvider
        |
        +-- AgentRecallStore        -> SQLite memory rows + ACL filters
        +-- EmbeddingClient        -> POST <embedding_base_url>/embeddings
        +-- Chat curation backend  -> codex-cli or OpenAI-compatible chat completions
```

Retrieval stores embeddings for semantic search and also scores lexical matches. Curation is separate from retrieval: the chat-model backend proposes memory candidates, and AgentRecall stores accepted candidates through the same memory pipeline as explicit `agent_recall_remember` calls.

## Installation

From the repository checkout:

```bash
cd /path/to/AgentRecall
python scripts/install_user_plugin.py
```

That creates a symlink:

```text
$HERMES_HOME/plugins/agent-recall -> /path/to/AgentRecall
```

Then restart Hermes and configure the memory provider:

```bash
hermes memory setup agent-recall
```

For a named profile:

```bash
hermes -p agentforge memory setup agent-recall
```

Installing files does not activate the provider. Hermes only uses AgentRecall after `memory.provider` is set to `agent-recall` through setup or config.

## Configuration

AgentRecall stores config in:

```text
$HERMES_HOME/agent-recall.json
```

Example for a local Hermes profile:

```json
{
  "db_path": "$HERMES_HOME/shared-memory/agent-recall.db",
  "workspace_id": "shared-workspace",
  "agent_id": "hermes",
  "embedding_base_url": "http://127.0.0.1:6660/v1",
  "embedding_model": "qwen3-embedding-4b",
  "embedding_api_key_env": "LLM_OPENAI_API_KEY",
  "embedding_dimensions": 0,
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
  "excluded_terms": ["example-sensitive-project"]
}
```

For another Hermes profile that should share the same database and workspace, keep `db_path` and `workspace_id` the same but change `agent_id`:

```json
{
  "agent_id": "agentforge"
}
```

Important config fields:

| Key | Purpose |
| --- | --- |
| `db_path` | SQLite database path. Use the same path for profiles that should share `shared` memories. |
| `workspace_id` | Memory namespace. Agents only see rows from their current workspace. |
| `agent_id` | Identity used for private/session ownership. |
| `embedding_base_url` | OpenAI-compatible base URL for embeddings. |
| `embedding_model` | Embedding model name. |
| `embedding_dimensions` | Positive value sends `dimensions`; `0` omits the parameter. |
| `raw_memories_enabled` | Enables explicit `agent_recall_remember` storage. Defaults to `true`. |
| `curated_memories_enabled` | Enables `agent_recall_curate` as a feature layer. Defaults to `true`. |
| `conclusions_enabled` | Enables inspectable conclusions with source provenance. Defaults to `false`. |
| `peer_profiles_enabled` | Enables peer profile synthesis. Defaults to `false`. |
| `workspace_profiles_enabled` | Enables workspace profile synthesis. Defaults to `false`. |
| `agent_profiles_enabled` | Enables agent profile synthesis. Defaults to `false`. |
| `dialectic_review_enabled` | Enables bounded review jobs. Defaults to `false`. |
| `conflict_detection_enabled` | Adds conflict detection instructions to review jobs. Defaults to `false`. |
| `staleness_detection_enabled` | Adds stale-memory detection instructions to review jobs. Defaults to `false`. |
| `promotion_rules_enabled` | Adds promotion recommendations to review jobs. Defaults to `false`. |
| `demotion_rules_enabled` | Adds demotion/archive recommendations to review jobs. Defaults to `false`. |
| `llm_curator_enabled` | Enables or disables chat-model memory curation and synthesis model calls. Defaults to `true`. |
| `llm_curator_backend` | Curation backend selector. Implemented: `codex-cli`, `openai-compatible` (alias: `chat-completions`). |
| `llm_curator_model` | Chat model passed to the curation backend. Default: `gpt-5.3-mini`. |
| `llm_curator_command` | CLI command for the `codex-cli` backend. Default: `codex`. |
| `llm_curator_base_url` | Base URL for the `openai-compatible` backend, for example `https://api.openai.com/v1`. |
| `llm_curator_api_key_env` | Environment variable that holds the chat API key for `openai-compatible`. |
| `auto_capture_turns` | Store completed conversation turns as transcript memories. Defaults to `false`. |
| `auto_capture_compression_checkpoints` | Store pre-compression checkpoints. Defaults to `false`. |
| `allow_any_agent_to_mutate_shared` | If `false`, shared memories are editable/deletable only by their owner. |
| `excluded_terms` | Case-insensitive terms that block storage/import/curation. |

Environment overrides are supported for common deployment fields:

- `AGENT_RECALL_WORKSPACE`
- `AGENT_RECALL_AGENT`
- `AGENT_RECALL_EMBEDDING_BASE_URL`
- `AGENT_RECALL_EMBEDDING_MODEL`
- `AGENT_RECALL_EMBEDDING_API_KEY_ENV`
- `AGENT_RECALL_EMBEDDING_DIMENSIONS`
- `AGENT_RECALL_LLM_CURATOR_BACKEND`
- `AGENT_RECALL_LLM_CURATOR_MODEL`
- `AGENT_RECALL_LLM_CURATOR_COMMAND`
- `AGENT_RECALL_LLM_CURATOR_BASE_URL`
- `AGENT_RECALL_LLM_CURATOR_API_KEY_ENV`

## Chat-model curation

`agent_recall_curate` takes source text, asks the configured curation backend for JSON memory candidates, and stores or previews the results.

Example tool behavior:

```text
agent_recall_curate(text="Long conversation or notes...", default_visibility="agent", dry_run=true)
```

The backend is configurable so users can choose the model/runtime that fits their environment.

Built-in backends:

- `codex-cli`: runs the configured command with `codex exec`-compatible arguments.
- `openai-compatible`: calls `<llm_curator_base_url>/chat/completions` with the configured model and API-key environment variable.

Example OpenAI-compatible curation config:

```json
{
  "llm_curator_backend": "openai-compatible",
  "llm_curator_base_url": "https://api.openai.com/v1",
  "llm_curator_model": "gpt-5.3-mini",
  "llm_curator_api_key_env": "OPENAI_API_KEY"
}
```

Turn curation off:

```json
{
  "llm_curator_enabled": false
}
```

Curation is useful for distilling durable facts from noisy text, but AgentRecall still enforces the same storage rules afterward: excluded terms, visibility, metadata validation, embeddings, and ACLs all apply to curated candidates.

## Optional intelligence modules

Honcho-style intelligence is modeled as explicit modules rather than hidden background behavior. Each module is inspectable and disableable through config.

- Raw memories: `raw_memories_enabled`; stores explicit facts via `agent_recall_remember`.
- Curated memories: `curated_memories_enabled` plus `llm_curator_enabled`; extracts durable candidates via `agent_recall_curate`.
- Conclusions: `conclusions_enabled`; stores source-backed conclusions via `agent_recall_conclude` with `source_ids`, `supersedes`, confidence, scope, and subject metadata.
- Peer profiles: `peer_profiles_enabled`; synthesizes peer profile candidates via `agent_recall_profile_synthesize`.
- Workspace profiles: `workspace_profiles_enabled`; synthesizes workspace profile candidates via `agent_recall_profile_synthesize`.
- Agent profiles: `agent_profiles_enabled`; synthesizes agent profile candidates via `agent_recall_profile_synthesize`.
- Dialectic review jobs: `dialectic_review_enabled`; runs bounded review via `agent_recall_review`.
- Conflict/staleness detection: `conflict_detection_enabled` and `staleness_detection_enabled`; add those review instructions when `agent_recall_review` runs.
- Promotion/demotion rules: `promotion_rules_enabled` and `demotion_rules_enabled`; add promotion/demotion recommendations when `agent_recall_review` runs.

Model-use guarantee: every model-using intelligence module routes through the same configured curation backend and `llm_curator_model`. Nothing in these modules forces local Qwen or couples itself to the main Hermes/AgentForge chat model. Use `dry_run=true` to inspect synthesized candidates or review recommendations before storage.

## Isolation model

Every memory row stores:

- `workspace_id`
- `agent_id`
- `source_agent_id`
- `user_id`
- `session_id`
- `visibility`

Visibility values:

- `agent`: visible only to the owning agent in the workspace.
- `shared`: readable by all agents in the workspace.
- `session`: visible only to the owning agent in the same session.

Automatic recall enforces:

```text
workspace_id == current workspace
AND (
  visibility == 'shared'
  OR (visibility == 'agent' AND agent_id == current agent)
  OR (visibility == 'session' AND agent_id == current agent AND session_id == current session)
)
```

Mutation policy:

- `agent` and `session` memories are owner-only.
- `shared` memories are readable cross-agent.
- `shared` memories are editable/deletable by their owner by default.
- Set `allow_any_agent_to_mutate_shared=true` only when all participating agents are trusted to edit shared facts.

## Tools

AgentRecall exposes these Hermes tools:

- `agent_recall_remember`: store an explicit durable memory.
- `agent_recall_search`: search visible memories with semantic and lexical scoring.
- `agent_recall_profile`: return current workspace/agent info plus relevant memories.
- `agent_recall_update`: update, retag, promote/demote, or archive a memory subject to ACLs.
- `agent_recall_forget`: delete a memory subject to ACLs.
- `agent_recall_curate`: use the configured chat-model backend to propose and store durable memories from source text.
- `agent_recall_conclude`: store inspectable conclusions with explicit provenance and source IDs.
- `agent_recall_profile_synthesize`: synthesize peer, workspace, or agent profile candidates through the configured chat-model backend.
- `agent_recall_review`: run bounded dialectic review for conflicts, staleness, and promotion/demotion recommendations.
- `agent_recall_stats`: show visible memory counts without leaking other agents' private rows.
- `agent_recall_import_markdown`: chunk and ingest markdown notes under configured import roots.

## Auto-capture

Auto-capture is intentionally conservative:

- `auto_capture_turns` defaults to `false`.
- `auto_capture_compression_checkpoints` defaults to `false`.

When enabled, completed turns or compression checkpoints are stored as session-scoped transcript memories unless configured otherwise. This is useful for transcript recall, but it can grow the database quickly. For production memory hygiene, prefer explicit `agent_recall_remember` or `agent_recall_curate` for durable facts.

## Markdown import safety

`agent_recall_import_markdown` is constrained by default:

- Only `.md` files are imported.
- Symlinks are rejected.
- Paths must be under configured `import_roots`.
- Files larger than `max_import_bytes` are rejected.
- `excluded_terms` are enforced before storage.

This keeps imports deliberate and avoids accidentally ingesting arbitrary local files.

## Excluded terms

`excluded_terms` blocks storage of content matching configured terms. Use it for retired projects, sensitive domains, or any phrase that should never enter memory.

Example:

```json
{
  "excluded_terms": ["example-sensitive-project", "legacy-sensitive-domain"]
}
```

## Development

Run tests:

```bash
uv run --with pytest python -m pytest tests -q
```

Run lint:

```bash
uv run --with ruff ruff check .
```

Compile check:

```bash
python -m py_compile __init__.py agent_recall_store.py agent_recall_curator.py cli.py scripts/install_user_plugin.py tests/*.py
```

Current test coverage includes:

- workspace isolation
- shared/private/session visibility
- owner-only shared mutation policy
- optional relaxed shared mutation policy
- provider tool behavior
- chat-model curation default/config/off-switch behavior
- config normalization
- pre-compression checkpoint hook
- built-in memory mirroring hook
- markdown import restrictions and chunking
- embedding client endpoint behavior
- excluded-term blocking
- public metadata wording regressions

## Production notes

Before using AgentRecall heavily:

1. Put `db_path` somewhere backed up.
2. Use stable `workspace_id` and `agent_id` values.
3. Keep `excluded_terms` current.
4. Decide whether shared memories should be owner-only or cross-agent mutable.
5. Test the embedding endpoint and curation backend with a dry run.
6. Monitor database size if enabling auto-capture.

SQLite JSON vector storage is simple and portable. For very large memory banks, a future version could add optional sqlite-vec, LanceDB, or pgvector backends.

## Security

AgentRecall does not require secrets in the repository. Runtime credentials should come from the environment or the configured CLI/runtime.

Recommended practices:

- Keep API keys in `.env` or your process manager, not in `agent-recall.json`.
- Use `embedding_api_key_env` to point at the relevant environment variable.
- Review curated candidates with `dry_run=true` before enabling high-volume workflows.
- Back up the SQLite database if memories are important.
- Treat shared memory as shared operational context, not as a secret store.

## Public release readiness

The repository includes:

- MIT license
- security policy
- contributing guide
- pytest and ruff CI workflow
- user-plugin installer
- smoke and unit tests
- no committed credentials

AgentRecall is suitable for experimentation and careful production use by Hermes users who want an inspectable memory provider. As with any memory system, start with a small deployment, validate retrieval quality, and tune visibility/configuration before relying on it broadly.
