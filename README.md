# AgentRecall

> Durable, local-first semantic memory for Hermes Agent. Your agents can now remember the important stuff, forget the embarrassing stuff on purpose, and stop treating every new session like they just woke up in a hedge.

AgentRecall is a Python memory-provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It gives Hermes profiles and agent processes a local, inspectable memory store with explicit workspace, agent, session, and shared-memory boundaries.

The memories live in SQLite. You can inspect them, back them up, delete them, and glare at them when an agent remembers the exact wrong thing. Nothing vanishes into an opaque vendor-side memory swamp.

Status: alpha, production-oriented. The project has tests, access controls, import restrictions, configuration validation, and a healthy distrust of magic.

## The short version

AgentRecall combines:

- SQLite for durable memory rows and ACL metadata.
- OpenAI-compatible embeddings for semantic retrieval, plus lexical scoring because exact words occasionally deserve a little respect.
- An optional chat-model curation backend that turns a wall of messy text into candidate memories instead of turning it into a new religion.
- Hermes provider tools for storing, searching, reviewing, importing, and curating memories.

It is designed for people who want several agents to share useful context without making every agent's private notes public. A daring concept.

## What it is for

Use AgentRecall when you want to:

- Share durable facts across several Hermes profiles in one workspace.
- Keep agent-specific memories private while allowing intentionally shared facts.
- Store user preferences, environment conventions, project facts, decisions, and source-backed conclusions.
- Retrieve relevant context automatically through Hermes memory prefetch.
- Curate a long conversation or note into reviewable durable-memory candidates.
- Import Markdown notes from explicitly allowed locations.
- Back up your memory database without asking an AI vendor for a boat and a search party.

It is not intended to be a secret store, a complete transcript archive, or an excuse to save every thought an agent has at 3:17 a.m.

## Design rules

- Local-first storage with explicit network endpoints.
- Clear workspace, agent, session, and shared-memory isolation.
- Configurable curation. No hard-coded loyalty to a particular model, provider, or mystical moon phase.
- Simple installation as a Hermes user plugin.
- Conservative import and auto-capture defaults.
- Public-repo hygiene: MIT license, security policy, contributing guide, CI, tests, and no committed credentials. Because leaking an API key is a memorable event in the bad sense.

## Architecture

```text
Hermes memory provider API
        |
        v
AgentRecallProvider
        |
        +-- AgentRecallStore  -> SQLite memory rows + ACL filters
        +-- EmbeddingClient   -> POST <embedding_base_url>/embeddings
        +-- Curation backend  -> codex-cli or OpenAI-compatible chat completions
```

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
hermes -p agentforge memory setup agent-recall
```

Installing the files does not activate the provider. Hermes only uses AgentRecall after `memory.provider` is set to `agent-recall`. This is intentional. Surprise memory systems have never been a universally beloved product category.

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

Profiles that should share the same database and workspace use the same `db_path` and `workspace_id`, but each profile gets its own `agent_id`:

```json
{
  "agent_id": "agentforge"
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
| `raw_memories_enabled` | Enables direct `agent_recall_remember` writes. |
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
python -m py_compile __init__.py agent_recall_store.py agent_recall_curator.py cli.py scripts/install_user_plugin.py tests/*.py
```

The test suite covers workspace isolation, private/shared/session visibility, shared-mutation policy, tool behavior, curation configuration, config normalization, checkpoint hooks, built-in-memory mirroring, Markdown import restrictions, embeddings endpoint behavior, excluded terms, and public wording. The tests are serious even if this README occasionally needs to lie down.

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
