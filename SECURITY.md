# Security Policy

AgentRecall stores memory locally in SQLite and calls only endpoints or local runtimes that are explicitly configured by the user.

## Runtime network behavior

AgentRecall has two model-facing paths:

1. Embeddings for storage/search:
   - `POST <embedding_base_url>/embeddings`

2. Chat-model curation, when `llm_curator_enabled=true`:
   - `codex-cli`: shells out to the configured `llm_curator_command`.
   - `openai-compatible`: calls `POST <llm_curator_base_url>/chat/completions`.

The curation backend, model, base URL, command, API-key env var, and timeout are configurable. Do not put API keys in this repository or in committed config examples.

## Stored data

The SQLite database can contain sensitive user preferences, operational details, and imported notes. Treat it as private application data:

- Back it up only to trusted storage.
- Do not commit `.db`, `.sqlite`, or runtime config files.
- Use `excluded_terms` for projects/domains that must never enter memory.
- Prefer `dry_run=true` for new curation workflows before storing generated candidates.

## Reporting issues

Please report security issues privately to the maintainer instead of opening a public issue with exploit details.

Include:

- AgentRecall version/commit.
- Hermes Agent version.
- Sanitized config shape, excluding secrets.
- Steps to reproduce.
- Expected and actual behavior.
