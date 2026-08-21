# Security Policy

AgentRecall stores memory locally in SQLite and calls only endpoints or local runtimes that are explicitly configured by the user.

## Runtime network behavior

AgentRecall has three optional network-facing paths:

1. Embeddings for storage/search:
   - `POST <embedding_base_url>/embeddings`

2. Chat-model curation, when `llm_curator_enabled=true`:
   - `codex-cli`: shells out to the configured `llm_curator_command`.
   - `openai-compatible`: calls `POST <llm_curator_base_url>/chat/completions`.

3. Optional MCP HTTP transport:
   - Disabled unless explicitly started.
   - Requires bearer authentication by default; never expose `--allow-unauthenticated-http` publicly.
   - Prefer `mcp_access=read-only` for clients that do not need mutation.

The curation backend, model, base URL, command, API-key env var, and timeout are configurable. MCP bearer tokens also belong in environment variables, never config files. Do not put API keys in this repository or in committed config examples.

## OpenClaw local bridge

The native OpenClaw plugin starts one local Python child process using `node:child_process` with an argument vector and `shell: false`. It communicates over inherited stdio only and opens no listening socket. OpenClaw's installer flags all external child-process use and blocks installation by default; `scripts/install_openclaw_plugin.py` supplies the explicit `--dangerously-force-unsafe-install` acknowledgement. Review the checkout before using that option.

Bridge requests are bounded, identity fields are validated, live session/core contexts are bounded, and the child receives a plugin-scoped `PYTHONPATH`. OpenClaw conversation access is enabled for completed-turn capture; disable `autoCaptureTurns` in plugin config or `auto_capture_turns` in AgentRecall config when raw turn capture is not appropriate.

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
- Hermes Agent and/or OpenClaw version.
- Sanitized config shape, excluding secrets.
- Steps to reproduce.
- Expected and actual behavior.
