# Installing AgentRecall into Hermes later

This file is informational only. It does not install, activate, or replace any current memory provider by itself.

## User-plugin install

Preferred install path from this repository checkout:

```bash
cd /path/to/AgentRecall
python scripts/install_user_plugin.py
```

That creates:

```text
$HERMES_HOME/plugins/agent-recall -> /path/to/AgentRecall
```

Start a fresh Hermes process so plugin discovery sees it, then run:

```bash
hermes memory setup agent-recall
```

For another profile:

```bash
hermes -p agentforge memory setup agent-recall
```

## Manual config example

Create or edit `$HERMES_HOME/agent-recall.json`:

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

For AgentForge, use the same `db_path` and `workspace_id`, but set:

```json
{
  "agent_id": "agentforge"
}
```

## Activating AgentRecall

Only after you intentionally want to switch the active provider:

```bash
hermes config set memory.provider agent-recall
hermes -p agentforge config set memory.provider agent-recall
```

Restart the affected Hermes sessions after changing the active provider.

## OpenAI-compatible curation

To use a normal OpenAI-compatible chat endpoint instead of the CLI backend:

```json
{
  "llm_curator_backend": "openai-compatible",
  "llm_curator_base_url": "https://api.openai.com/v1",
  "llm_curator_model": "gpt-5.3-mini",
  "llm_curator_api_key_env": "OPENAI_API_KEY"
}
```

## Turning curation off

If you want storage/retrieval without chat-model curation:

```json
{
  "llm_curator_enabled": false
}
```
