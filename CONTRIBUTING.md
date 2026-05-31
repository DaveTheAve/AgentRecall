# Contributing

## Development setup

```bash
uv run --with pytest python -m pytest tests -q
uv run --with ruff ruff check .
python -m py_compile __init__.py agent_recall_store.py agent_recall_curator.py cli.py scripts/install_user_plugin.py tests/*.py
```

## Design constraints

- Preserve workspace + per-agent + per-session isolation.
- Tests must cover ACL behavior for every new mutation or recall path.
- Chat-model curation must remain configurable: backend, model, command/base URL, API-key env var, timeout, and off switch.
- New model backends need unit tests that prove request shape, config usage, failure handling, and no secret logging.
- Keep storage/retrieval and curation as separate stages so curation can be changed or disabled without changing memory ACLs.
- Keep the plugin usable as a plain Hermes user plugin under `$HERMES_HOME/plugins/agent-recall`.

## Public release checklist

- Unit tests pass.
- Ruff passes or documented exceptions are intentional.
- README install instructions work from a clean Hermes profile.
- No secrets, accidental local database files, or private project residue in committed examples.
