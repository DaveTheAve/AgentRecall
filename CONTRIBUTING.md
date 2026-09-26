# Contributing

## Development setup

```bash
python -m pip install -e '.[dev,mcp]'
python -m pytest
ruff check .
python -m py_compile __init__.py hermes_plugin/__init__.py agent_recall_core.py agent_recall_store.py agent_recall_curator.py agent_recall_schemas.py agent_recall_mcp.py agent_recall_bridge.py cli.py scripts/*.py tests/*.py
npm test
python scripts/benchmark_retrieval.py --mode synthetic --gate --json
```

## Hermes and MCP compatibility tests

Use an installed Hermes runtime for host-backed checks (not a bare interpreter
with only this project's dependencies):

```bash
export AGENT_RECALL_HERMES_SOURCE=/path/to/hermes-agent
export AGENT_RECALL_HERMES_PYTHON=/path/to/hermes-runtime/bin/python
python -m pytest tests/test_hermes_compatibility.py tests/test_release_artifacts.py
```

The host probes use disposable homes and disable automatic dependency installation;
they do not change live profiles. Both source and packaged manifests are checked by
Hermes Plugin Doctor, then real provider discovery and MemoryManager callbacks.
MemoryProvider lifecycle methods must not be declared as generic manifest hooks.

The MCP extra supports SDK 1 and SDK 2. Run the full suite in separate environments
with `mcp==1.30.0` and `mcp==2.0.0`; do not change the host's managed environment.
`tests/test_mcp_sdk.py` exercises real stdio and authenticated Streamable HTTP,
including memory round trips, malformed-frame recovery and error sanitization.
`tests/test_mcp_release.py` repeats the stdio contract against an installed wheel.

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
- Release versions agree across Python, Hermes, OpenClaw, npm, and `uv.lock` metadata.
- A wheel installs in a clean virtual environment; its native provider and manifest import through `hermes_plugin`, and both console entry points start.
- `twine check` and `check-wheel-contents --ignore W009` pass; W009 is expected for the intentionally flat compatibility modules.
- `npm pack --dry-run` contains only the intended OpenClaw bridge files, and the extracted package passes the persistent bridge remember/search smoke.
- The synthetic retrieval gate and a disposable online-backup migration rehearsal pass.
