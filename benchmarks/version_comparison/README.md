# Actual-version comparison

Run with Python 3.11+ and the project's optional development/MCP dependencies:

```sh
python scripts/benchmark_version_comparison.py \
  --baseline-root /path/to/v0.2-checkout \
  --candidate-root /path/to/current-checkout \
  --rounds 5 --warmups 10 --samples 100 \
  --json-output /tmp/agentrecall-comparison/report.json \
  --markdown-output /tmp/agentrecall-comparison/summary.md
```

The baseline must be an actual v0.2 checkout, not a simulated legacy algorithm. First verify its Git revision without initializing a provider against live data. Do not change the live plugin symlink or configuration. Both roots must be Git checkouts; a dirty candidate is allowed and fingerprinted.

The harness runs each version in separate subprocesses with isolated HOME/HERMES_HOME, disposable synthetic SQLite stores, stripped credentials, and a Python audit hook preventing network, subprocess, and external state access. Workers start with `-B -X pycache_prefix=<empty temporary directory>` before importing checkout code: `-B` alone would still read stale checkout bytecode. Existing checkout caches are neither consumed nor modified. The harness supplies only the database and embedding fields shared by v0.2 and the candidate, forces lexical fallback after construction, and never invokes optional curation or capture workflows. This is defense in depth for trusted local source, not an OS sandbox for arbitrary hostile code. No live memories or model calls are used.

Measurements:

- Identical authored synthetic corpus with 24 queries, same lexical distractors, and private/shared/session/workspace ACL partitions.
- Recall@1, recall@5, and MRR from top-50 rankings, plus direct-ID and search ACL leak counts.
- Warm median and nearest-rank p95 of top-5 core searches and writes through the explicit read-write MCP adapter when available.
- Paired seeded randomized version order, alternating read/write operation order, warmups excluded, raw samples retained.
- Adapter and registered MCP SDK-handler observations: strict rejection before a counted backend call, injected backend error sanitization, nested session metadata omission, and read-only non-mutation/discovery.

Security probes using an injected backend are labeled fault injection; retrieval and persistence checks use real core/SQLite. The in-process SDK-handler checks are not wire tests. The separate pytest MCP security suites exercise actual stdio/HTTP transports.

Missing adapter/factory capabilities or an absent top-level optional `mcp` package are N/A. SDK import dependency failures and factory construction failures are **error**, not N/A. Invalid arguments count as rejected only with a typed contract exception or an explicit safe MCP rejection code, zero backend calls, and a successful valid-input control that calls the backend exactly once. Arbitrary `ValueError`/`RuntimeError`, opaque `isError` responses, and failed valid controls are not validation successes. Legacy `MCPAccessError` is recognized by its actual class. For SDK errors, a temporary in-process observer retains the active typed validation exception (including the SDK's explicitly chained Pydantic error) before the SDK flattens it to text; it leaves the actual handler response unchanged. Error-looking text is never used as validation evidence.

The deliberate backend-error injection must reach its backend exactly once. Its raw exception leakage is a measured safety observation, not an unexpected harness failure. Unexpected probe errors and failed valid controls propagate to version/report `status: error`, an exit status of 1, and readable errors for **every round**, rather than first-round-only green-looking counts. `measured` means the observations ran, not that every safety property passed. JSON schema version 2 records these statuses, valid controls, and bounded error classifications; no raw response/error text is published.

All available `agent_recall_*.py` sources, including lazy-import modules, are fingerprinted before imports, checked immediately after imports and measurements, compared with the parent's pre-worker manifest, and rechecked before publication. Imported paths/hashes must match that manifest. Changing source during a run requires rerunning it after freezing the implementation.

## Publication and preservation

`--json-output` must end in `.json` and `--markdown-output` must end in `.md`. Both destinations are caller-specified and checked **before any worker starts**. Existing files, directories, symlinks, and broken symlinks are refused. Choose new filenames for a rerun; the harness never overwrites prior measurements.

Complete UTF-8 files are staged and flushed on each destination filesystem, then published with exclusive atomic hard links, Markdown first and JSON last. Hard-link creation fails if any competing destination appears after preflight, so there is no unsafe check-then-overwrite window. Each file is atomic; two independent filenames cannot be committed as one filesystem transaction. A collision on the final JSON can leave the complete Markdown output from this attempt. The harness deliberately never rolls back a public path, since a racer could have replaced it. A failed publication is not a successful report pair; retry with fresh names after investigating. Filesystems without hard-link support fail closed rather than falling back to clobbering writes.

## Focused verification

```sh
AGENT_RECALL_BENCHMARK_BASELINE_TEST=/path/to/verified/baseline \
  uv run --python 3.12 --with '.[dev,mcp]' pytest tests/test_version_comparison.py
```

Without the environment variable, the actual-root smoke is skipped; synthetic unit tests are not performance measurements. This test module skips on Python below 3.11 because the harness uses `tomllib`; AgentRecall's runtime support remains Python 3.10+.

Limitations: this deliberately simple synthetic benchmark measures lexical fallback and contract safety, not embedding quality, LLM curation, production data, cold start, transport throughput, or statistically significant speedups. A feature absent from one checkout is reported as unavailable, not scored as zero.

The JSON report contains local absolute source paths to verify imports. Keep it outside the public checkout, or publish a copy with baseline/candidate path labels while retaining code hashes. The accompanying Markdown summary has no source-root paths.
