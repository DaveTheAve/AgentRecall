# v0.2.0 compatibility and v0.3.0 release review

## Verified baseline

The public baseline is `63ca33df443b599d2e5afffbecf78d602b986045` in [DaveTheAve/AgentRecall](https://github.com/DaveTheAve/AgentRecall).

Git records:

- Subject: `[verified] feat: release AgentRecall 0.2.0 multi-host memory`
- Parent: `a603d918dba1c5f23f047761f20844837e483707`
- Commit date: 2026-08-20 (UTC-04:00)
- Change size: 32 files, 6,451 insertions, 807 deletions.

These values were rechecked against Git during the public-commit review. Earlier candidate notes contained inconsistent baseline metadata and should not be used as release evidence.

## Compatibility surface

The candidate preserves the host-neutral Python core, persistent JSONL bridge, optional MCP server, and native Hermes/OpenClaw adapters. The new lifecycle fields are exposed through all three host surfaces:

- Hermes and MCP: `canonical_key` and `expires_at` on remember, `expires_at` on update.
- OpenClaw: `canonicalKey` and `expiresAt` on store, `expiresAt` on update.
- Physical cleanup remains a direct store maintenance API, not an LLM-callable tool.

Durable canonical identities are unique by workspace, owner, visibility, and key across sessions. Session-scoped canonical identities additionally include the session. Concurrent canonical writes are serialized by SQLite and report whether they added or updated the row.

## Migration and rollback

The review generated a disposable database using the exact baseline's `agent_recall_store.py`, inserted shared/private/session fixtures, and made a SQLite online backup while the source connection remained open. The candidate opened the backup, preserved the original rows with permanent defaults, and accepted a new canonical write. The exact v0.2.0 store then reopened that migrated copy and accepted a legacy write. SQLite integrity checks passed, and the original fixture retained the v0.2.0 schema.

This verifies schema readability/writability, not equivalent feature semantics after rollback: v0.2.0 does not understand expiration or canonical upsert. Do not rely on expiration filtering when old hosts share a migrated database. Candidate migration also reconciles duplicate canonical rows from earlier development schemas, retaining the newest row and deleting older duplicates and orphan links. Back up before upgrading.

No private production database, installed profile, or live plugin checkout was modified for this review.

## Retrieval behavior and limits

The final score combines BM25 rank (0.25), lexical overlap (0.15), vector similarity (0.55), importance (0.03), recency (0.02), and an all-query-terms lexical boost (0.16). Exact ties use memory ID ascending.

Public search returns at most 50 rows. Python candidate materialization/fusion is capped at 500 rows after SQL ACL/scalar/tag filters. Hybrid retrieval divides that window between BM25 matches and an importance/recency-ordered vector source window. This is approximate semantic retrieval: an older or lower-importance semantic match outside that window can be missed. SQL filtering and sorting can examine more than 500 rows, so this is not a constant-time database scan or latency guarantee.

Only embedding-free FTS queries use safely quoted prefix terms. Prefix matching is not arbitrary substring matching. Prefetch deduplicates canonical projections, applies scope preference and context limits, and records access only for rows emitted into context.

## Verification evidence and reproduction

The public-commit review ran the Python and Node suites, Ruff, the synthetic retrieval gate, wheel/source builds, wheel metadata/content checks, clean wheel installation, both console entry points, and the extracted npm package's persistent bridge smoke. The installed wheel plus optional MCP dependency audit reported no known vulnerabilities. The npm package has no required runtime dependencies; its OpenClaw peer is optional.

Reproduce the core gates from the repository with:

```bash
python -m pip install -e '.[dev,mcp]'
python -m pytest
ruff check .
npm test
python scripts/benchmark_retrieval.py --mode synthetic --gate --json
```

See `.github/workflows/tests.yml` for artifact build and extracted-package checks. A local pass does not establish a remote CI pass; use the GitHub Actions run for the actual published commit.

The initial synthetic review run passed all five cases: top-1, top-5, and MRR were 1.0, with zero ACL leaks or expired hits. Synthetic latency is machine-dependent and this small adversarial corpus is not a production-scale performance claim.

`WEIGHT_TUNING.md` preserves historical development measurements over a private corpus. Its private inputs are not shipped, and those measurements were not independently reproduced during this public-commit review. They are context for the selected weights, not a reproducible public benchmark or a fresh security certification.
