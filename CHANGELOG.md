# Changelog

All notable changes to AgentRecall are documented here.

## [0.3.0] - Unreleased

### Added

- Optional `canonical_key` support for deterministic in-place durable fact updates.
- Optional `expires_at` timestamps; omitted values and `0` remain permanent.
- Direct maintenance API for physical cleanup of expired rows, orphan links, and optional SQLite/FTS maintenance.
- Synthetic and disposable real-copy retrieval benchmarks with ACL, expiration, relevance, and latency gates.

### Changed

- Retrieval now uses deterministic FTS5/BM25 and embedding fusion with bounded candidate processing.
- Embedding-free lexical fallback supports quoted FTS prefix candidates without broadening normal embedding-backed search.
- Hermes, MCP, and OpenClaw expose canonical update and expiration fields consistently.
- Automatic prefetch deduplicates equivalent canonical projections before context truncation and records access only for final context rows.
- Durable `agent` and `shared` canonical identities span sessions; `session` canonical identities remain session-isolated.

### Reliability and security

- Legacy v0.2.0 databases migrate additively on first open.
- Concurrent multi-host first-open migration and canonical writes are serialized safely and covered by multiprocessing tests.
- ACL, scalar, and tag filters are applied before bounded candidate limits.
- Public limits remain capped at 50; internal candidate processing remains capped at 500.
- Synthetic benchmark output fails closed for existing databases, SQLite sidecars, broken symlinks, concurrent creators, and partial-copy failures.

### Compatibility

- The schema migration is backward-compatible with v0.2.0 readers and writers: older code ignores the added columns and indexes.
- Migration does not expire permanent rows and does not run physical cleanup automatically.
- A SQLite online backup is still recommended immediately before a production upgrade.
- Schema rollback compatibility does not preserve v0.3.0 semantics in older hosts: v0.2.0 ignores expiration and canonical upsert.

## [0.2.0] - 2026-08-19

- Added the host-neutral core, persistent local bridge, MCP server, native OpenClaw integration, multi-host identity isolation, lifecycle hooks, packaging, and concurrency hardening.
