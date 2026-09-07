# Changelog

All notable changes to AgentRecall are documented here.

## [0.3.0] - 2026-09-07

### Added

- Optional `canonical_key` support for deterministic in-place durable fact updates across Hermes, OpenClaw, and MCP.
- Optional `expires_at` timestamps; omitted values and `0` keep memories permanent.
- A direct store maintenance API for purging expired rows and orphan links, with optional SQLite and FTS maintenance.
- An optional Hermes SessionArchive companion for read-only search, session reads, scrolling, and recent-session browsing within the current Hermes profile.

### Changed

- MCP now defaults to read-only access and uses strict request schemas, allowlisted response projections, fixed server identity, permission-aware discovery, bounded workers and queues, operation deadlines, and size-limited stdio and HTTP input.
- MCP protocol, transport, and backend failures return stable public errors without exposing exceptions, request data, credentials, session identifiers, or server paths.
- Authenticated MCP HTTP validates bearer tokens, Host and Origin headers, duplicate headers and JSON keys, request envelopes, and incrementally counted body limits before SDK dispatch.
- Retrieval uses deterministic FTS5/BM25 and embedding fusion with bounded candidate processing. Embedding-free fallback supports quoted lexical prefixes.
- ACL, scalar, and tag filters are applied before candidate limits. Public search remains capped at 50 rows and internal candidate processing at 500 rows.
- Prefetch deduplicates equivalent canonical projections before context truncation and records access only for rows included in final context.
- Durable `agent` and `shared` canonical identities span sessions; `session` canonical identities remain session-isolated.
- Codex curation passes prompts over standard input. Curation transport and parsed output are bounded, external failure details are redacted, and model-proposed canonical hints cannot overwrite existing memories.
- SessionArchive delegates to the host-owned archive, accepts no caller-supplied profile or database path, fails closed when current-profile isolation cannot be established, and treats returned conversations as untrusted data.

### Compatibility

- Existing v0.2.0 databases migrate additively on first open; concurrent first-open migration and canonical writes are serialized.
- The added columns and indexes remain readable by v0.2.0, but v0.2.0 ignores expiration and canonical-upsert semantics.
- Migration preserves permanent rows and does not run physical cleanup. Use SQLite online backup before production upgrades.

## [0.2.0] - 2026-08-19

- Added the host-neutral core, persistent local bridge, MCP server, native OpenClaw integration, multi-host identity isolation, lifecycle hooks, packaging, and concurrency hardening.
