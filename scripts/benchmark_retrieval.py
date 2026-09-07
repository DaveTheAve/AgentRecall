from __future__ import annotations

# ruff: noqa: E402,I001 -- this script amends sys.path so it can run directly from scripts/.

import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_recall_store import AgentRecallStore, EmbeddingClient, FUSION_CONFIG, _json_loads


SYNTHETIC_CASES = [
    ("inference endpoint port", "The local inference endpoint uses an operator-configured port for direct model-server calls."),
    ("concise terminal reports", "The user prefers concise terminal-friendly technical reports."),
    ("shared memory isolation", "Shared memory isolation keeps shared memories visible across agents while private memories stay private."),
    ("embedding fallback lexical", "Embedding fallback lexical search uses SQLite FTS5 when embeddings fail."),
    ("physical cleanup expired", "Physical cleanup purges expired memories and orphan relationship rows."),
]

QUERY_EMBEDDING = [1.0, 0.0]
TARGET_EMBEDDING = [0.0, 1.0]
DECOY_EMBEDDING = [1.0, 0.0]
STOPWORDS = {
    "about",
    "above",
    "after",
    "agent",
    "agentrecall",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "default",
    "does",
    "from",
    "have",
    "hermes",
    "into",
    "memory",
    "profile",
    "should",
    "that",
    "their",
    "there",
    "this",
    "through",
    "using",
    "with",
    "without",
    "workspace",
}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def _legacy_lexical(query: str, row: Any) -> float:
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    terms = [term for term in q.replace('"', " ").split() if len(term) > 1]
    if not terms:
        return 0.0
    hay = " ".join(
        [
            row["title"] or "",
            row["content"] or "",
            row["summary"] or "",
            row["category"] or "",
            row["tags_json"] or "",
        ]
    ).lower()
    return sum(1 for term in terms if term in hay) / len(terms)


def _visible_leak(row: dict[str, Any], *, workspace_id: str, agent_id: str, session_id: str) -> bool:
    if row.get("workspace_id") != workspace_id:
        return True
    visibility = row.get("visibility")
    if visibility == "shared":
        return False
    if visibility == "agent":
        return row.get("agent_id") != agent_id
    if visibility == "session":
        return row.get("agent_id") != agent_id or row.get("session_id") != session_id
    return True


def _legacy_search(
    store: AgentRecallStore,
    query: str,
    query_embedding: list[float],
    *,
    workspace_id: str,
    agent_id: str,
    session_id: str,
    category: str = "",
    include_shared: bool = True,
    limit: int = 5,
) -> list[dict[str, Any]]:
    scope, args = store._scope_sql(workspace_id, agent_id, session_id, include_shared, alias="m")
    if category:
        scope += " AND m.category = ?"
        args.append(category)
    # Simulate the pre-change algorithm over the same non-expired, ACL-filtered corpus:
    # one SQL scan, simple substring lexical overlap, vector-dominant weighted sum.
    with store._lock:
        rows = list(store.conn.execute(f"SELECT m.* FROM memories m WHERE {scope}", args))
    now = time.time()
    scored: list[tuple[float, Any]] = []
    for row in rows:
        vector_score = _cosine(query_embedding, _json_loads(row["embedding_json"], []))
        lexical = _legacy_lexical(query, row)
        recency_days = max(0.0, (now - float(row["updated_at"])) / 86400.0)
        recency = 1.0 / (1.0 + recency_days / 30.0)
        importance = float(row["importance"] or 0.5)
        score = (0.58 * vector_score) + (0.25 * lexical) + (0.10 * importance) + (0.07 * recency)
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "id": int(row["id"]),
            "workspace_id": row["workspace_id"],
            "agent_id": row["agent_id"],
            "visibility": row["visibility"],
            "session_id": row["session_id"],
            "score": round(score, 4),
        }
        for score, row in scored[:limit]
    ]


def _summarize(
    name: str,
    timings: list[float],
    correct1: int,
    correct5: int,
    reciprocal_ranks: list[float],
    acl_leaks: int,
    expired_hits: int,
    total: int,
) -> dict[str, object]:
    p95_ms = max(timings) if len(timings) < 2 else statistics.quantiles(timings, n=20, method="inclusive")[18]
    return {
        "name": name,
        "top1_accuracy": correct1 / total if total else 0.0,
        "top5_recall": correct5 / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / total if total else 0.0,
        "acl_leaks": acl_leaks,
        "expired_hits": expired_hits,
        "p95_ms": round(p95_ms, 3),
    }


def _relevant_rank(ranked_ids: list[int], relevant_ids: set[int]) -> int:
    return next((rank for rank, memory_id in enumerate(ranked_ids, 1) if memory_id in relevant_ids), 0)


def _compare_cases(
    store: AgentRecallStore,
    cases: list[dict[str, Any]],
    *,
    workspace_id: str,
    agent_id: str,
    session_id: str,
    category: str = "",
    include_shared: bool = True,
    use_expected_embedding: bool = False,
) -> dict[str, object]:
    current_timings: list[float] = []
    legacy_timings: list[float] = []
    current_correct1 = legacy_correct1 = 0
    current_correct5 = legacy_correct5 = 0
    current_rr: list[float] = []
    legacy_rr: list[float] = []
    current_acl_leaks = legacy_acl_leaks = 0
    current_expired_hits = legacy_expired_hits = 0
    examples: list[dict[str, object]] = []
    now = time.time()
    for case in cases:
        expected_id = int(case["expected_id"])
        relevant_ids = {int(memory_id) for memory_id in case.get("relevant_ids", [expected_id])}
        query = str(case["query"])
        query_embedding = list(case.get("embedding") or []) if use_expected_embedding else list(case.get("query_embedding") or [])

        start = time.perf_counter()
        legacy_rows = _legacy_search(
            store,
            query,
            query_embedding,
            workspace_id=workspace_id,
            agent_id=agent_id,
            session_id=session_id,
            category=category,
            include_shared=include_shared,
            limit=5,
        )
        legacy_timings.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        current_rows = store.search(
            workspace_id=workspace_id,
            agent_id=agent_id,
            session_id=session_id,
            query=query,
            query_embedding=query_embedding,
            include_shared=include_shared,
            category=category,
            limit=5,
            track_access=False,
        )
        current_timings.append((time.perf_counter() - start) * 1000)

        legacy_ids = [row["id"] for row in legacy_rows]
        current_ids = [row["id"] for row in current_rows]
        legacy_rank = _relevant_rank(legacy_ids, relevant_ids)
        current_rank = _relevant_rank(current_ids, relevant_ids)
        if legacy_rank == 1:
            legacy_correct1 += 1
        if current_rank == 1:
            current_correct1 += 1
        if legacy_rank:
            legacy_correct5 += 1
            legacy_rr.append(1.0 / legacy_rank)
        else:
            legacy_rr.append(0.0)
        if current_rank:
            current_correct5 += 1
            current_rr.append(1.0 / current_rank)
        else:
            current_rr.append(0.0)
        legacy_acl_leaks += sum(1 for row in legacy_rows if _visible_leak(row, workspace_id=workspace_id, agent_id=agent_id, session_id=session_id))
        current_acl_leaks += sum(1 for row in current_rows if _visible_leak(row, workspace_id=workspace_id, agent_id=agent_id, session_id=session_id))
        legacy_expired_hits += sum(1 for row in legacy_rows if float(case.get("expires_by_id", {}).get(str(row["id"]), 0)) > 0 and float(case["expires_by_id"][str(row["id"])]) <= now)
        current_expired_hits += sum(1 for row in current_rows if float(case.get("expires_by_id", {}).get(str(row["id"]), 0)) > 0 and float(case["expires_by_id"][str(row["id"])]) <= now)
        if len(examples) < 10:
            examples.append(
                {
                    "case_id": case.get("case_id"),
                    "expected_id": expected_id,
                    "relevant_count": len(relevant_ids),
                    "legacy_rank": legacy_rank,
                    "current_rank": current_rank,
                    "legacy_top_id": legacy_ids[0] if legacy_ids else None,
                    "current_top_id": current_ids[0] if current_ids else None,
                }
            )
    total = len(cases)
    legacy = _summarize("legacy_vector_weighted_overlap", legacy_timings, legacy_correct1, legacy_correct5, legacy_rr, legacy_acl_leaks, legacy_expired_hits, total)
    current = _summarize("current_bm25_fusion", current_timings, current_correct1, current_correct5, current_rr, current_acl_leaks, current_expired_hits, total)
    return {
        "cases": total,
        "legacy": legacy,
        "current": current,
        "delta": {
            "top1_accuracy": round(float(current["top1_accuracy"]) - float(legacy["top1_accuracy"]), 4),
            "top5_recall": round(float(current["top5_recall"]) - float(legacy["top5_recall"]), 4),
            "mrr": round(float(current["mrr"]) - float(legacy["mrr"]), 4),
            "p95_ms": round(float(current["p95_ms"]) - float(legacy["p95_ms"]), 3),
        },
        "examples": examples,
    }


def run_synthetic_benchmark(db_path: Path) -> dict[str, object]:
    target_path = db_path
    protected_paths = (
        target_path,
        Path(f"{target_path}-wal"),
        Path(f"{target_path}-shm"),
        Path(f"{target_path}-journal"),
    )
    conflict = next((path for path in protected_paths if path.exists() or path.is_symlink()), None)
    if conflict is not None:
        raise FileExistsError(f"refusing to overwrite existing benchmark database path or sidecar: {conflict}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    working_dir = Path(tempfile.mkdtemp(prefix="agent-recall-synthetic-"))
    db_path = working_dir / "bench.db"
    store = AgentRecallStore(db_path)
    cases: list[dict[str, Any]] = []
    for idx, (query, content) in enumerate(SYNTHETIC_CASES, 1):
        expected_id = store.add_memory(
            workspace_id="bench",
            agent_id="hermes",
            source_agent_id="hermes",
            visibility="agent",
            category="benchmark",
            content=content,
            embedding=TARGET_EMBEDDING,
            embedding_model="bench-fake",
            canonical_key=f"bench.case.{idx}",
        )
        store.add_memory(
            workspace_id="bench",
            agent_id="hermes",
            source_agent_id="hermes",
            visibility="agent",
            category="benchmark",
            content=f"Semantic-vector decoy {idx}; unrelated infrastructure note with no target query terms.",
            embedding=DECOY_EMBEDDING,
            embedding_model="bench-fake",
            canonical_key=f"bench.decoy.{idx}",
        )
        cases.append({"case_id": idx, "query": query, "expected_id": expected_id, "query_embedding": QUERY_EMBEDDING})
    store.add_memory(
        workspace_id="bench",
        agent_id="other-agent",
        source_agent_id="other-agent",
        visibility="agent",
        category="benchmark",
        content="private memory must never leak to hermes benchmark searches",
        embedding=DECOY_EMBEDDING,
        embedding_model="bench-fake",
        canonical_key="bench.private.other",
    )
    store.add_memory(
        workspace_id="bench",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="agent",
        category="benchmark",
        content="expired memory must not appear in retrieval",
        embedding=DECOY_EMBEDDING,
        embedding_model="bench-fake",
        expires_at=time.time() - 1,
        canonical_key="bench.expired",
    )
    expires_by_id = {
        str(row["id"]): float(row["expires_at"])
        for row in store.conn.execute("SELECT id, expires_at FROM memories").fetchall()
    }
    for case in cases:
        case["expires_by_id"] = expires_by_id
    comparison = _compare_cases(store, cases, workspace_id="bench", agent_id="hermes", session_id="", category="benchmark")
    cleanup = store.physical_cleanup(purge_expired=True, optimize_fts=True, vacuum=False)
    payload = {
        "mode": "synthetic",
        "db_path": str(target_path),
        "queries": len(cases),
        "fusion_config": dict(FUSION_CONFIG),
        **comparison,
        "cleanup": cleanup,
    }
    payload["gate_passed"] = (
        payload["current"]["top1_accuracy"] >= 0.9
        and payload["current"]["acl_leaks"] == 0
        and payload["current"]["expired_hits"] == 0
        and payload["current"]["p95_ms"] < 50
        and cleanup["expired_memories_deleted"] == 1
    )
    store.close()
    publish_conflict = next(
        (path for path in protected_paths if path.exists() or path.is_symlink()),
        None,
    )
    if publish_conflict is not None:
        db_path.unlink(missing_ok=True)
        working_dir.rmdir()
        raise FileExistsError(
            f"refusing to overwrite racing benchmark database path or sidecar: {publish_conflict}"
        )
    publish_dir = Path(
        tempfile.mkdtemp(prefix=f".{target_path.name}.publish-", dir=target_path.parent)
    )
    staged_path = publish_dir / "bench.db"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    staged_fd = os.open(staged_path, flags, 0o600)
    try:
        with db_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(staged_fd, remaining)
                    if written <= 0:
                        raise OSError("failed to make progress while staging benchmark database")
                    remaining = remaining[written:]
        os.fsync(staged_fd)
    except Exception:
        os.close(staged_fd)
        staged_path.unlink(missing_ok=True)
        publish_dir.rmdir()
        db_path.unlink(missing_ok=True)
        working_dir.rmdir()
        raise
    os.close(staged_fd)
    reserved_sidecars: list[dict[str, Any]] = []
    staged_identity = staged_path.lstat()
    published = False

    def capture_expected_path(
        path: Path,
        captured_path: Path,
        expected_device: int,
        expected_inode: int,
        description: str,
    ) -> None:
        try:
            current = path.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"{description} disappeared before release: {path}") from exc
        if (current.st_dev, current.st_ino) != (expected_device, expected_inode):
            raise RuntimeError(f"{description} was replaced; preserved replacement at {path}")
        os.rename(path, captured_path)
        captured = captured_path.lstat()
        if (captured.st_dev, captured.st_ino) != (expected_device, expected_inode):
            try:
                os.link(captured_path, path)
            except FileExistsError as exc:
                raise RuntimeError(
                    f"{description} was replaced; preserved replacement at {captured_path} because {path} raced restoration"
                ) from exc
            captured_path.unlink()
            raise RuntimeError(f"{description} was replaced; restored replacement at {path}")
        captured_path.unlink()

    def release_sidecar_reservations() -> None:
        errors: list[Exception] = []
        for index, reservation in enumerate(list(reversed(reserved_sidecars))):
            path = reservation["path"]
            if not reservation["closed"]:
                os.close(reservation["descriptor"])
                reservation["closed"] = True
            captured_path = publish_dir / f"reserved-sidecar-{index}"
            try:
                capture_expected_path(
                    path,
                    captured_path,
                    reservation["device"],
                    reservation["inode"],
                    "benchmark sidecar reservation",
                )
            except RuntimeError as exc:
                # A replacement is caller-owned. The reservation inode is no
                # longer reachable by path, so closing its descriptor is the
                # only cleanup that is safe.
                if "was replaced" in str(exc) or "disappeared" in str(exc):
                    reserved_sidecars.remove(reservation)
                errors.append(exc)
            else:
                reserved_sidecars.remove(reservation)
        if errors:
            raise errors[0]

    def rollback_published_target() -> None:
        if not published:
            return
        try:
            current = target_path.lstat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) != (staged_identity.st_dev, staged_identity.st_ino):
            return
        capture_expected_path(
            target_path,
            publish_dir / "published-main.db",
            staged_identity.st_dev,
            staged_identity.st_ino,
            "published benchmark database",
        )

    try:
        sidecar_path = protected_paths[1]
        try:
            for sidecar_path in protected_paths[1:]:
                descriptor = os.open(sidecar_path, flags, 0o600)
                reservation = os.fstat(descriptor)
                reserved_sidecars.append(
                    {
                        "path": sidecar_path,
                        "descriptor": descriptor,
                        "device": reservation.st_dev,
                        "inode": reservation.st_ino,
                        "closed": False,
                    }
                )
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite racing benchmark database sidecar: {sidecar_path}"
            ) from exc
        try:
            os.link(staged_path, target_path)
            published = True
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite racing benchmark database path: {target_path}") from exc
        release_sidecar_reservations()
        try:
            published_identity = target_path.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"published benchmark database was replaced or removed: {target_path}") from exc
        if (published_identity.st_dev, published_identity.st_ino) != (
            staged_identity.st_dev,
            staged_identity.st_ino,
        ):
            raise RuntimeError(f"published benchmark database was replaced; preserved replacement at {target_path}")
    except Exception:
        cleanup_errors: list[Exception] = []
        try:
            rollback_published_target()
        except Exception as cleanup_error:
            cleanup_errors.append(cleanup_error)
        try:
            release_sidecar_reservations()
        except Exception as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise RuntimeError("benchmark publication rollback could not safely restore every reserved path") from cleanup_errors[0]
        raise
    finally:
        staged_path.unlink(missing_ok=True)
        db_path.unlink(missing_ok=True)
        with suppress(OSError):
            publish_dir.rmdir()
        with suppress(OSError):
            working_dir.rmdir()
    return payload


def _terms(text: str, max_terms: int = 5) -> list[str]:
    out: list[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,}", text):
        lowered = token.lower().strip("_.:-")
        if len(lowered) < 4 or lowered in STOPWORDS or lowered in out:
            continue
        out.append(lowered)
        if len(out) >= max_terms:
            break
    return out


def _real_cases(
    store: AgentRecallStore,
    *,
    workspace_id: str,
    agent_id: str,
    session_id: str,
    include_shared: bool,
    max_cases: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scope, args = store._scope_sql(workspace_id, agent_id, session_id, include_shared, alias="m")
    with store._lock:
        rows = list(
            store.conn.execute(
                f"""
                SELECT m.* FROM memories m
                WHERE {scope} AND length(trim(m.content)) >= 30
                ORDER BY m.updated_at DESC
                LIMIT ?
                """,
                args + [max_cases * 5],
            )
        )
        visible_count = store.conn.execute(f"SELECT COUNT(*) FROM memories m WHERE {scope}", args).fetchone()[0]
        expires_by_id = {
            str(row["id"]): float(row["expires_at"])
            for row in store.conn.execute(f"SELECT m.id, m.expires_at FROM memories m WHERE {scope}", args).fetchall()
        }
    cases: list[dict[str, Any]] = []
    normalized_titles: dict[str, set[int]] = {}
    normalized_contents: dict[str, set[int]] = {}
    for row in rows:
        title_key = re.sub(r"\s+", " ", (row["title"] or "").strip().lower())
        content_key = re.sub(r"\s+", " ", (row["content"] or "").strip().lower())
        if title_key:
            normalized_titles.setdefault(title_key, set()).add(int(row["id"]))
        if content_key:
            normalized_contents.setdefault(content_key, set()).add(int(row["id"]))
    seen_queries: set[str] = set()
    for row in rows:
        title = (row["title"] or "").strip()
        title_key = re.sub(r"\s+", " ", title.lower())
        content_key = re.sub(r"\s+", " ", (row["content"] or "").strip().lower())
        if len(title) >= 8:
            query = title
            relevant_ids = set(normalized_titles.get(title_key, set()))
        else:
            text = "\n".join([row["summary"] or "", row["content"] or "", row["category"] or "", row["tags_json"] or ""])
            terms = _terms(text)
            if len(terms) < 2:
                continue
            query = " ".join(terms)
            relevant_ids = {int(row["id"])}
        relevant_ids.update(normalized_contents.get(content_key, set()))
        query_key = query.lower()
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)
        cases.append(
            {
                "case_id": int(row["id"]),
                "query": query,
                "expected_id": int(row["id"]),
                "relevant_ids": sorted(relevant_ids),
                "embedding": _json_loads(row["embedding_json"], []),
                "query_embedding": [],
                "expires_by_id": expires_by_id,
            }
        )
        if len(cases) >= max_cases:
            break
    metadata = {"visible_rows": int(visible_count), "eligible_rows_seen": len(rows), "cases_built": len(cases)}
    return cases, metadata


def run_real_benchmark(
    db_path: Path,
    *,
    workspace_id: str,
    agent_id: str,
    session_id: str = "",
    include_shared: bool = True,
    max_cases: int = 50,
    embedding_base_url: str = "",
    embedding_model: str = "",
    embedding_api_key: str = "",
) -> dict[str, object]:
    if not db_path.exists():
        raise FileNotFoundError(f"real DB copy does not exist: {db_path}")
    store = AgentRecallStore(db_path)
    cases, metadata = _real_cases(
        store,
        workspace_id=workspace_id,
        agent_id=agent_id,
        session_id=session_id,
        include_shared=include_shared,
        max_cases=max_cases,
    )
    if not cases:
        raise RuntimeError("no suitable visible real-memory cases could be built")
    lexical = _compare_cases(
        store,
        cases,
        workspace_id=workspace_id,
        agent_id=agent_id,
        session_id=session_id,
        include_shared=include_shared,
        use_expected_embedding=False,
    )
    live_query_embeddings: dict[str, object] | None = None
    if embedding_base_url:
        if not embedding_model:
            raise ValueError("embedding_model is required when embedding_base_url is provided")
        embedder = EmbeddingClient(
            embedding_base_url,
            embedding_model,
            api_key=embedding_api_key,
            timeout=30.0,
        )
        for case in cases:
            case["query_embedding"] = embedder.embed(str(case["query"]))
        live_query_embeddings = _compare_cases(
            store,
            cases,
            workspace_id=workspace_id,
            agent_id=agent_id,
            session_id=session_id,
            include_shared=include_shared,
            use_expected_embedding=False,
        )
    embedding_cases = [case for case in cases if case.get("embedding")]
    embedding_oracle = _compare_cases(
        store,
        embedding_cases,
        workspace_id=workspace_id,
        agent_id=agent_id,
        session_id=session_id,
        include_shared=include_shared,
        use_expected_embedding=True,
    ) if embedding_cases else {"cases": 0, "legacy": {}, "current": {}, "delta": {}, "examples": []}
    # Cleanup is intentionally conservative on copied real DBs: optimize FTS only, do not purge or vacuum.
    cleanup = store.physical_cleanup(purge_expired=False, optimize_fts=True, vacuum=False)
    payload = {
        "mode": "real-copy",
        "db_path": str(db_path),
        "workspace_id": workspace_id,
        "agent_id": agent_id,
        "session_id": session_id,
        "include_shared": include_shared,
        "fusion_config": dict(FUSION_CONFIG),
        "metadata": metadata,
        "lexical_only": lexical,
        "live_query_embeddings": live_query_embeddings,
        "stored_embedding_oracle": embedding_oracle,
        "cleanup": cleanup,
    }
    payload["gate_passed"] = (
        lexical["current"]["acl_leaks"] == 0
        and lexical["current"]["expired_hits"] == 0
        and lexical["current"]["p95_ms"] < 50
        and lexical["current"]["top5_recall"] >= 0.8
        and (
            live_query_embeddings is None
            or (
                live_query_embeddings["current"]["acl_leaks"] == 0
                and live_query_embeddings["current"]["expired_hits"] == 0
                and live_query_embeddings["current"]["p95_ms"] < 50
                and live_query_embeddings["current"]["top5_recall"] >= 0.9
            )
        )
    )
    store.close()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AgentRecall retrieval benchmark gates on disposable or copied DBs.")
    parser.add_argument("--db", type=Path, default=None, help="SQLite path. Synthetic mode defaults to a temp file.")
    parser.add_argument("--mode", choices=["synthetic", "real-copy"], default="synthetic")
    parser.add_argument("--workspace-id", default="shared-workspace")
    parser.add_argument("--agent-id", default="hermes")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--no-shared", action="store_true", help="Do not include shared memories in real-copy mode.")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--embedding-base-url", default="", help="Optional OpenAI-compatible endpoint for real query embeddings.")
    parser.add_argument("--embedding-model", default="", help="Embedding model used with --embedding-base-url.")
    parser.add_argument("--embedding-api-key-env", default="", help="Optional environment variable containing the embedding API key.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    parser.add_argument("--gate", action="store_true", help="Exit non-zero unless quality/latency gates pass.")
    args = parser.parse_args(argv)
    db_path = args.db or Path(tempfile.mkdtemp(prefix="agent-recall-bench-")) / "bench.db"
    if args.mode == "synthetic":
        # Preserve the final component so the publisher can reject symlinks,
        # including broken links, rather than publishing at their referents.
        payload = run_synthetic_benchmark(db_path.expanduser().absolute())
    else:
        payload = run_real_benchmark(
            db_path.expanduser().resolve(),
            workspace_id=args.workspace_id,
            agent_id=args.agent_id,
            session_id=args.session_id,
            include_shared=not args.no_shared,
            max_cases=max(1, int(args.max_cases or 50)),
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            embedding_api_key=os.environ.get(args.embedding_api_key_env, "") if args.embedding_api_key_env else "",
        )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if (not args.gate or payload["gate_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
