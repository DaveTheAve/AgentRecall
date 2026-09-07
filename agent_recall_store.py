from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
import urllib.request
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any


def _now() -> float:
    return time.time()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def normalize_text(text: str, max_chars: int = 12000) -> str:
    text = (text or "").replace("\x00", " ").strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text[:max_chars]


class EmbeddingClient:
    """Tiny OpenAI-compatible embeddings client for semantic indexing."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        dimensions: int = 0,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.dimensions = int(dimensions or 0)
        self.timeout = float(timeout or 20.0)

    def embed(self, text: str) -> list[float]:
        if not self.base_url:
            raise RuntimeError("embedding_base_url is not configured")
        body: dict[str, Any] = {"model": self.model, "input": text}
        # Some OpenAI-compatible/llama.cpp servers reject dimensions; only send when explicitly positive.
        if self.dimensions > 0:
            body["dimensions"] = self.dimensions
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/embeddings",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        emb = payload.get("data", [{}])[0].get("embedding")
        if not isinstance(emb, list) or not emb:
            raise RuntimeError("embedding endpoint returned no embedding")
        return [float(x) for x in emb]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


FUSION_CONFIG = {
    "bm25": 0.25,
    "lexical": 0.15,
    "vector": 0.55,
    "importance": 0.03,
    "recency": 0.02,
    "exact_lexical_boost": 0.16,
}


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    source_agent_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    visibility TEXT NOT NULL CHECK (visibility IN ('agent','shared','session')),
    category TEXT NOT NULL DEFAULT 'general',
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    embedding_json TEXT NOT NULL DEFAULT '[]',
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dimensions INTEGER NOT NULL DEFAULT 0,
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.8,
    canonical_key TEXT NOT NULL DEFAULT '',
    expires_at REAL NOT NULL DEFAULT 0,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_accessed_at REAL NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(workspace_id, visibility, agent_id, session_id, archived);
CREATE INDEX IF NOT EXISTS idx_mem_category ON memories(workspace_id, category, archived);
CREATE INDEX IF NOT EXISTS idx_mem_updated ON memories(workspace_id, updated_at DESC);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    title, content, summary, tags, category,
    content='memories', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, title, content, summary, tags, category)
    VALUES (new.id, new.title, new.content, new.summary, new.tags_json, new.category);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content, summary, tags, category)
    VALUES ('delete', old.id, old.title, old.content, old.summary, old.tags_json, old.category);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content, summary, tags, category)
    VALUES ('delete', old.id, old.title, old.content, old.summary, old.tags_json, old.category);
    INSERT INTO memories_fts(rowid, title, content, summary, tags, category)
    VALUES (new.id, new.title, new.content, new.summary, new.tags_json, new.category);
END;
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    from_id INTEGER NOT NULL,
    to_id INTEGER NOT NULL,
    relation TEXT NOT NULL DEFAULT 'related',
    weight REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL,
    UNIQUE(workspace_id, from_id, to_id, relation)
);
"""


class AgentRecallStore:
    def __init__(self, db_path: str | Path, *, busy_timeout_ms: int = 5_000) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = max(1_000, int(busy_timeout_ms or 5_000))
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.busy_timeout_ms / 1000.0,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self.conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            self.conn.execute("PRAGMA foreign_keys=ON")
            with suppress(Exception):
                self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            self._migrate_schema_locked()
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _ensure_column_locked(self, table: str, column: str, ddl: str) -> None:
        columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column in columns:
            return
        try:
            self.conn.execute(ddl)
        except sqlite3.OperationalError:
            # Multiple host processes can open the same old shared DB at once.
            # If another process won the ALTER race, accept the now-present
            # column; propagate every other migration failure.
            columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in columns:
                raise

    def _migrate_schema_locked(self) -> None:
        self._ensure_column_locked(
            "memories",
            "canonical_key",
            "ALTER TABLE memories ADD COLUMN canonical_key TEXT NOT NULL DEFAULT ''",
        )
        self._ensure_column_locked(
            "memories",
            "expires_at",
            "ALTER TABLE memories ADD COLUMN expires_at REAL NOT NULL DEFAULT 0",
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_expires ON memories(workspace_id, expires_at, archived)")
        # An earlier development index included session_id for every visibility,
        # which allowed durable agent/shared duplicates across sessions. Keep the
        # newest row before installing the corrected durable uniqueness rule.
        self.conn.execute(
            """
            DELETE FROM memories
            WHERE id IN (
              SELECT older.id
              FROM memories AS older
              JOIN memories AS newer
                ON newer.workspace_id = older.workspace_id
               AND newer.agent_id = older.agent_id
               AND newer.visibility = older.visibility
               AND newer.canonical_key = older.canonical_key
               AND (
                    newer.updated_at > older.updated_at
                    OR (newer.updated_at = older.updated_at AND newer.id > older.id)
               )
              WHERE older.canonical_key <> '' AND older.visibility <> 'session'
            )
            """
        )
        self.conn.execute(
            "DELETE FROM links WHERE from_id NOT IN (SELECT id FROM memories) OR to_id NOT IN (SELECT id FROM memories)"
        )
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mem_canonical_durable "
            "ON memories(workspace_id, agent_id, visibility, canonical_key) "
            "WHERE canonical_key <> '' AND visibility <> 'session'"
        )
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mem_canonical_session "
            "ON memories(workspace_id, agent_id, visibility, session_id, canonical_key) "
            "WHERE canonical_key <> '' AND visibility = 'session'"
        )
        self.conn.execute("DROP INDEX IF EXISTS idx_mem_canonical")

    def add_memory(
        self,
        *,
        workspace_id: str,
        agent_id: str,
        content: str,
        embedding: Sequence[float],
        embedding_model: str,
        visibility: str = "agent",
        source_agent_id: str = "",
        user_id: str = "",
        session_id: str = "",
        category: str = "general",
        title: str = "",
        summary: str = "",
        tags: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        confidence: float = 0.8,
        canonical_key: str = "",
        expires_at: float = 0,
        return_action: bool = False,
    ) -> int | tuple[int, str]:
        if visibility not in {"agent", "shared", "session"}:
            raise ValueError("visibility must be one of: agent, shared, session")
        content = normalize_text(content)
        if not content:
            raise ValueError("content is required")
        ts = _now()
        canonical_key = (canonical_key or "").strip()
        expires_at = max(0.0, float(expires_at or 0))
        values = (
            workspace_id,
            agent_id,
            source_agent_id or agent_id,
            user_id or "",
            session_id or "",
            visibility,
            category or "general",
            title or "",
            content,
            summary or "",
            _json_dumps(list(tags or [])),
            _json_dumps(metadata or {}),
            _json_dumps(list(embedding)),
            embedding_model or "",
            len(embedding),
            float(importance),
            float(confidence),
            canonical_key,
            expires_at,
            ts,
            ts,
        )
        with self._lock, self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO memories (
                  workspace_id, agent_id, source_agent_id, user_id, session_id, visibility,
                  category, title, content, summary, tags_json, metadata_json, embedding_json,
                  embedding_model, embedding_dimensions, importance, confidence, canonical_key,
                  expires_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                values,
            )
            if cur.rowcount == 1:
                memory_id = int(cur.lastrowid)
                action = "added"
            elif canonical_key:
                row = self.conn.execute(
                    """
                    SELECT id FROM memories
                    WHERE workspace_id = ? AND agent_id = ? AND visibility = ?
                      AND session_id = ? AND canonical_key = ?
                    """,
                    [workspace_id, agent_id, visibility, session_id or "", canonical_key],
                ).fetchone()
                if row is None and visibility != "session":
                    # Durable canonical uniqueness is cross-session, so every
                    # conflict mode must resolve the durable winner globally.
                    row = self.conn.execute(
                        """
                        SELECT id FROM memories
                        WHERE workspace_id = ? AND agent_id = ? AND visibility = ?
                          AND canonical_key = ?
                        """,
                        [workspace_id, agent_id, visibility, canonical_key],
                    ).fetchone()
                if row is None:
                    raise sqlite3.IntegrityError("canonical conflict row disappeared during upsert")
                memory_id = int(row["id"])
                self.conn.execute(
                    """
                    UPDATE memories SET
                      source_agent_id = ?, user_id = ?, category = ?, title = ?, content = ?, summary = ?,
                      tags_json = ?, metadata_json = ?, embedding_json = ?, embedding_model = ?,
                      embedding_dimensions = ?, importance = ?, confidence = ?, expires_at = ?,
                      archived = 0, updated_at = ?
                    WHERE id = ?
                    """,
                    [
                        values[2],
                        values[3],
                        values[6],
                        values[7],
                        values[8],
                        values[9],
                        values[10],
                        values[11],
                        values[12],
                        values[13],
                        values[14],
                        values[15],
                        values[16],
                        values[18],
                        values[20],
                        memory_id,
                    ],
                )
                action = "updated"
            else:
                raise sqlite3.IntegrityError("unexpected non-canonical memory conflict")
            self.conn.commit()
            return (memory_id, action) if return_action else memory_id

    def _scope_sql(
        self,
        workspace_id: str,
        agent_id: str,
        session_id: str,
        include_shared: bool = True,
        *,
        alias: str = "",
    ) -> tuple[str, list[Any]]:
        prefix = f"{alias}." if alias else ""
        clauses = [
            f"{prefix}workspace_id = ?",
            f"{prefix}archived = 0",
            f"({prefix}expires_at <= 0 OR {prefix}expires_at > ?)",
        ]
        args: list[Any] = [workspace_id, _now()]
        visible = [f"({prefix}visibility = 'agent' AND {prefix}agent_id = ?)"]
        args.append(agent_id)
        if session_id:
            visible.append(f"({prefix}visibility = 'session' AND {prefix}agent_id = ? AND {prefix}session_id = ?)")
            args.extend([agent_id, session_id])
        if include_shared:
            visible.append(f"{prefix}visibility = 'shared'")
        clauses.append("(" + " OR ".join(visible) + ")")
        return " AND ".join(clauses), args

    def get_visible(
        self, memory_id: int, workspace_id: str, agent_id: str, session_id: str, include_shared: bool = True
    ) -> sqlite3.Row | None:
        scope, args = self._scope_sql(workspace_id, agent_id, session_id, include_shared)
        with self._lock:
            row = self.conn.execute(f"SELECT * FROM memories WHERE id = ? AND {scope}", [memory_id] + args).fetchone()
        return row

    def canonical_memory_id(
        self, workspace_id: str, agent_id: str, visibility: str, session_id: str, canonical_key: str
    ) -> int | None:
        key = (canonical_key or "").strip()
        if not key:
            return None
        with self._lock:
            row = self.conn.execute(
                """
                SELECT id FROM memories
                WHERE workspace_id = ? AND agent_id = ? AND visibility = ?
                  AND (? <> 'session' OR session_id = ?) AND canonical_key = ?
                """,
                [workspace_id, agent_id, visibility, visibility, session_id or "", key],
            ).fetchone()
        return int(row["id"]) if row else None

    def exact_visible_durable_memory_id(
        self,
        *,
        workspace_id: str,
        agent_id: str,
        content: str,
        include_shared: bool = True,
    ) -> int | None:
        """Find equal active durable content without mutating access counters."""
        normalized = " ".join((content or "").casefold().split())
        if not normalized:
            return None
        visible = "(visibility = 'agent' AND agent_id = ?)"
        args: list[Any] = [workspace_id, _now(), agent_id]
        if include_shared:
            visible += " OR visibility = 'shared'"
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT id, content FROM memories
                WHERE workspace_id = ? AND archived = 0
                  AND (expires_at <= 0 OR expires_at > ?)
                  AND visibility IN ('agent', 'shared')
                  AND ({visible})
                ORDER BY id
                """,
                args,
            ).fetchall()
        for row in rows:
            if " ".join(str(row["content"] or "").casefold().split()) == normalized:
                return int(row["id"])
        return None


    def _fts_query(self, query: str, *, prefix: bool = False) -> str:
        terms = [t.lower() for t in re.findall(r"[\w]+", query or "") if len(t) > 1]
        # FTS5 syntax is unforgiving; quote every term. Prefix candidates are
        # enabled only when embeddings are unavailable so lexical fallback keeps
        # prior substring recall without broadening normal semantic fusion.
        suffix = "*" if prefix else ""
        return " OR ".join('"' + term.replace('"', '""') + '"' + suffix for term in terms[:12])

    def _lexical_overlap(self, query: str, row: sqlite3.Row) -> float:
        terms = [t.lower() for t in re.findall(r"[\w]+", query or "") if len(t) > 1]
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

    def search(
        self,
        *,
        workspace_id: str,
        agent_id: str,
        session_id: str,
        query: str,
        query_embedding: Sequence[float] | None,
        include_shared: bool = True,
        category: str = "",
        tags: Sequence[str] | None = None,
        visibility: str = "",
        source_agent_id: str = "",
        min_importance: float | None = None,
        updated_after: float | None = None,
        min_score: float | None = None,
        explain: bool = False,
        track_access: bool = True,
        limit: int = 8,
        _max_limit: int = 50,
    ) -> list[dict[str, Any]]:
        max_limit = max(50, min(int(_max_limit or 50), 500))
        limit = max(1, min(int(limit or 8), max_limit))
        pool_limit = max(50, min(500, limit * 12))
        tagset = {str(tag).lower() for tag in (tags or []) if tag}
        scope, args = self._scope_sql(workspace_id, agent_id, session_id, include_shared, alias="m")
        if category:
            scope += " AND m.category = ?"
            args.append(category)
        if visibility:
            if visibility not in {"agent", "shared", "session"}:
                raise ValueError("visibility must be one of: agent, shared, session")
            scope += " AND m.visibility = ?"
            args.append(visibility)
        if source_agent_id:
            scope += " AND m.source_agent_id = ?"
            args.append(source_agent_id)
        if min_importance is not None:
            scope += " AND m.importance >= ?"
            args.append(max(0.0, min(float(min_importance), 1.0)))
        if updated_after is not None:
            scope += " AND m.updated_at >= ?"
            args.append(float(updated_after))
        if tagset:
            tag_values = sorted(tagset)
            qmarks = ",".join("?" for _ in tag_values)
            scope += (
                " AND EXISTS (SELECT 1 FROM json_each(m.tags_json) AS memory_tag "
                f"WHERE lower(CAST(memory_tag.value AS TEXT)) IN ({qmarks}))"
            )
            args.extend(tag_values)

        fts_ranks: dict[int, int] = {}
        rows_by_id: dict[int, sqlite3.Row] = {}
        fts_query = self._fts_query(query, prefix=not bool(query_embedding))
        hybrid_sources = bool(fts_query and query_embedding)
        fts_candidate_limit = (pool_limit + 1) // 2 if hybrid_sources else pool_limit
        with self._lock:
            if fts_query:
                try:
                    fts_rows = list(
                        self.conn.execute(
                            f"""
                            SELECT m.*, bm25(memories_fts) AS _bm25
                            FROM memories_fts
                            JOIN memories m ON m.id = memories_fts.rowid
                            WHERE memories_fts MATCH ? AND {scope}
                            ORDER BY _bm25 ASC, m.id ASC
                            LIMIT ?
                            """,
                            [fts_query] + args + [fts_candidate_limit],
                        )
                    )
                except sqlite3.Error:
                    fts_rows = []
                for rank, row in enumerate(fts_rows, 1):
                    if tagset and not tagset.intersection({str(t).lower() for t in _json_loads(row["tags_json"], [])}):
                        continue
                    rid = int(row["id"])
                    rows_by_id[rid] = row
                    fts_ranks[rid] = rank

            vector_source_limit = max(0, pool_limit - len(fts_ranks)) if query_embedding else 0
            vector_rows = (
                list(
                    self.conn.execute(
                        f"""
                        SELECT m.* FROM memories m
                        WHERE {scope}
                        ORDER BY m.importance DESC, m.updated_at DESC, m.id ASC
                        LIMIT ?
                        """,
                        args + [vector_source_limit],
                    )
                )
                if vector_source_limit
                else []
            )
            fallback_rows = (
                list(
                    self.conn.execute(
                        f"""
                        SELECT m.* FROM memories m
                        WHERE {scope}
                        ORDER BY m.importance DESC, m.updated_at DESC, m.id ASC
                        LIMIT ?
                        """,
                        args + [pool_limit],
                    )
                )
                if not query_embedding and not fts_query
                else []
            )

        vector_ranked: list[tuple[float, sqlite3.Row]] = []
        for row in vector_rows:
            if tagset and not tagset.intersection({str(t).lower() for t in _json_loads(row["tags_json"], [])}):
                continue
            rows_by_id[int(row["id"])] = row
        # Score the deduplicated union, including FTS hits outside the bounded
        # importance/recency source window. The union cannot exceed pool_limit.
        for row in rows_by_id.values() if query_embedding else ():
            emb = _json_loads(row["embedding_json"], [])
            vector_score = cosine(query_embedding or [], emb)
            if vector_score > 0:
                vector_ranked.append((vector_score, row))
        vector_ranked.sort(key=lambda item: (-item[0], int(item[1]["id"])))
        vector_ranks = {int(row["id"]): rank for rank, (_score, row) in enumerate(vector_ranked, 1)}
        vector_scores = {int(row["id"]): score for score, row in vector_ranked}
        candidate_ids = set(fts_ranks) | set(vector_ranks)
        if not candidate_ids and fallback_rows:
            for row in fallback_rows:
                rows_by_id[int(row["id"])] = row
            candidate_ids = {int(row["id"]) for row in fallback_rows}
        if not candidate_ids and not query_embedding and fts_query:
            candidate_ids = set(fts_ranks)

        score_floor = None if min_score is None else max(0.0, min(float(min_score), 1.0))
        now = _now()
        scored: list[tuple[float, sqlite3.Row, dict[str, float]]] = []
        for rid in sorted(candidate_ids):
            row = rows_by_id[rid]
            vector_score = max(0.0, min(1.0, float(vector_scores.get(rid, 0.0))))
            lexical = self._lexical_overlap(query, row)
            bm25_signal = 1.0 / float(fts_ranks[rid]) if rid in fts_ranks else 0.0
            recency_days = max(0.0, (now - float(row["updated_at"])) / 86400.0)
            recency = 1.0 / (1.0 + recency_days / 30.0)
            importance = float(row["importance"] or 0.5)
            exact_lexical_boost = FUSION_CONFIG["exact_lexical_boost"] if lexical >= 1.0 else 0.0
            # Tuned over 4,744 candidate blends using synthetic adversarial
            # cases and duplicate-aware real-query embeddings from a disposable
            # copy of the shared AgentRecall DB. A narrow all-terms lexical boost
            # preserves exact matches without making partial overlap overpower a
            # strong semantic match.
            score = (
                (FUSION_CONFIG["bm25"] * bm25_signal)
                + (FUSION_CONFIG["lexical"] * lexical)
                + (FUSION_CONFIG["vector"] * vector_score)
                + (FUSION_CONFIG["importance"] * importance)
                + (FUSION_CONFIG["recency"] * recency)
                + exact_lexical_boost
            )
            score = max(0.0, min(1.0, score))
            if score_floor is not None and score < score_floor:
                continue
            scored.append(
                (
                    score,
                    row,
                    {
                        "vector": round(vector_score, 4),
                        "lexical": round(lexical, 4),
                        "bm25": round(bm25_signal, 4),
                        "exact_lexical_boost": round(exact_lexical_boost, 4),
                        "importance": round(importance, 4),
                        "recency": round(recency, 4),
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], int(item[1]["id"])))
        if explain:
            out = [self._format_row(row, score=score, score_explanation=details) for score, row, details in scored[:limit]]
        else:
            out = [self._format_row(row, score=score) for score, row, _details in scored[:limit]]
        ids = [r["id"] for r in out]
        if ids and track_access:
            self.record_access(ids)
        return out

    def record_access(self, memory_ids: Sequence[int]) -> None:
        ids = sorted({int(memory_id) for memory_id in memory_ids if int(memory_id) > 0})
        if not ids:
            return
        qmarks = ",".join("?" for _ in ids)
        with self._lock:
            self.conn.execute(
                f"UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id IN ({qmarks})",
                [_now()] + ids,
            )
            self.conn.commit()

    def get_memory(
        self,
        memory_id: int,
        workspace_id: str,
        agent_id: str,
        session_id: str,
        *,
        include_shared: bool = True,
    ) -> dict[str, Any] | None:
        row = self.get_visible(memory_id, workspace_id, agent_id, session_id, include_shared=include_shared)
        return self._format_row(row) if row else None

    def update_memory(
        self,
        memory_id: int,
        workspace_id: str,
        agent_id: str,
        session_id: str,
        *,
        allow_shared_mutation: bool = False,
        expected_updated_at: float | None = None,
        **updates: Any,
    ) -> bool:
        row = self.get_visible(memory_id, workspace_id, agent_id, session_id, include_shared=True)
        if not row:
            return False
        # Private/session memories are owner-only. Shared memories are readable
        # across the workspace, but are owner-only by default to prevent one
        # agent from silently rewriting another agent's published facts.
        if row["agent_id"] != agent_id and not (row["visibility"] == "shared" and allow_shared_mutation):
            return False
        if row["agent_id"] != agent_id and (
            "target_session_id" in updates
            or ("visibility" in updates and updates["visibility"] != "shared")
        ):
            return False
        allowed = {
            "content",
            "title",
            "summary",
            "category",
            "visibility",
            "target_session_id",
            "tags_json",
            "metadata_json",
            "embedding_json",
            "embedding_model",
            "embedding_dimensions",
            "importance",
            "confidence",
            "archived",
            "expires_at",
        }
        sets = []
        args: list[Any] = []
        for key, val in updates.items():
            if key in allowed:
                column = "session_id" if key == "target_session_id" else key
                sets.append(f"{column} = ?")
                args.append(val)
        if not sets:
            return True
        sets.append("updated_at = ?")
        args.append(_now())
        mutation_version = float(expected_updated_at) if expected_updated_at is not None else float(row["updated_at"])
        where = "id = ? AND workspace_id = ? AND updated_at = ?"
        args.extend([memory_id, workspace_id, mutation_version])
        if row["agent_id"] == agent_id:
            where += " AND agent_id = ?"
            args.append(agent_id)
        else:
            where += " AND visibility = 'shared'"
        with self._lock, self.conn:
            cursor = self.conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE {where}", args)
        return cursor.rowcount > 0

    def delete_memory(
        self,
        memory_id: int,
        workspace_id: str,
        agent_id: str,
        session_id: str,
        *,
        allow_shared_mutation: bool = False,
    ) -> bool:
        row = self.get_visible(memory_id, workspace_id, agent_id, session_id, include_shared=True)
        if not row:
            return False
        if row["agent_id"] != agent_id and not (row["visibility"] == "shared" and allow_shared_mutation):
            return False
        where = "id = ? AND workspace_id = ? AND updated_at = ?"
        args: list[Any] = [memory_id, workspace_id, float(row["updated_at"])]
        if row["agent_id"] == agent_id:
            where += " AND agent_id = ?"
            args.append(agent_id)
        else:
            where += " AND visibility = 'shared'"
        with self._lock:
            cursor = self.conn.execute(f"DELETE FROM memories WHERE {where}", args)
            self.conn.commit()
        return cursor.rowcount > 0

    def physical_cleanup(
        self,
        *,
        purge_expired: bool = True,
        optimize_fts: bool = True,
        vacuum: bool = False,
    ) -> dict[str, Any]:
        now = _now()
        with self._lock:
            expired_deleted = 0
            if purge_expired:
                cur = self.conn.execute("DELETE FROM memories WHERE expires_at > 0 AND expires_at <= ?", [now])
                expired_deleted = int(cur.rowcount if cur.rowcount is not None else 0)
            cur = self.conn.execute(
                """
                DELETE FROM links
                WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = links.from_id AND m.workspace_id = links.workspace_id)
                   OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = links.to_id AND m.workspace_id = links.workspace_id)
                """
            )
            orphan_links_deleted = int(cur.rowcount if cur.rowcount is not None else 0)
            fts_optimized = False
            if optimize_fts:
                with suppress(Exception):
                    self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')")
                    fts_optimized = True
            self.conn.commit()
        vacuumed = False
        if vacuum:
            with self._lock:
                self.conn.execute("VACUUM")
                vacuumed = True
        return {
            "expired_memories_deleted": expired_deleted,
            "orphan_links_deleted": orphan_links_deleted,
            "fts_optimized": fts_optimized,
            "vacuumed": vacuumed,
        }

    def stats(
        self, workspace_id: str, agent_id: str, session_id: str = "", include_shared: bool = True
    ) -> dict[str, Any]:
        scope, args = self._scope_sql(workspace_id, agent_id, session_id, include_shared)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT visibility, agent_id, category, COUNT(*) n FROM memories WHERE {scope} GROUP BY visibility, agent_id, category",
                args,
            ).fetchall()
        return {
            "workspace_id": workspace_id,
            "current_agent_id": agent_id,
            "db_path": str(self.db_path),
            "buckets": [dict(r) for r in rows],
        }

    def health(self) -> dict[str, Any]:
        with self._lock:
            journal_mode = str(self.conn.execute("PRAGMA journal_mode").fetchone()[0])
            busy_timeout_ms = int(self.conn.execute("PRAGMA busy_timeout").fetchone()[0])
            quick_check = str(self.conn.execute("PRAGMA quick_check").fetchone()[0])
        return {
            "db_path": str(self.db_path),
            "journal_mode": journal_mode,
            "busy_timeout_ms": busy_timeout_ms,
            "quick_check": quick_check,
        }

    def _format_row(
        self,
        row: sqlite3.Row,
        score: float = 0.0,
        score_explanation: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        result = {
            "id": int(row["id"]),
            "score": round(float(score), 4),
            "workspace_id": row["workspace_id"],
            "agent_id": row["agent_id"],
            "source_agent_id": row["source_agent_id"],
            "visibility": row["visibility"],
            "session_id": row["session_id"],
            "category": row["category"],
            "title": row["title"],
            "content": row["content"],
            "summary": row["summary"],
            "tags": _json_loads(row["tags_json"], []),
            "metadata": _json_loads(row["metadata_json"], {}),
            "importance": float(row["importance"]),
            "confidence": float(row["confidence"]),
            "canonical_key": row["canonical_key"],
            "expires_at": float(row["expires_at"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if score_explanation is not None:
            result["score_explanation"] = score_explanation
        return result
