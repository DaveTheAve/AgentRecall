from __future__ import annotations

import json
import math
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
        # Qwen/llama.cpp often rejects dimensions; only send when explicitly positive.
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
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            with suppress(Exception):
                self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

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
    ) -> int:
        if visibility not in {"agent", "shared", "session"}:
            raise ValueError("visibility must be one of: agent, shared, session")
        content = normalize_text(content)
        if not content:
            raise ValueError("content is required")
        ts = _now()
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO memories (
                  workspace_id, agent_id, source_agent_id, user_id, session_id, visibility,
                  category, title, content, summary, tags_json, metadata_json, embedding_json,
                  embedding_model, embedding_dimensions, importance, confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id, agent_id, source_agent_id or agent_id, user_id or "", session_id or "",
                    visibility, category or "general", title or "", content, summary or "",
                    _json_dumps(list(tags or [])), _json_dumps(metadata or {}), _json_dumps(list(embedding)),
                    embedding_model or "", len(embedding), float(importance), float(confidence), ts, ts,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def _scope_sql(self, workspace_id: str, agent_id: str, session_id: str, include_shared: bool = True) -> tuple[str, list[Any]]:
        clauses = ["workspace_id = ?", "archived = 0"]
        args: list[Any] = [workspace_id]
        visible = ["(visibility = 'agent' AND agent_id = ?)"]
        args.append(agent_id)
        visible.append("(visibility = 'session' AND agent_id = ? AND session_id = ?)")
        args.extend([agent_id, session_id or ""])
        if include_shared:
            visible.append("visibility = 'shared'")
        clauses.append("(" + " OR ".join(visible) + ")")
        return " AND ".join(clauses), args

    def get_visible(self, memory_id: int, workspace_id: str, agent_id: str, session_id: str, include_shared: bool = True) -> sqlite3.Row | None:
        scope, args = self._scope_sql(workspace_id, agent_id, session_id, include_shared)
        with self._lock:
            row = self.conn.execute(f"SELECT * FROM memories WHERE id = ? AND {scope}", [memory_id] + args).fetchone()
        return row

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
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 8), 50))
        scope, args = self._scope_sql(workspace_id, agent_id, session_id, include_shared)
        if category:
            scope += " AND category = ?"
            args.append(category)
        with self._lock:
            rows = list(self.conn.execute(f"SELECT * FROM memories WHERE {scope} ORDER BY updated_at DESC", args))
        tagset = {t.lower() for t in (tags or []) if t}
        if tagset:
            rows = [r for r in rows if tagset.intersection({str(t).lower() for t in _json_loads(r['tags_json'], [])})]

        q = (query or "").strip().lower()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            emb = _json_loads(row["embedding_json"], [])
            vector_score = cosine(query_embedding or [], emb) if query_embedding else 0.0
            hay = " ".join([row["title"] or "", row["content"] or "", row["summary"] or "", row["category"] or "", row["tags_json"] or ""]).lower()
            lexical = 0.0
            if q:
                terms = [t for t in q.replace('"', ' ').split() if len(t) > 1]
                if terms:
                    lexical = sum(1 for t in terms if t in hay) / len(terms)
            recency_days = max(0.0, (_now() - float(row["updated_at"])) / 86400.0)
            recency = 1.0 / (1.0 + recency_days / 30.0)
            importance = float(row["importance"] or 0.5)
            score = (0.58 * vector_score) + (0.25 * lexical) + (0.10 * importance) + (0.07 * recency)
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [self._format_row(row, score=score) for score, row in scored[:limit]]
        ids = [r["id"] for r in out]
        if ids:
            qmarks = ",".join("?" for _ in ids)
            with self._lock:
                self.conn.execute(f"UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id IN ({qmarks})", [_now()] + ids)
                self.conn.commit()
        return out

    def update_memory(
        self,
        memory_id: int,
        workspace_id: str,
        agent_id: str,
        session_id: str,
        *,
        allow_shared_mutation: bool = False,
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
        allowed = {"content", "title", "summary", "category", "visibility", "tags_json", "metadata_json", "embedding_json", "embedding_model", "embedding_dimensions", "importance", "confidence", "archived"}
        sets = []
        args: list[Any] = []
        for key, val in updates.items():
            if key in allowed:
                sets.append(f"{key} = ?")
                args.append(val)
        if not sets:
            return True
        sets.append("updated_at = ?")
        args.append(_now())
        args.append(memory_id)
        with self._lock:
            self.conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", args)
            self.conn.commit()
        return True

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
        with self._lock:
            self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.conn.commit()
        return True

    def stats(self, workspace_id: str, agent_id: str, session_id: str = "", include_shared: bool = True) -> dict[str, Any]:
        scope, args = self._scope_sql(workspace_id, agent_id, session_id, include_shared)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT visibility, agent_id, category, COUNT(*) n FROM memories WHERE {scope} GROUP BY visibility, agent_id, category",
                args,
            ).fetchall()
        return {"workspace_id": workspace_id, "current_agent_id": agent_id, "db_path": str(self.db_path), "buckets": [dict(r) for r in rows]}

    def _format_row(self, row: sqlite3.Row, score: float = 0.0) -> dict[str, Any]:
        return {
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
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
