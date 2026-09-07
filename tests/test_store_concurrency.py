from __future__ import annotations

import multiprocessing
import sqlite3
import threading
from pathlib import Path

from conftest import FakeEmbedder

from agent_recall_core import AgentIdentity, AgentRecallCore
from agent_recall_store import SCHEMA, AgentRecallStore


def _write_memories(db_path: str, agent_id: str, count: int, queue) -> None:
    try:
        store = AgentRecallStore(db_path, busy_timeout_ms=10_000)
        for index in range(count):
            store.add_memory(
                workspace_id="shared",
                agent_id=agent_id,
                source_agent_id=agent_id,
                session_id=f"{agent_id}-session",
                visibility="shared",
                category="concurrency",
                content=f"{agent_id} memory {index}",
                embedding=[],
                embedding_model="",
            )
        store.close()
        queue.put(None)
    except Exception as exc:  # pragma: no cover - returned to parent for assertion
        queue.put(repr(exc))


def _open_existing_store(db_path: str, gate, queue) -> None:
    gate.wait(timeout=10)
    try:
        store = AgentRecallStore(db_path, busy_timeout_ms=10_000)
        store.close()
        queue.put(None)
    except Exception as exc:  # pragma: no cover - returned to parent for assertion
        queue.put(f"{type(exc).__name__}: {exc}")


def test_store_configures_busy_timeout_and_wal(tmp_path):
    store = AgentRecallStore(tmp_path / "concurrency.db", busy_timeout_ms=12_345)
    assert store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 12_345
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    store.close()


def test_concurrent_first_open_migrates_legacy_schema_without_host_failures(tmp_path):
    context = multiprocessing.get_context("spawn")
    legacy_schema = SCHEMA.replace("    canonical_key TEXT NOT NULL DEFAULT '',\n", "").replace(
        "    expires_at REAL NOT NULL DEFAULT 0,\n", ""
    )
    for round_index in range(3):
        db_path = str(tmp_path / f"legacy-upgrade-{round_index}.db")
        with sqlite3.connect(db_path) as conn:
            conn.executescript(legacy_schema)
        gate = context.Event()
        queue = context.Queue()
        processes = [context.Process(target=_open_existing_store, args=(db_path, gate, queue)) for _ in range(8)]
        for process in processes:
            process.start()
        gate.set()
        for process in processes:
            process.join(timeout=20)
            assert not process.is_alive()
            assert process.exitcode == 0
        assert [queue.get(timeout=5) for _ in processes] == [None] * len(processes)
        migrated = AgentRecallStore(db_path)
        columns = {row[1] for row in migrated.conn.execute("PRAGMA table_info(memories)")}
        assert {"canonical_key", "expires_at"} <= columns
        assert migrated.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        migrated.close()


def test_multiple_host_processes_can_write_same_workspace(tmp_path):
    db_path = str(Path(tmp_path) / "shared.db")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_write_memories, args=(db_path, agent_id, 30, queue))
        for agent_id in ("hermes", "openclaw")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert [queue.get(timeout=5) for _ in processes] == [None, None]

    store = AgentRecallStore(db_path)
    count = store.conn.execute("SELECT COUNT(*) FROM memories WHERE workspace_id = 'shared'").fetchone()[0]
    assert count == 60
    store.close()


def test_concurrent_canonical_writes_report_one_add_and_one_update(tmp_path):
    db_path = str(Path(tmp_path) / "canonical-race.db")
    cores = []
    for agent_session in ("s1", "s2"):
        core = AgentRecallCore(
            {
                "db_path": db_path,
                "workspace_id": "shared",
                "agent_id": "hermes",
                "embedding_base_url": "",
                "embedding_model": "fake",
            },
            AgentIdentity("shared", "hermes", agent_session),
        )
        core.embedder = FakeEmbedder()
        cores.append(core)
    barrier = threading.Barrier(2)
    results = []

    def write(core, content):
        barrier.wait(timeout=5)
        results.append(
            core.remember(
                {
                    "content": content,
                    "canonical_key": "project.concurrent.fact",
                    "visibility": "agent",
                }
            )
        )

    threads = [
        threading.Thread(target=write, args=(core, f"canonical value {index}"))
        for index, core in enumerate(cores)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert sorted(result["action"] for result in results) == ["added", "updated"]
    assert len({result["id"] for result in results}) == 1
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    for core in cores:
        core.close()


def test_schema_migration_collapses_legacy_cross_session_durable_duplicates(tmp_path):
    db_path = tmp_path / "legacy-canonical-index.db"
    legacy = AgentRecallStore(db_path)
    with legacy._lock:
        legacy.conn.execute("DROP INDEX idx_mem_canonical_durable")
        legacy.conn.execute("DROP INDEX idx_mem_canonical_session")
        legacy.conn.execute(
            "CREATE UNIQUE INDEX idx_mem_canonical "
            "ON memories(workspace_id, agent_id, visibility, session_id, canonical_key) "
            "WHERE canonical_key <> ''"
        )
        legacy.conn.commit()
    first = legacy.add_memory(
        workspace_id="shared",
        agent_id="hermes",
        session_id="s1",
        visibility="agent",
        content="legacy canonical value one",
        embedding=[],
        embedding_model="",
        canonical_key="project.legacy.fact",
    )
    second = legacy.add_memory(
        workspace_id="shared",
        agent_id="hermes",
        session_id="s2",
        visibility="agent",
        content="legacy canonical value two",
        embedding=[],
        embedding_model="",
        canonical_key="project.legacy.fact",
    )
    assert first != second
    legacy.close()

    migrated = AgentRecallStore(db_path)
    rows = migrated.conn.execute(
        "SELECT id, content FROM memories WHERE canonical_key = 'project.legacy.fact'"
    ).fetchall()

    assert [(row["id"], row["content"]) for row in rows] == [(second, "legacy canonical value two")]
    indexes = {row[1] for row in migrated.conn.execute("PRAGMA index_list(memories)")}
    assert {"idx_mem_canonical_durable", "idx_mem_canonical_session"} <= indexes
    assert "idx_mem_canonical" not in indexes
    migrated.close()


def test_schema_migration_recovers_when_another_process_wins_the_column_race():
    class Cursor:
        def __init__(self, rows):
            self.rows = rows

        def fetchall(self):
            return self.rows

    class RacingConnection:
        def __init__(self):
            self.canonical_visible = False

        def execute(self, sql):
            if sql == "PRAGMA table_info(memories)":
                columns = ["id", "workspace_id"]
                if self.canonical_visible:
                    columns.append("canonical_key")
                return Cursor([(index, name) for index, name in enumerate(columns)])
            if "ADD COLUMN canonical_key" in sql:
                self.canonical_visible = True
                raise sqlite3.OperationalError("duplicate column name: canonical_key")
            if "ADD COLUMN expires_at" in sql or sql.lstrip().startswith(("CREATE ", "DROP INDEX", "DELETE FROM")):
                return Cursor([])
            raise AssertionError(f"unexpected SQL: {sql}")

    store = AgentRecallStore.__new__(AgentRecallStore)
    store.conn = RacingConnection()

    store._migrate_schema_locked()
