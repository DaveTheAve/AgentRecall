from __future__ import annotations

import multiprocessing
from pathlib import Path

from agent_recall_store import AgentRecallStore


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


def test_store_configures_busy_timeout_and_wal(tmp_path):
    store = AgentRecallStore(tmp_path / "concurrency.db", busy_timeout_ms=12_345)
    assert store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 12_345
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    store.close()


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
