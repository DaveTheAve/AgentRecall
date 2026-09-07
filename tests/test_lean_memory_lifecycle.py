from __future__ import annotations

import sqlite3
import time

from conftest import FakeEmbedder

import agent_recall_store
from agent_recall_core import AgentIdentity, AgentRecallCore
from agent_recall_store import AgentRecallStore


def test_hybrid_search_processes_no_more_than_the_configured_candidate_pool(tmp_path, monkeypatch):
    store = AgentRecallStore(tmp_path / "bounded-search.db")
    for index in range(700):
        store.add_memory(
            workspace_id="ws",
            agent_id="hermes",
            content=f"bounded candidate fact {index}",
            embedding=[1.0, float(index % 7) / 10.0],
            embedding_model="fake",
            importance=float(index % 10) / 10.0,
        )

    cosine_calls = 0
    lexical_calls = 0
    real_cosine = agent_recall_store.cosine
    real_lexical = store._lexical_overlap

    def counted_cosine(left, right):
        nonlocal cosine_calls
        cosine_calls += 1
        return real_cosine(left, right)

    def counted_lexical(query, row):
        nonlocal lexical_calls
        lexical_calls += 1
        return real_lexical(query, row)

    monkeypatch.setattr(agent_recall_store, "cosine", counted_cosine)
    monkeypatch.setattr(store, "_lexical_overlap", counted_lexical)
    results = store.search(
        workspace_id="ws",
        agent_id="hermes",
        session_id="",
        query="bounded candidate",
        query_embedding=[1.0, 0.0],
        limit=50,
        _max_limit=500,
    )

    assert len(results) == 50
    # Both signals score the same bounded candidate union, not just their own
    # source window; FTS hits must retain their semantic contribution.
    assert cosine_calls == lexical_calls
    assert cosine_calls <= 500
    assert lexical_calls <= 500
    store.close()


def make_core(tmp_path) -> AgentRecallCore:
    core = AgentRecallCore(
        {
            "db_path": str(tmp_path / "lean.db"),
            "workspace_id": "ws",
            "agent_id": "hermes",
            "embedding_base_url": "",
            "embedding_model": "fake",
            "shared_recall": True,
        },
        AgentIdentity("ws", "hermes", "s1"),
    )
    core.embedder = FakeEmbedder()
    return core


def test_canonical_key_remember_updates_in_place_and_removes_old_search_projection(tmp_path):
    core = make_core(tmp_path)

    first = core.remember(
        {
            "canonical_key": "user.preference.response_length",
            "content": "The user prefers extremely verbose responses.",
            "visibility": "agent",
            "category": "preference",
        }
    )
    second = core.remember(
        {
            "canonical_key": "user.preference.response_length",
            "content": "The user prefers concise terminal-friendly responses.",
            "visibility": "agent",
            "category": "preference",
        }
    )

    assert second["id"] == first["id"]
    assert second["action"] == "updated"
    row = core.get_memory(first["id"])["memory"]
    assert row["canonical_key"] == "user.preference.response_length"
    assert row["content"] == "The user prefers concise terminal-friendly responses."

    with sqlite3.connect(tmp_path / "lean.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'verbose'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'concise'").fetchone()[0] == 1
    core.close()


def test_durable_canonical_upsert_crosses_sessions_but_session_scope_stays_isolated(tmp_path):
    core = make_core(tmp_path)
    durable_first = core.remember(
        {
            "content": "The durable fact from session one",
            "canonical_key": "project.fact.cross_session",
            "visibility": "agent",
        }
    )
    session_first = core.remember(
        {
            "content": "Session one temporary fact",
            "canonical_key": "temporary.fact",
            "visibility": "session",
        }
    )

    core.rotate_session("s2")
    durable_second = core.remember(
        {
            "content": "The durable fact updated in session two",
            "canonical_key": "project.fact.cross_session",
            "visibility": "agent",
        }
    )
    session_second = core.remember(
        {
            "content": "Session two temporary fact",
            "canonical_key": "temporary.fact",
            "visibility": "session",
        }
    )

    assert durable_second["action"] == "updated"
    assert durable_second["id"] == durable_first["id"]
    assert session_second["action"] == "added"
    assert session_second["id"] != session_first["id"]
    rows = core.store.conn.execute(
        "SELECT visibility, canonical_key, COUNT(*) FROM memories GROUP BY visibility, canonical_key"
    ).fetchall()
    assert {(row[0], row[1], row[2]) for row in rows} == {
        ("agent", "project.fact.cross_session", 1),
        ("session", "temporary.fact", 2),
    }
    core.close()


def test_bm25_exact_match_beats_semantic_only_neighbor(tmp_path):
    store = AgentRecallStore(tmp_path / "rank.db")
    exact_id = store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="agent",
        content="The inference endpoint listens on a configured service port.",
        embedding=[0.0, 1.0],
        embedding_model="fake",
        importance=0.5,
    )
    store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="agent",
        content="General unrelated model infrastructure note.",
        embedding=[1.0, 0.0],
        embedding_model="fake",
        importance=0.5,
    )

    results = store.search(
        workspace_id="ws",
        agent_id="hermes",
        session_id="",
        query="inference configured service port",
        query_embedding=[1.0, 0.0],
        explain=True,
        limit=2,
    )

    assert results[0]["id"] == exact_id
    assert results[0]["score_explanation"]["bm25"] > 0
    store.close()


def test_strong_semantic_match_beats_partial_lexical_neighbor(tmp_path):
    store = AgentRecallStore(tmp_path / "balanced-fusion.db")
    semantic_id = store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="agent",
        content="PostgreSQL with columnar extensions is the chosen warehouse.",
        embedding=[1.0, 0.0],
        embedding_model="fake",
    )
    store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="agent",
        content="Preferred database migration checklist.",
        embedding=[0.0, 1.0],
        embedding_model="fake",
    )

    results = store.search(
        workspace_id="ws",
        agent_id="hermes",
        session_id="",
        query="preferred database for analytics",
        query_embedding=[1.0, 0.0],
        limit=2,
    )

    assert results[0]["id"] == semantic_id
    store.close()


def test_fusion_ties_use_memory_id_as_a_deterministic_secondary_order(tmp_path):
    store = AgentRecallStore(tmp_path / "deterministic-ties.db")
    ids = [
        store.add_memory(
            workspace_id="ws",
            agent_id="hermes",
            source_agent_id="hermes",
            visibility="agent",
            content="identical deterministic retrieval candidate",
            embedding=[1.0, 0.0],
            embedding_model="fake",
            importance=0.5,
        )
        for _ in range(3)
    ]
    with store._lock:
        store.conn.execute("UPDATE memories SET created_at = 1000, updated_at = 1000")
        store.conn.commit()

    first = store.search(
        workspace_id="ws",
        agent_id="hermes",
        session_id="",
        query="identical deterministic retrieval candidate",
        query_embedding=[1.0, 0.0],
        limit=3,
    )
    second = store.search(
        workspace_id="ws",
        agent_id="hermes",
        session_id="",
        query="identical deterministic retrieval candidate",
        query_embedding=[1.0, 0.0],
        limit=3,
    )

    assert [row["id"] for row in first] == ids
    assert [row["id"] for row in second] == ids
    store.close()


def test_lexical_fallback_preserves_prefix_substring_recall_without_embeddings(tmp_path):
    store = AgentRecallStore(tmp_path / "prefix-fallback.db")
    expected_id = store.add_memory(
        workspace_id="shared",
        agent_id="hermes",
        visibility="agent",
        content="PostgreSQL handbook",
        embedding=[],
        embedding_model="",
    )

    results = store.search(
        workspace_id="shared",
        agent_id="hermes",
        session_id="",
        query="postgres",
        query_embedding=None,
        limit=5,
        track_access=False,
    )

    assert [row["id"] for row in results] == [expected_id]
    store.close()


def test_filtered_search_scans_beyond_the_first_unfiltered_candidate_page(tmp_path):
    store = AgentRecallStore(tmp_path / "filtered-candidate-window.db")
    for _ in range(55):
        store.add_memory(
            workspace_id="shared",
            agent_id="hermes",
            visibility="agent",
            tags=["noise"],
            content="target candidate",
            embedding=[],
            embedding_model="",
        )
    expected_id = store.add_memory(
        workspace_id="shared",
        agent_id="hermes",
        visibility="agent",
        tags=["wanted"],
        content="target candidate wanted",
        embedding=[],
        embedding_model="",
    )

    results = store.search(
        workspace_id="shared",
        agent_id="hermes",
        session_id="",
        query="target candidate",
        query_embedding=None,
        tags=["wanted"],
        limit=1,
        track_access=False,
    )

    assert [row["id"] for row in results] == [expected_id]
    empty_query_results = store.search(
        workspace_id="shared",
        agent_id="hermes",
        session_id="",
        query="",
        query_embedding=None,
        tags=["wanted"],
        limit=1,
        track_access=False,
    )
    assert [row["id"] for row in empty_query_results] == [expected_id]
    store.close()


def test_expired_memories_are_hidden_and_physical_cleanup_purges_rows_and_links(tmp_path):
    store = AgentRecallStore(tmp_path / "cleanup.db")
    expired_id = store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="agent",
        content="temporary expired fact",
        embedding=[1.0],
        embedding_model="fake",
        expires_at=time.time() - 60,
    )
    live_id = store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="agent",
        content="durable live fact",
        embedding=[1.0],
        embedding_model="fake",
    )
    with store._lock:
        store.conn.execute(
            "INSERT INTO links (workspace_id, from_id, to_id, relation, created_at) VALUES (?, ?, ?, ?, ?)",
            ("ws", live_id, expired_id, "related", time.time()),
        )
        store.conn.commit()

    before_cleanup = store.search(
        workspace_id="ws",
        agent_id="hermes",
        session_id="",
        query="temporary expired",
        query_embedding=[1.0],
        limit=5,
    )
    assert expired_id not in {row["id"] for row in before_cleanup}

    report = store.physical_cleanup(purge_expired=True, optimize_fts=True, vacuum=False)

    assert report["expired_memories_deleted"] == 1
    assert report["orphan_links_deleted"] == 1
    with sqlite3.connect(tmp_path / "cleanup.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (expired_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (live_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM links").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'temporary'").fetchone()[0] == 0
    store.close()


def test_expiration_is_optional_and_cleanup_preserves_default_forever_memories(tmp_path):
    store = AgentRecallStore(tmp_path / "forever.db")
    memory_id = store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="agent",
        content="durable memory with no expiration sticks around forever by default",
        embedding=[1.0],
        embedding_model="fake",
    )

    before = store.get_memory(memory_id, "ws", "hermes", "")
    report = store.physical_cleanup(purge_expired=True, optimize_fts=True, vacuum=False)
    after = store.get_memory(memory_id, "ws", "hermes", "")

    assert before is not None
    assert before["expires_at"] == 0.0
    assert report["expired_memories_deleted"] == 0
    assert after is not None
    assert after["id"] == memory_id
    store.close()
