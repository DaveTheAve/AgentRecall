from __future__ import annotations

import json
import threading
import time

import pytest
from conftest import FakeEmbedder

from agent_recall_core import AgentIdentity, AgentRecallCore, AgentRecallError, load_config


def make_core(tmp_path, *, agent_id: str, session_id: str = "s1") -> AgentRecallCore:
    core = AgentRecallCore(
        {
            "db_path": str(tmp_path / "shared.db"),
            "workspace_id": "shared-workspace",
            "agent_id": agent_id,
            "embedding_base_url": "",
            "embedding_model": "fake",
            "shared_recall": True,
        },
        AgentIdentity("shared-workspace", agent_id, session_id),
    )
    core.embedder = FakeEmbedder()
    return core


def test_core_enforces_cross_host_visibility_and_provenance(tmp_path):
    hermes = make_core(tmp_path, agent_id="hermes")
    private = hermes.remember({"content": "Hermes-only preference", "visibility": "agent"})
    shared = hermes.remember(
        {
            "content": "Shared operational convention",
            "visibility": "shared",
            "metadata": {"source": "architecture-test"},
        }
    )
    hermes.close()

    openclaw = make_core(tmp_path, agent_id="openclaw")
    result = openclaw.search({"query": "operational convention", "explain": True})
    assert [row["id"] for row in result["results"]] == [shared["id"]]
    assert result["results"][0]["metadata"]["source"] == "architecture-test"
    assert result["results"][0]["score_explanation"]["lexical"] > 0
    assert openclaw.get_memory(shared["id"])["memory"]["source_agent_id"] == "hermes"
    assert openclaw.get_memory(private["id"])["success"] is False
    openclaw.close()


def test_core_rejects_new_session_memory_without_a_session_identity(tmp_path):
    core = make_core(tmp_path, agent_id="openclaw", session_id="")

    with pytest.raises(AgentRecallError, match="non-empty session identity"):
        core.remember({"content": "orphan session memory", "visibility": "session"})

    core.close()


def test_core_rejects_session_visibility_update_without_a_session_identity(tmp_path):
    core = make_core(tmp_path, agent_id="openclaw", session_id="")
    memory = core.remember({"content": "agent memory", "visibility": "agent"})

    with pytest.raises(AgentRecallError, match="non-empty session identity"):
        core.update(memory["id"], {"visibility": "session"})

    stored = core.get_memory(memory["id"])["memory"]
    assert stored["visibility"] == "agent"
    core.close()


def test_core_session_visibility_update_moves_memory_to_current_session(tmp_path):
    core = make_core(tmp_path, agent_id="openclaw", session_id="old-session")
    memory = core.remember({"content": "move to current session", "visibility": "agent"})
    core.rotate_session("new-session")

    updated = core.update(memory["id"], {"visibility": "session"})

    assert updated["success"] is True
    assert core.get_memory(memory["id"])["memory"]["session_id"] == "new-session"
    core.rotate_session("old-session")
    assert core.get_memory(memory["id"])["success"] is False
    core.close()


def test_core_import_markdown_fails_closed_with_empty_import_roots(tmp_path):
    core = make_core(tmp_path, agent_id="mcp")
    core.config["import_roots"] = []
    note = tmp_path / "outside.md"
    note.write_text("must not be imported without an allowlisted root", encoding="utf-8")

    with pytest.raises(AgentRecallError, match="No import_roots are configured"):
        core.import_markdown({"path": str(note)})

    assert core.search({"query": "allowlisted root"})["results"] == []
    core.close()


def test_core_import_markdown_rejects_blank_import_root_entries(tmp_path, monkeypatch):
    core = make_core(tmp_path, agent_id="mcp")
    core.config["import_roots"] = [""]
    monkeypatch.chdir(tmp_path)
    note = tmp_path / "cwd.md"
    note.write_text("blank roots must not trust the current directory", encoding="utf-8")

    with pytest.raises(AgentRecallError, match="No import_roots are configured"):
        core.import_markdown({"path": str(note)})

    assert core.search({"query": "current directory"})["results"] == []
    core.close()


def test_prefetch_context_obeys_budget_and_reports_recall_reasons(tmp_path):
    core = make_core(tmp_path, agent_id="hermes")
    core.remember(
        {
            "title": "Formatting preference",
            "content": "The user prefers concise terminal-friendly technical reports.",
            "visibility": "agent",
            "importance": 0.9,
        }
    )

    result = core.prefetch_context("How should I format this report?", limit=4, max_chars=220, explain=True)

    assert result["success"] is True
    assert len(result["context"]) <= 220
    assert "concise terminal-friendly" in result["context"]
    assert result["results"][0]["score_explanation"]["importance"] == 0.9
    assert core.search({"query": "unrelated", "min_score": 0.99})["results"] == []
    core.close()


def test_prefetch_access_telemetry_only_tracks_rows_injected_within_the_character_budget(tmp_path):
    core = make_core(tmp_path, agent_id="hermes")
    query = "bounded telemetry fact"
    embedding = core.embedder.embed(query)
    first_id = core.store.add_memory(
        workspace_id="shared-workspace",
        agent_id="hermes",
        content="Bounded telemetry fact one.",
        embedding=embedding,
        embedding_model="fake",
        importance=1.0,
    )
    second_id = core.store.add_memory(
        workspace_id="shared-workspace",
        agent_id="hermes",
        content="Bounded telemetry fact two should not fit in the injected context.",
        embedding=embedding,
        embedding_model="fake",
        importance=0.1,
    )
    first_only = core.prefetch_context(query, limit=1, max_chars=2_000, track_access=False)
    injected_id = int(first_only["results"][0]["id"])
    omitted_id = second_id if injected_id == first_id else first_id
    result = core.prefetch_context(
        query,
        limit=2,
        max_chars=len(first_only["context"]) + 10,
        track_access=True,
    )

    assert str(injected_id) in result["context"]
    assert str(omitted_id) not in result["context"]
    accessed = core.store.conn.execute(
        "SELECT id FROM memories WHERE access_count > 0 ORDER BY id"
    ).fetchall()
    assert [row[0] for row in accessed] == [injected_id]
    core.close()


def test_prefetch_negative_limit_clamps_to_one_without_expanding_the_private_window(tmp_path):
    core = make_core(tmp_path, agent_id="hermes")
    for index in range(4):
        core.remember({"content": f"bounded negative limit fact {index}", "category": "preference"})

    result = core.prefetch_context("bounded negative limit fact", limit=-1, max_chars=4_000)

    assert result["count"] == 1
    accessed = core.store.conn.execute("SELECT COUNT(*) FROM memories WHERE access_count > 0").fetchone()[0]
    assert accessed == 1
    core.close()


def test_prefetch_context_deduplicates_same_canonical_fact_across_visibility_scopes(tmp_path):
    core = make_core(tmp_path, agent_id="hermes")
    for visibility in ("agent", "shared"):
        core.remember(
            {
                "title": "Preferred response format",
                "content": "The user prefers concise terminal-friendly technical reports.",
                "visibility": visibility,
                "canonical_key": "user.preference.response_format",
            }
        )

    result = core.prefetch_context("concise terminal reports", limit=6, max_chars=2_000)

    assert result["count"] == 1
    assert len(result["results"]) == 1
    assert result["context"].count("concise terminal-friendly") == 1
    core.close()


def test_prefetch_overfetch_keeps_unique_fact_after_many_cross_agent_duplicates(tmp_path):
    core = make_core(tmp_path, agent_id="hermes")
    for index in range(55):
        core.store.add_memory(
            workspace_id=core.identity.workspace_id,
            agent_id=f"peer-{index}",
            source_agent_id=f"peer-{index}",
            visibility="shared",
            content=f"target duplicate projection {index}",
            embedding=[],
            embedding_model="",
            canonical_key="shared.target.duplicate",
        )
    unique_id = core.store.add_memory(
        workspace_id=core.identity.workspace_id,
        agent_id="peer-unique",
        source_agent_id="peer-unique",
        visibility="shared",
        content="target unique supporting fact",
        embedding=[],
        embedding_model="",
        canonical_key="shared.target.unique",
    )

    result = core.prefetch_context("target duplicate", limit=6, max_chars=4_000)

    assert unique_id in {row["id"] for row in result["results"]}
    assert result["count"] == 2
    accessed = core.store.conn.execute("SELECT id FROM memories WHERE access_count > 0 ORDER BY id").fetchall()
    assert [row[0] for row in accessed] == sorted(row["id"] for row in result["results"])
    core.close()


def test_update_can_clear_optional_expiration_before_it_elapses(tmp_path):
    core = make_core(tmp_path, agent_id="hermes")
    memory = core.remember(
        {
            "content": "Temporary deployment window",
            "expires_at": time.time() + 3_600,
        }
    )

    assert core.update(memory["id"], {"expires_at": 0})["updated"] is True
    stored = core.get_memory(memory["id"])["memory"]
    assert stored["expires_at"] == 0.0
    core.close()


def test_core_capabilities_and_health_are_host_neutral(tmp_path):
    core = make_core(tmp_path, agent_id="openclaw")
    capabilities = core.capabilities()
    health = core.health()

    assert capabilities["native_host_required"] is False
    assert capabilities["workspace_id"] == "shared-workspace"
    assert "prefetch_context" in capabilities["operations"]
    assert health["success"] is True
    assert health["sqlite"]["journal_mode"].lower() == "wal"
    assert health["sqlite"]["busy_timeout_ms"] >= 1000
    core.close()


def test_explicit_adapter_config_fails_closed_but_default_host_config_remains_optional(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_config(tmp_path, missing)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_config(tmp_path, malformed)

    defaults = load_config(tmp_path)
    assert defaults["db_path"] == str(tmp_path / "agent-recall.db")


def test_partial_text_update_reembeds_the_complete_merged_memory(tmp_path):
    core = make_core(tmp_path, agent_id="hermes")
    memory = core.remember(
        {
            "title": "Old title",
            "summary": "Stable summary",
            "content": "Stable body",
        }
    )
    core.embedder.calls.clear()

    assert core.update(memory["id"], {"title": "New title"})["updated"] is True

    assert core.embedder.calls == ["New title\nStable summary\nStable body"]
    core.close()


def test_background_turn_capture_keeps_the_session_that_scheduled_it(tmp_path):
    class BlockingEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def embed(self, text: str):
            self.started.set()
            assert self.release.wait(timeout=2)
            return super().embed(text)

    core = make_core(tmp_path, agent_id="hermes", session_id="old-session")
    core.config["auto_capture_turns"] = True
    embedder = BlockingEmbedder()
    core.embedder = embedder

    assert core.capture_turn("remember this", "captured", background=True) is True
    assert embedder.started.wait(timeout=1)
    core.rotate_session("new-session")
    embedder.release.set()
    core.close()

    check = make_core(tmp_path, agent_id="hermes", session_id="old-session")
    rows = check.search({"query": "remember this", "include_shared": False})["results"]
    assert len(rows) == 1
    assert rows[0]["session_id"] == "old-session"
    check.close()


def test_close_waits_for_inflight_background_capture_before_closing_sqlite(tmp_path):
    class DelayedEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.release = threading.Event()

        def embed(self, text: str):
            assert self.release.wait(timeout=3)
            return super().embed(text)

    core = make_core(tmp_path, agent_id="hermes", session_id="session")
    core.config["auto_capture_turns"] = True
    embedder = DelayedEmbedder()
    core.embedder = embedder
    assert core.capture_turn("durable shutdown", "must finish", background=True) is True
    worker = core._sync_threads[0]
    timer = threading.Timer(2.1, embedder.release.set)
    timer.start()

    core.close()
    worker.join(timeout=1)
    timer.cancel()

    check = make_core(tmp_path, agent_id="hermes", session_id="session")
    rows = check.search({"query": "durable shutdown", "include_shared": False})["results"]
    assert len(rows) == 1
    check.close()


def test_concurrent_partial_updates_retry_until_embedding_matches_merged_text(tmp_path):
    class BlockingEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def embed(self, text: str):
            self.started.set()
            assert self.release.wait(timeout=2)
            return super().embed(text)

    first = make_core(tmp_path, agent_id="hermes")
    memory = first.remember(
        {
            "title": "Old title",
            "summary": "Stable summary",
            "content": "Old body",
        }
    )
    second = make_core(tmp_path, agent_id="hermes")
    blocking = BlockingEmbedder()
    first.embedder = blocking
    outcome = {}

    worker = threading.Thread(target=lambda: outcome.update(first.update(memory["id"], {"title": "New title"})))
    worker.start()
    assert blocking.started.wait(timeout=1)
    assert second.update(memory["id"], {"content": "New body"})["updated"] is True
    blocking.release.set()
    worker.join(timeout=2)

    row = first.get_memory(memory["id"])["memory"]
    stored = first.store.conn.execute("SELECT embedding_json FROM memories WHERE id = ?", (memory["id"],)).fetchone()
    expected = FakeEmbedder().embed("New title\nStable summary\nNew body")
    assert outcome["updated"] is True
    assert row["title"] == "New title"
    assert row["content"] == "New body"
    assert json.loads(stored["embedding_json"]) == expected
    first.close()
    second.close()
