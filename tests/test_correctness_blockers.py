from __future__ import annotations

import json
import subprocess
import sys
import venv
from pathlib import Path

import pytest

import agent_recall_store
from agent_recall_store import AgentRecallStore

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_visibility_collision_releases_writer_and_preserves_acl(tmp_path):
    first = AgentRecallStore(tmp_path / "collision.db")
    second = AgentRecallStore(tmp_path / "collision.db")
    try:
        ids = [
            first.add_memory(
                workspace_id="ws", agent_id="owner", content=f"original {visibility}",
                visibility=visibility, canonical_key="same-key", embedding=[1.0], embedding_model="fake",
            )
            for visibility in ("agent", "shared")
        ]
        originals = [dict(row) for row in first.conn.execute("SELECT * FROM memories ORDER BY id")]
        second.conn.execute("PRAGMA busy_timeout = 0")
        with pytest.raises(agent_recall_store.sqlite3.IntegrityError):
            first.update_memory(ids[0], "ws", "owner", "", visibility="shared", content="must rollback")
        # A second connection must be able to write immediately, not after close/retry.
        second.add_memory(workspace_id="ws", agent_id="other", content="independent write",
                          embedding=[1.0], embedding_model="fake")
        assert not first.conn.in_transaction
        assert [dict(row) for row in second.conn.execute("SELECT * FROM memories WHERE id IN (?, ?) ORDER BY id", ids)] == originals
        assert second.get_visible(ids[0], "ws", "other", "") is None
        assert second.get_visible(ids[1], "ws", "other", "") is not None
        assert not second.update_memory(ids[1], "ws", "other", "", content="unauthorized")
    finally:
        first.close()
        second.close()


def test_fts_candidate_outside_vector_window_gets_semantic_score(tmp_path, monkeypatch):
    store = AgentRecallStore(tmp_path / "fusion.db")
    try:
        target = store.add_memory(workspace_id="ws", agent_id="owner", content="needle",
                                  embedding=[1.0, 0.0], embedding_model="fake", importance=0.01)
        for index in range(600):
            store.add_memory(workspace_id="ws", agent_id="owner", content=f"unrelated {index}",
                             embedding=[0.0, 1.0], embedding_model="fake", importance=1.0)
        calls = []
        real_cosine = agent_recall_store.cosine

        def counted_cosine(left, right):
            calls.append(right)
            return real_cosine(left, right)

        monkeypatch.setattr(agent_recall_store, "cosine", counted_cosine)
        results = store.search(workspace_id="ws", agent_id="owner", session_id="", query="needle",
                               query_embedding=[1.0, 0.0], explain=True, track_access=False, limit=50)
        result = next(row for row in results if row["id"] == target)
        assert result["score_explanation"]["vector"] == 1.0
        assert len(calls) <= 500
    finally:
        store.close()


@pytest.mark.parametrize("relative", [False, True])
def test_openclaw_installer_preserves_venv_python_prefix(tmp_path, relative):
    env = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(env)
    python = env / "bin" / "python"
    assert python.is_symlink()
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/install_openclaw_plugin.py"), "--dry-run",
         "--python-command", "./venv/bin/python" if relative else str(python)],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    commands = [json.loads(line) for line in process.stdout.splitlines()]
    setting = next(command for command in commands if command[3].endswith(".pythonCommand"))
    selected = json.loads(setting[4])
    prefix = subprocess.check_output([selected, "-c", "import sys; print(sys.prefix)"], text=True).strip()
    assert prefix == str(env)
    assert selected == str(python)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_openclaw_installer_rejects_nonexecutable_python(tmp_path, kind):
    python = tmp_path / "not-python"
    if kind == "file":
        python.write_text("not executable", encoding="utf-8")
    else:
        python.mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/install_openclaw_plugin.py"), "--openclaw", "/bin/true",
         "--python-command", str(python), "--config-path", str(config)],
        text=True, capture_output=True, check=False,
    )
    assert process.returncode != 0
    assert "Python executable not found" in process.stderr
