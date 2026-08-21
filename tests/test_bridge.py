from __future__ import annotations

import json
import subprocess
import sys

import pytest

import agent_recall_bridge
from agent_recall_bridge import AgentRecallBridge, BridgeProtocolError


def write_config(tmp_path):
    path = tmp_path / "agent-recall.json"
    path.write_text(
        json.dumps(
            {
                "db_path": str(tmp_path / "shared.db"),
                "embedding_base_url": "",
                "embedding_model": "fake",
                "shared_recall": True,
                "auto_capture_turns": True,
                "auto_capture_compression_checkpoints": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def identity(agent_id: str, session_id: str = "session-1") -> dict[str, str]:
    return {"workspace_id": "shared-workspace", "agent_id": agent_id, "session_id": session_id}


def test_bridge_preserves_cross_host_acl_and_releases_session_context(tmp_path):
    bridge = AgentRecallBridge(config_path=write_config(tmp_path), max_contexts=4)

    private = bridge.dispatch(
        {
            "operation": "remember",
            "identity": identity("openclaw:main"),
            "args": {"content": "private OpenClaw fact", "visibility": "agent"},
        }
    )
    shared = bridge.dispatch(
        {
            "operation": "remember",
            "identity": identity("openclaw:main"),
            "args": {"content": "shared cross-host fact", "visibility": "shared"},
        }
    )

    other = bridge.dispatch(
        {
            "operation": "search",
            "identity": identity("openclaw:reviewer"),
            "args": {"query": "fact", "include_shared": True},
        }
    )
    assert [row["id"] for row in other["results"]] == [shared["id"]]
    assert private["id"] != shared["id"]
    assert bridge.context_count == 2

    result = bridge.dispatch({"operation": "release", "identity": identity("openclaw:reviewer"), "args": {}})
    assert result == {"success": True, "released": True}
    assert bridge.context_count == 1
    bridge.close()


def test_bridge_maps_native_turn_and_compaction_lifecycle(tmp_path):
    bridge = AgentRecallBridge(config_path=write_config(tmp_path))
    current = identity("openclaw:main", "session-native")

    captured = bridge.dispatch(
        {
            "operation": "capture_turn",
            "identity": current,
            "args": {"user_content": "Remember native hooks", "assistant_content": "OpenClaw has real prompt hooks."},
        }
    )
    checkpoint = bridge.dispatch(
        {
            "operation": "save_checkpoint",
            "identity": current,
            "args": {
                "messages": [
                    {"role": "user", "content": "Preserve checkpoint context"},
                    {"role": "assistant", "content": "Checkpoint stored before compaction"},
                ]
            },
        }
    )

    assert captured == {"success": True, "captured": True}
    assert checkpoint["success"] is True
    assert checkpoint["saved"] is True
    assert isinstance(checkpoint["id"], int)
    found = bridge.dispatch(
        {
            "operation": "search",
            "identity": current,
            "args": {"query": "checkpoint compaction", "include_shared": False},
        }
    )
    assert {row["category"] for row in found["results"]} >= {"compression_checkpoint"}
    bridge.close()


def test_bridge_rejects_missing_or_unbounded_identity(tmp_path):
    bridge = AgentRecallBridge(config_path=write_config(tmp_path))
    with pytest.raises(BridgeProtocolError, match="workspace_id"):
        bridge.dispatch({"operation": "search", "identity": {"agent_id": "openclaw"}, "args": {"query": "x"}})
    with pytest.raises(BridgeProtocolError, match="agent_id"):
        bridge.dispatch(
            {
                "operation": "search",
                "identity": {"workspace_id": "ws", "agent_id": "x" * 300, "session_id": "s"},
                "args": {"query": "x"},
            }
        )
    bridge.close()


def test_bridge_curate_uses_the_core_operation_shape(tmp_path, monkeypatch):
    class FakeCurator:
        def curate(self, text, *, default_visibility="agent"):
            return [{"content": f"curated: {text}", "visibility": default_visibility}]

    config = write_config(tmp_path)
    monkeypatch.setattr(agent_recall_bridge, "build_curator", lambda _config: FakeCurator())
    bridge = AgentRecallBridge(config_path=config)
    result = bridge.dispatch(
        {
            "operation": "curate",
            "identity": identity("openclaw:main"),
            "args": {"text": "bridge candidate", "dry_run": True},
        }
    )
    assert result["candidates"][0]["content"] == "curated: bridge candidate"
    assert result["stored"] == 0
    bridge.close()


def test_jsonl_bridge_protocol_round_trip(tmp_path):
    config = write_config(tmp_path)
    requests = [
        {
            "id": 1,
            "operation": "remember",
            "identity": identity("openclaw:main"),
            "args": {"content": "JSONL bridge memory", "visibility": "agent"},
        },
        {
            "id": 2,
            "operation": "search",
            "identity": identity("openclaw:main"),
            "args": {"query": "JSONL bridge"},
        },
        {"id": 3, "operation": "shutdown", "identity": identity("openclaw:main"), "args": {}},
    ]
    process = subprocess.run(
        [sys.executable, "-m", "agent_recall_bridge", "--config", str(config)],
        input="".join(json.dumps(item) + "\n" for item in requests),
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert process.returncode == 0, process.stderr
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert [item["id"] for item in responses] == [1, 2, 3]
    assert all(item["ok"] for item in responses)
    assert responses[1]["result"]["results"][0]["content"] == "JSONL bridge memory"


def test_jsonl_bridge_rejects_oversized_line_without_losing_framing(tmp_path):
    config = write_config(tmp_path)
    valid = {
        "id": 20,
        "operation": "capabilities",
        "identity": identity("openclaw:main"),
        "args": {},
    }
    shutdown = {
        "id": 21,
        "operation": "shutdown",
        "identity": identity("openclaw:main"),
        "args": {},
    }
    process = subprocess.run(
        [sys.executable, "-m", "agent_recall_bridge", "--config", str(config)],
        input="x" * 1_000_100 + "\n" + json.dumps(valid) + "\n" + json.dumps(shutdown) + "\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert responses[0]["ok"] is False and "size limit" in responses[0]["error"]
    assert responses[1]["ok"] is True and responses[1]["id"] == 20
    assert responses[2]["ok"] is True and responses[2]["id"] == 21


def test_jsonl_bridge_recovers_after_malformed_input(tmp_path):
    config = write_config(tmp_path)
    valid = {
        "id": 9,
        "operation": "capabilities",
        "identity": identity("openclaw:main"),
        "args": {},
    }
    shutdown = {
        "id": 10,
        "operation": "shutdown",
        "identity": identity("openclaw:main"),
        "args": {},
    }
    process = subprocess.run(
        [sys.executable, "-m", "agent_recall_bridge", "--config", str(config)],
        input="{not-json}\n" + json.dumps(valid) + "\n" + json.dumps(shutdown) + "\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode == 0
    assert responses[0]["ok"] is False and responses[0]["id"] is None
    assert responses[1]["ok"] is True and responses[1]["id"] == 9
    assert responses[2]["ok"] is True and responses[2]["id"] == 10
