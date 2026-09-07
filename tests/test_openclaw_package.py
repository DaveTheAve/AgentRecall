from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_openclaw_manifest_and_package_declare_native_memory_contract():
    manifest = json.loads((ROOT / "openclaw.plugin.json").read_text(encoding="utf-8"))
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

    assert manifest["id"] == "agent-recall"
    assert manifest["kind"] == "memory"
    assert manifest["activation"]["onStartup"] is True
    assert manifest["activation"]["onCommands"] == ["memory"]
    assert {"memory_search", "memory_get", "memory_store", "memory_forget"} <= set(manifest["contracts"]["tools"])
    assert package["openclaw"]["extensions"] == ["./openclaw_plugin/index.js"]
    assert package["openclaw"]["compat"]["pluginApi"] == ">=2026.5.22"
    assert "agent_recall_bridge.py" in package["files"]
    assert "agent_recall_session_archive.py" in package["files"]
    assert "agent_recall_session_learning.py" not in package["files"]
    assert "CHANGELOG.md" in package["files"]
    assert "scripts/install_openclaw_plugin.py" in package["files"]
    assert "docs/OPENCLAW.md" in package["files"]
    assert "docs/ARCHITECTURE.md" in package["files"]
    assert "docs/MCP.md" in package["files"]
    assert "openclaw_plugin/index.js" in package["files"]
    assert "openclaw_plugin/bridge-client.js" in package["files"]
    assert "openclaw_plugin/" not in package["files"]
    assert "node_modules/agent-recall/scripts/install_openclaw_plugin.py --copy" in (
        ROOT / "README.md"
    ).read_text(encoding="utf-8")


def test_openclaw_installer_persists_relative_python_path_as_absolute(tmp_path):
    relative_python = tmp_path / "venv" / "bin" / "python3"
    relative_python.parent.mkdir(parents=True)
    relative_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    relative_python.chmod(0o755)
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "install_openclaw_plugin.py"),
            "--dry-run",
            "--python-command",
            "./venv/bin/python3",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert process.returncode == 0, process.stderr
    commands = [json.loads(line) for line in process.stdout.splitlines()]
    python_setting = next(command for command in commands if command[3].endswith(".pythonCommand"))
    assert json.loads(python_setting[4]) == str(relative_python.resolve())


def test_openclaw_install_docs_use_the_declared_python3_prerequisite():
    install_docs = "\n".join(
        (ROOT / path).read_text(encoding="utf-8") for path in ("README.md", "docs/OPENCLAW.md")
    )

    assert "\npython scripts/install_openclaw_plugin.py" not in install_docs
    assert "\npython node_modules/agent-recall/scripts/install_openclaw_plugin.py" not in install_docs
    assert "python3 scripts/install_openclaw_plugin.py" in install_docs
    assert "python3 node_modules/agent-recall/scripts/install_openclaw_plugin.py" in install_docs


def test_openclaw_installer_is_explicit_about_process_scan_and_conversation_access(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "install_openclaw_plugin.py"),
            "--dry-run",
            "--root",
            str(ROOT),
            "--config-path",
            str(tmp_path / "agent-recall.json"),
            "--workspace-id",
            "shared-workspace",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    commands = [json.loads(line) for line in process.stdout.splitlines()]
    assert "--dangerously-force-unsafe-install" in commands[0]
    flattened = [" ".join(command) for command in commands]
    assert any("hooks.allowConversationAccess true --json" in line for line in flattened)
    assert any("hooks.allowPromptInjection true --json" in line for line in flattened)
    assert any('config.workspaceId "shared-workspace" --json' in line for line in flattened)
    assert flattened[-1].endswith("plugins inspect agent-recall --runtime --json")


def test_openclaw_installer_fails_before_partial_install_when_config_is_missing(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "install_openclaw_plugin.py"),
            "--openclaw",
            "/bin/true",
            "--python-command",
            sys.executable,
            "--config-path",
            str(tmp_path / "missing.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode != 0
    assert "config file not found" in process.stderr


@pytest.mark.skipif(
    not (os.environ.get("OPENCLAW_BIN") or shutil.which("openclaw")),
    reason="OpenClaw CLI is not installed",
)
def test_real_openclaw_loader_accepts_memory_plugin(tmp_path):
    openclaw = os.environ.get("OPENCLAW_BIN") or shutil.which("openclaw")
    assert openclaw is not None
    state = tmp_path / "state"
    state.mkdir()
    env = {**os.environ, "OPENCLAW_STATE_DIR": str(state), "OPENCLAW_CONFIG_PATH": str(state / "openclaw.json")}

    install = subprocess.run(
        [
            openclaw,
            "plugins",
            "install",
            "--link",
            "--dangerously-force-unsafe-install",
            str(ROOT),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert install.returncode == 0, install.stderr
    for key in ("allowConversationAccess", "allowPromptInjection"):
        configured = subprocess.run(
            [openclaw, "config", "set", f"plugins.entries.agent-recall.hooks.{key}", "true", "--json"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert configured.returncode == 0, configured.stderr
    agent_recall_config = tmp_path / "agent-recall.json"
    agent_recall_config.write_text(
        json.dumps({"db_path": str(tmp_path / "shared.db"), "embedding_base_url": ""}),
        encoding="utf-8",
    )
    for path, value in (
        ("plugins.entries.agent-recall.config.configPath", str(agent_recall_config)),
        ("plugins.entries.agent-recall.config.workspaceId", "loader-test"),
    ):
        configured = subprocess.run(
            [openclaw, "config", "set", path, json.dumps(value), "--json"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert configured.returncode == 0, configured.stderr

    inspected = subprocess.run(
        [openclaw, "plugins", "inspect", "agent-recall", "--runtime", "--json"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert inspected.returncode == 0, inspected.stderr
    result = json.loads(inspected.stdout)
    assert result["plugin"]["status"] == "loaded"
    assert result["plugin"]["memorySlotSelected"] is True
    assert len(result["tools"]) == 10
    assert "memory" in result["cliCommands"]
    assert {"before_prompt_build", "agent_end", "before_compaction", "before_reset", "session_end"} <= {
        hook["name"] for hook in result["typedHooks"]
    }
    assert result["diagnostics"] == []

    status = subprocess.run(
        [openclaw, "memory", "status", "--agent", "main", "--json"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["provider"] == "agent-recall"
