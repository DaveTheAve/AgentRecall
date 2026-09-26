"""Exercise the installed wheel, not imports from a development checkout."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_installed_wheel_mcp_contract_and_learning_removed(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is required for isolated wheel installation")
    source = tmp_path / "source"
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info",
        "build", "dist", "node_modules", "Archive.tar.gz", ".env", ".venv",
    ))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HERMES_HOME": str(tmp_path / "home")}
    built = subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
        cwd=source, env=env, capture_output=True, text=True, timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next((source / "dist").glob("*.whl"))
    installed = tmp_path / "installed"
    result = subprocess.run(
        [uv, "pip", "install", "--python", sys.executable, "--no-deps", "--target", str(installed), str(wheel)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "db_path": str(tmp_path / "isolated.db"), "embedding_base_url": "",
        "llm_curator_enabled": False, "workspace_id": "wheel-test", "agent_id": "client",
    }))
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
import agent_recall_mcp
import agent_recall_core
assert Path(agent_recall_mcp.__file__).parent == Path(sys.path[0])
assert Path(agent_recall_core.__file__).parent == Path(sys.path[0])
agent_recall_mcp.main()
"""

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-I", "-c", code, str(installed), "--config", str(config), "--transport", "stdio"],
            env={"PYTHONDONTWRITEBYTECODE": "1", "HERMES_HOME": str(tmp_path / "home")},
            cwd=str(tmp_path),
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            names = {tool.name for tool in tools}
            assert {"search", "get_memory", "health", "capabilities"} <= names
            assert not {"remember", "update_memory", "forget", "curate"} & names
            for tool in tools:
                assert tool.model_dump(by_alias=True)["outputSchema"], tool.name
                assert tool.model_dump(by_alias=True)["inputSchema"].get("additionalProperties") is False, tool.name
            health = await session.call_tool("health", {})
            assert not health.model_dump(by_alias=True)["isError"]
            rejected = await session.call_tool("search", {"query": "sample", "agent_id": "spoof"})
            assert rejected.model_dump(by_alias=True)["isError"]
            assert "spoof" not in rejected.model_dump_json()

    asyncio.run(run())
