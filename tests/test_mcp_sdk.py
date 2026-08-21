from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

try:
    import httpx
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    pytest.skip("optional MCP SDK is not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]


def test_real_mcp_stdio_discovery_and_health(tmp_path):
    config = tmp_path / "agent-recall.json"
    config.write_text(
        json.dumps(
            {
                "db_path": str(tmp_path / "mcp-smoke.db"),
                "workspace_id": "mcp-smoke",
                "agent_id": "client",
                "embedding_base_url": "",
                "llm_curator_enabled": False,
            }
        ),
        encoding="utf-8",
    )

    async def run() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "agent_recall_mcp.py"), "--config", str(config), "--transport", "stdio"],
        )
        async with stdio_client(parameters) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {"remember", "search", "prefetch_context", "get_memory", "health", "capabilities"} <= names
            health = await session.call_tool("health", {})
            assert health.isError is False
            assert "quick_check" in str(health.structuredContent or health.content)

    asyncio.run(run())


def test_streamable_http_refuses_to_start_without_authentication(tmp_path):
    config = tmp_path / "agent-recall.json"
    config.write_text(
        json.dumps({"db_path": str(tmp_path / "no-auth.db"), "embedding_base_url": ""}),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.pop("AGENT_RECALL_MCP_TOKEN", None)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "agent_recall_mcp.py"),
            "--config",
            str(config),
            "--transport",
            "streamable-http",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "HTTP MCP requires a bearer token" in result.stderr


def test_streamable_http_requires_and_accepts_bearer_token(tmp_path):
    config = tmp_path / "agent-recall.json"
    config.write_text(
        json.dumps(
            {
                "db_path": str(tmp_path / "http-smoke.db"),
                "workspace_id": "http-smoke",
                "agent_id": "openclaw",
                "embedding_base_url": "",
            }
        ),
        encoding="utf-8",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "AGENT_RECALL_TEST_TOKEN": "test-secret-token"}
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "agent_recall_mcp.py"),
            "--config",
            str(config),
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--auth-token-env",
            "AGENT_RECALL_TEST_TOKEN",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        for _ in range(100):
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"HTTP MCP server exited early: {stdout}\n{stderr}")
            try:
                response = httpx.post(url, timeout=0.2)
                if response.status_code == 401:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        else:
            pytest.fail("HTTP MCP server did not become ready with an authentication challenge")

        async def run() -> None:
            async with (
                httpx.AsyncClient(headers={"Authorization": "Bearer test-secret-token"}) as client,
                streamable_http_client(url, http_client=client) as (read, write, _),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                health = await session.call_tool("health", {})
                assert health.isError is False

        asyncio.run(run())
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
