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
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    pytest.skip("optional MCP SDK is not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]


async def assert_non_tool_errors_are_public(session):
    from datetime import timedelta

    from mcp import types
    try:
        from mcp.shared.exceptions import MCPError as McpError
    except ImportError:
        from mcp.shared.exceptions import McpError
    from pydantic import AnyUrl

    # A generic request models a future/extension method the SDK union lacks.
    with pytest.raises(McpError) as unknown:
        await session.send_request(
            types.Request(method="private-unknown-method-sentinel", params={}),
            types.EmptyResult,
            request_read_timeout_seconds=(timedelta(seconds=2) if hasattr(types.ClientRequest, "model_fields") else 2.0),
        )
    assert unknown.value.error.code == -32601
    assert unknown.value.error.message == "Method not found"
    assert unknown.value.error.data is None
    await session.send_ping()

    sentinel = "dummy-private-rpc-sentinel"
    responses = []
    for call in (
        lambda: session.get_prompt(sentinel),
        lambda: session.read_resource(AnyUrl(f"file:///private/{sentinel}.sqlite")
                                      if hasattr(types.ClientRequest, "model_fields") else f"file:///private/{sentinel}.sqlite"),
    ):
        with pytest.raises(McpError) as caught:
            await call()
        responses.append(caught.value.error.model_dump_json())
    assert all(sentinel not in response for response in responses), responses
    await session.send_ping()
    assert (await session.list_prompts()).prompts == []
    assert (await session.list_resources()).resources == []


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
            args=[str(ROOT / "agent_recall_mcp.py"), "--config", str(config), "--transport", "stdio", "--access", "read-write"],
        )
        async with stdio_client(parameters) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {"remember", "search", "prefetch_context", "get_memory", "health", "capabilities"} <= names
            remember_tool = next(tool for tool in tools.tools if tool.name == "remember")
            assert {"canonical_key", "expires_at"} <= set(remember_tool.model_dump(by_alias=True)["inputSchema"]["properties"])
            health = await session.call_tool("health", {})
            assert health.model_dump(by_alias=True)["isError"] is False
            assert "quick_check" in str(health.model_dump(by_alias=True)["structuredContent"] or health.content)
            await assert_non_tool_errors_are_public(session)
            saved = await session.call_tool("remember", {"content": "transportneedle durable memory"})
            assert saved.model_dump(by_alias=True)["isError"] is False
            memory_id = saved.model_dump(by_alias=True)["structuredContent"]["id"]
            found = await session.call_tool("search", {"query": "transportneedle"})
            assert memory_id in {row["id"] for row in found.model_dump(by_alias=True)["structuredContent"]["results"]}
            removed = await session.call_tool("forget", {"id": memory_id})
            assert removed.model_dump(by_alias=True)["isError"] is False
            found = await session.call_tool("search", {"query": "transportneedle"})
            assert found.model_dump(by_alias=True)["structuredContent"]["count"] == 0

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

        auth = {"Authorization": "Bearer test-secret-token", "Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        assert httpx.post(url, headers={"Authorization": "Bearer invalid"}).status_code == 401
        assert httpx.post(url, headers={**auth, "Host": "evil.example"}, json={}).status_code == 421
        assert httpx.post(url, headers={**auth, "Origin": "https://evil.example"}, json={}).status_code == 403
        secret = "NONSECRET_PRIVATE_SENTINEL /home/service/private/data.db"
        for body in (json.dumps({"jsonrpc": "2.0", "id": 1, "method": {"secret": secret}}), '{"secret":"' + secret):
            response = httpx.post(url, headers=auth, content=body)
            assert response.status_code == 400
            assert secret not in response.text
            assert "invalid_request" in response.text
        response = httpx.post(url, headers=auth, content=b"x" * 131073)
        assert response.status_code == 413
        assert "request_too_large" in response.text
        # Chunked bodies must be counted, not just Content-Length checked.
        response = httpx.post(url, headers=auth, content=iter([b"x" * 70000, b"x" * 70000]))
        assert response.status_code == 413

        async def run() -> None:
            rpc_transports = []
            async def observe_response(response):
                if response.request.method == "POST":
                    method = json.loads(response.request.content).get("method")
                    if method in {"prompts/get", "resources/read"}:
                        rpc_transports.append((method, response.status_code, response.headers.get("content-type", "")))
            async with (
                httpx.AsyncClient(headers={"Authorization": "Bearer test-secret-token"},
                                  event_hooks={"response": [observe_response]}) as client,
                streamable_http_client(url, http_client=client) as streams,
                ClientSession(*streams[:2]) as session,
            ):
                await session.initialize()
                health = await session.call_tool("health", {})
                assert health.model_dump(by_alias=True)["isError"] is False
                await assert_non_tool_errors_are_public(session)
                assert {method for method, _, _ in rpc_transports} == {"prompts/get", "resources/read"}
                assert all(status == 200 and "text/event-stream" in content_type
                           for _, status, content_type in rpc_transports)
                from agent_recall_mcp import MCPAdapter
                names = {tool.name for tool in (await session.list_tools()).tools}
                assert not names & MCPAdapter.WRITE_TOOLS
                capabilities = (await session.call_tool("capabilities", {})).model_dump(by_alias=True)["structuredContent"]
                assert set(capabilities["tools"]) == names == set(capabilities["operations"])
                for name in MCPAdapter.WRITE_TOOLS:
                    result = await session.call_tool(name, {"content": secret, "dry_run": True})
                    assert result.model_dump(by_alias=True)["isError"] and "forbidden" in result.model_dump_json()
                    assert secret not in result.model_dump_json()
                for args in ({"query": secret, "agent_id": secret}, {"query": secret, "limit": "secret"}, {"query": "x" * 12001}):
                    result = await session.call_tool("search", args)
                    assert result.model_dump(by_alias=True)["isError"] and "invalid_arguments" in result.model_dump_json()
                    assert secret not in result.model_dump_json()

        asyncio.run(run())
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_stdio_malformed_frames_are_sanitized_and_recover(tmp_path):
    config = tmp_path / "stdio-security.json"
    config.write_text(json.dumps({"db_path": str(tmp_path / "stdio-security.db"), "embedding_base_url": ""}))
    secret = "NONSECRET_PRIVATE_SENTINEL /home/service/private/data.db"
    async def run():
        proc = await asyncio.create_subprocess_exec(sys.executable, str(ROOT / "agent_recall_mcp.py"),
            "--config", str(config), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=300000)
        async def request(data):
            proc.stdin.write(data + b"\n")
            await proc.stdin.drain()
            return json.loads(await asyncio.wait_for(proc.stdout.readline(), 5))
        try:
            malformed = json.dumps({"jsonrpc": "2.0", "id": 1, "method": {"private": secret}}).encode()
            result = await request(malformed)
            assert secret not in json.dumps(result)
            assert "invalid_request" in json.dumps(result)
            assert result["id"] is None
            result = await request(b"x" * 131073)
            assert "request_too_large" in json.dumps(result)
            result = await request(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {
                "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}}).encode())
            assert "result" in result
        finally:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), 5)
    asyncio.run(run())
