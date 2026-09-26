from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from test_mcp_adapter import make_adapter

from agent_recall_mcp import MCPAccessError, MCPAdapter, build_fastmcp
from agent_recall_mcp_contracts import OUTPUT_SCHEMAS, MCPPublicError, validate
from agent_recall_mcp_runtime import HTTPBoundary

SECRET = "NONSECRET_PRIVATE_SENTINEL /home/service/private/database.sqlite source_session_id"


@pytest.mark.parametrize("operation", ["profile_synthesize", "review"])
def test_curator_canonical_collision_requires_destructive_hint(tmp_path, operation):
    adapter = make_adapter(tmp_path)
    class Curator:
        def curate(self, text, *, default_visibility="agent"):
            return [{"content": "Replacement curated fact", "canonical_key": "collision",
                     "visibility": default_visibility}]
    adapter.core.config.update(llm_curator_enabled=True, peer_profiles_enabled=True,
                               dialectic_review_enabled=True)
    adapter.core.curator_factory = lambda: Curator()
    server = build_fastmcp(adapter)
    try:
        original = adapter.call("remember", {"content": "Original fact", "canonical_key": "collision"})
        result = adapter.call(operation, {"dry_run": False})
        assert result["results"][0]["id"] == original["id"]
        row = adapter.core.store.conn.execute(
            "SELECT content FROM memories WHERE id = ?", (original["id"],)).fetchone()
        assert row[0] == "Replacement curated fact"
        tools = asyncio.run(server.list_tools())
        assert next(tool for tool in tools if tool.name == operation).annotations.model_dump(by_alias=True)["destructiveHint"] is True
    finally:
        adapter.close()


@pytest.mark.parametrize("failure", ["exception", "rpc_exception", "rpc_result", "tool_result"])
@pytest.mark.parametrize("request_name", ["GetPromptRequest", "ReadResourceRequest", "SubscribeRequest", "CompleteRequest"])
def test_all_registered_handler_errors_are_sanitized(tmp_path, monkeypatch, failure, request_name):
    from mcp import types
    try:
        from mcp.server.lowlevel.server import HandlerEntry
        from mcp.server.mcpserver import MCPServer as FastMCP
        from mcp.shared.exceptions import MCPError as McpError
        sdk2 = True
    except ImportError:
        from mcp.server.fastmcp import FastMCP
        from mcp.shared.exceptions import McpError
        sdk2 = False

    setup = FastMCP.__init__ if sdk2 else FastMCP._setup_handlers
    async def failing_handler(*args):
        error = types.ErrorData(code=-32602, message=SECRET, data={"private": SECRET})
        if failure == "exception":
            raise ValueError(SECRET)
        if failure == "rpc_exception":
            raise McpError(error.code, error.message, error.data) if sdk2 else McpError(error)
        if failure == "rpc_result":
            return error
        result = types.CallToolResult(isError=True,
            content=[types.TextContent(type="text", text=SECRET)],
            structuredContent={"private": SECRET})
        return result if sdk2 else types.ServerResult(result)
    method = getattr(types, request_name).model_fields["method"].default
    def setup_with_failure(self, *args, **kwargs):
        setup(self, *args, **kwargs)
        if sdk2:
            self._lowlevel_server._request_handlers[method] = HandlerEntry(types.RequestParams, failing_handler)
        else:
            self._mcp_server.request_handlers[getattr(types, request_name)] = failing_handler
    monkeypatch.setattr(FastMCP, "__init__" if sdk2 else "_setup_handlers", setup_with_failure)
    adapter = make_adapter(tmp_path)
    server = build_fastmcp(adapter)
    async def run():
        try:
            if sdk2:
                result = await server._lowlevel_server._request_handlers[method].handler(None, None)
            else:
                result = await server._mcp_server.request_handlers[getattr(types, request_name)](None)
        except McpError as exc:
            assert failure != "tool_result"
            assert SECRET not in exc.error.model_dump_json()
            assert exc.error.data is None
            if failure.startswith("rpc"):
                assert exc.error.code == -32602
        else:
            assert failure == "tool_result"
            assert (result if sdk2 else result.root).model_dump(by_alias=True)["isError"]
            assert SECRET not in result.model_dump_json()
    try:
        asyncio.run(run())
    finally:
        adapter.close()


def test_default_permissions_discovery_and_allowlist(tmp_path):
    writer = make_adapter(tmp_path)
    reader = MCPAdapter(writer.core)
    assert reader.access == "read-only"
    for name in reader.WRITE_TOOLS:
        with pytest.raises(MCPAccessError):
            reader.call(name, {})
    restricted = MCPAdapter(writer.core, access="read-write", tools=["health", "capabilities"])
    assert restricted.call("capabilities")["tools"] == ["capabilities", "health"]
    assert restricted.call("capabilities")["operations"] == ["capabilities", "health"]
    with pytest.raises(MCPAccessError):
        restricted.call("search", {"query": "x"})
    writer.close()


@pytest.mark.parametrize("operation,args", [
    ("search", {"query": "x", "workspace_id": SECRET}),
    ("search", {"query": "x", "extra": SECRET}),
    ("search", {"query": "x", "limit": True}),
    ("search", {"query": "x", "limit": "2"}),
    ("search", {"query": "x", "min_importance": float("nan")}),
    ("search", {"query": "x", "updated_after": float("inf")}),
    ("search", {"query": "x", "visibility": SECRET}),
    ("remember", {"content": "x", "confidence": True}),
    ("remember", {"content": "x", "metadata": {"source_session_id": SECRET}}),
    ("remember", {"content": "x", "metadata": {"nested": {"agent_id": SECRET}}}),
    ("remember", {"content": "x" * 12001}),
    ("remember", {"content": "x", "tags": ["x"] * 65}),
    ("remember", {"content": "x", "metadata": {"a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": 1}}}}}}}}}}),
    ("remember", {"content": "x", "metadata": {str(i): "x" * 4096 for i in range(64)}}),
    ("profile_synthesize", {"scope": SECRET}),
    ("get_memory", {"id": 0}),
])
def test_strict_validation_precedes_backend(tmp_path, monkeypatch, operation, args):
    adapter = make_adapter(tmp_path)
    reached = []
    monkeypatch.setattr(adapter.core, operation, lambda *a, **k: reached.append(True))
    with pytest.raises(ValueError) as caught:
        adapter.call(operation, args)
    assert str(caught.value) == "invalid_arguments: Invalid tool arguments."
    assert reached == []
    adapter.close()


def test_backend_errors_and_public_output_projection(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    def fail(*a, **k):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(adapter.core, "search", fail)
    with pytest.raises(Exception) as caught:
        adapter.call("search", {"query": "private"})
    assert str(caught.value) == "backend_error: Operation failed."
    memory = {"id": 1, "content": SECRET, "session_id": SECRET,
              "unknown": SECRET, "metadata": {"source_session_id": SECRET,
              "internal_path": SECRET, "nested": {"source_session_id": SECRET},
              "source_ids": [1, 2], "scope": "general"}}
    monkeypatch.setattr(adapter.core, "search", lambda args: {
        "success": True, "results": [memory], "count": 1,
        "embedding_warning": SECRET, "extra": SECRET})
    result = adapter.call("search", {"query": "x"})
    assert result == {"success": True, "results": [{"id": 1, "content": SECRET,
                     "metadata": {"source_ids": [1, 2], "scope": "general"}}], "count": 1}
    adapter.close()


def test_sdk_sanitizes_validation_and_annotations(tmp_path):
    adapter = make_adapter(tmp_path)
    server = build_fastmcp(adapter)
    async def run():
        tools = await server.list_tools()
        for tool in tools:
            assert tool.model_dump(by_alias=True)["inputSchema"]["additionalProperties"] is False
            assert tool.model_dump(by_alias=True)["outputSchema"]["additionalProperties"] is False
            assert tool.annotations.model_dump(by_alias=True)["openWorldHint"] is (tool.name in {"search", "prefetch_context", "profile", "remember", "update", "curate", "conclude", "profile_synthesize", "review"})
        for args in ({"query": SECRET, "limit": "bad"}, {"query": SECRET, "agent_id": SECRET}):
            result = await server.call_tool("search", args)
            assert result.model_dump(by_alias=True)["isError"]
            assert SECRET not in result.model_dump_json()
            assert "invalid_arguments" in result.model_dump_json()
        result = await server.call_tool(SECRET, {})
        assert result.model_dump(by_alias=True)["isError"] and SECRET not in result.model_dump_json()
    asyncio.run(run())
    adapter.close()


def test_deadlines_capacity_cancellation_and_responsive_health(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    def slow(args):
        entered.set()
        release.wait(3)
        completed.set()
        return {"success": True, "id": 1, "action": "added", "visibility": "agent"}
    monkeypatch.setattr(adapter.core, "remember", slow)
    server = build_fastmcp(adapter, workers=1, queue_capacity=0, operation_timeout=0.05, shutdown_timeout=0.05)
    async def run():
        task = asyncio.create_task(server.call_tool("remember", {"content": "x"}))
        await asyncio.sleep(0.01)
        assert entered.is_set()
        health = await asyncio.wait_for(server.call_tool("health", {}), 0.1)
        assert not health.model_dump(by_alias=True)["isError"]
        result = await task
        assert result.model_dump(by_alias=True)["isError"] and "may complete" in result.model_dump_json()
        assert not completed.is_set()
        busy = await server.call_tool("remember", {"content": "x"})
        assert busy.model_dump(by_alias=True)["isError"] and "busy" in busy.model_dump_json()
        release.set()
        while not completed.is_set():
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.01)
        release.clear()
        completed.clear()
        task = asyncio.create_task(server.call_tool("remember", {"content": "x"}))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        busy = await server.call_tool("remember", {"content": "x"})
        assert "busy" in busy.model_dump_json()
        start = time.monotonic()
        adapter.close()
        assert time.monotonic() - start < 0.2
        release.set()
    try:
        asyncio.run(run())
    finally:
        release.set()
        adapter.close()


def test_all_public_shapes_with_deterministic_curator(tmp_path):
    adapter = make_adapter(tmp_path)
    class Curator:
        def curate(self, text, *, default_visibility="agent"):
            return [{"content": "Deterministic public candidate", "visibility": default_visibility,
                     "metadata": {"source_session_id": SECRET, "private_path": SECRET}}]
    adapter.core.config.update({key: True for key in (
        "llm_curator_enabled", "curated_memories_enabled", "conclusions_enabled",
        "peer_profiles_enabled", "workspace_profiles_enabled", "agent_profiles_enabled",
        "dialectic_review_enabled", "conflict_detection_enabled")})
    adapter.core.curator_factory = lambda: Curator()
    seen = set()
    def call(name, args=None):
        result = adapter.call(name, args)
        validate(result, OUTPUT_SCHEMAS[name])
        assert result["success"] is True
        assert SECRET not in json.dumps(result)
        seen.add(name)
        return result
    try:
        memory = call("remember", {"content": "Durable public fact"})
        call("update", {"id": memory["id"], "summary": "Public summary"})
        assert call("get_memory", {"id": memory["id"]})["memory"]["summary"] == "Public summary"
        call("search", {"query": "public", "explain": True})
        call("prefetch_context", {"query": "public", "include_results": True})
        call("profile", {})
        call("stats")
        call("health")
        call("capabilities")
        call("conclude", {"content": "Public conclusion", "source_ids": [memory["id"]]})
        for dry_run in (True, False):
            for name, args in (("curate", {"text": "Public input"}),
                               ("review", {"focus": "public"})):
                result = call(name, {**args, "dry_run": dry_run})
                assert result["stored"] == (0 if dry_run else 1)
            for scope in ("peer", "workspace", "agent"):
                result = call("profile_synthesize", {"scope": scope, "dry_run": dry_run})
                assert result["stored"] == (0 if dry_run else 1)
        call("forget", {"id": memory["id"]})
        assert seen == set(adapter.TOOLS)
    finally:
        adapter.close()


def test_queued_cancellation_shutdown_and_execution_permissions(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def slow(args):
        calls.append(args["content"])
        entered.set()
        release.wait(3)
        return {"success": True, "id": 1}
    monkeypatch.setattr(adapter.core, "remember", slow)
    server = build_fastmcp(adapter, workers=1, queue_capacity=2, operation_timeout=2, shutdown_timeout=0.02)
    async def run():
        first = asyncio.create_task(server.call_tool("remember", {"content": "first"}))
        assert await asyncio.to_thread(entered.wait, 1)
        queued = asyncio.create_task(server.call_tool("remember", {"content": "cancelled"}))
        await asyncio.sleep(0.01)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        revoked = asyncio.create_task(server.call_tool("remember", {"content": "revoked"}))
        await asyncio.sleep(0.01)
        adapter._tools = frozenset({"health", "capabilities"})
        assert not (await asyncio.wait_for(server.call_tool("capabilities", {}), .1)).model_dump(by_alias=True)["isError"]
        release.set()
        assert not (await first).model_dump(by_alias=True)["isError"]
        result = await revoked
        assert result.model_dump(by_alias=True)["structuredContent"]["error"]["code"] == "forbidden"
        assert calls == ["first"]
        adapter.close()
        with pytest.raises(MCPPublicError, match="closed"):
            await adapter.runtime.run(lambda: calls.append("after-close"))
        assert (await server.call_tool("health", {})).model_dump(by_alias=True)["structuredContent"]["error"]["code"] == "closed"
    try:
        asyncio.run(run())
    finally:
        release.set()
        adapter.close()


@pytest.mark.parametrize("header,value", [
    ("host", "localhost:80@evil.example"),
    ("origin", "http://localhost:80@evil.example"),
    ("origin", "http://localhost:80/evil"),
])
def test_http_guard_rejects_prefix_confusion(header, value):
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    from mcp.server.transport_security import TransportSecuritySettings
    async def app(scope, receive, send):
        from starlette.responses import JSONResponse
        await JSONResponse({"ok": True})(scope, receive, send)
    guarded = HTTPBoundary(app, bearer_token="test", security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                    allowed_hosts=["localhost:*"], allowed_origins=["http://localhost:*"]))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=guarded), base_url="http://localhost:8765") as client:
            response = await client.get("/mcp", headers={"Authorization": "Bearer test", header: value})
            assert response.status_code in (403, 421)
            assert value not in response.text
    asyncio.run(run())


@pytest.mark.parametrize("stop", ["deadline", "shutdown"])
def test_queued_work_never_executes_after_deadline_or_shutdown(stop):
    from agent_recall_mcp_runtime import BoundedRuntime
    runtime = BoundedRuntime(workers=1, queue_capacity=1, operation_timeout=.05, shutdown_timeout=.02)
    entered, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def slow():
        entered.set()
        release.wait(2)
        calls.append("running")
    async def run():
        first = asyncio.create_task(runtime.run(slow))
        assert await asyncio.to_thread(entered.wait, 1)
        queued = asyncio.create_task(runtime.run(lambda: calls.append("queued")))
        await asyncio.sleep(.005)
        if stop == "shutdown":
            start = time.monotonic()
            runtime.close(cleaned.set)
            assert time.monotonic() - start < .2
            assert not cleaned.is_set()
            with pytest.raises(asyncio.CancelledError):
                await queued
            with pytest.raises(MCPPublicError, match="closed"):
                await runtime.run(lambda: calls.append("new"))
        else:
            with pytest.raises(MCPPublicError, match="timeout"):
                await queued
            assert runtime.status()["active"] == 2
        with pytest.raises(MCPPublicError, match="timeout"):
            await first
        release.set()
        if stop == "deadline":
            # Let the worker drain normally: close() itself cancels queued work
            # and would otherwise conceal a broken timeout cancellation path.
            async def drained():
                while runtime.status()["active"]:
                    await asyncio.sleep(.005)
            await asyncio.wait_for(drained(), 1)
            assert calls == ["running"]
            runtime.close(cleaned.set)
        assert await asyncio.to_thread(cleaned.wait, 1)
        assert calls == ["running"]
        assert runtime.status()["active"] == 0
    try:
        asyncio.run(run())
    finally:
        release.set()
        runtime.close(cleaned.set)


@pytest.mark.parametrize("request_id", [0, "correlated-id"])
def test_unknown_method_preserves_validated_id(request_id):
    from agent_recall_mcp_runtime import UnknownMethod, parse_message

    with pytest.raises(UnknownMethod) as caught:
        parse_message(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": SECRET}).encode())
    assert caught.value.response == {"jsonrpc": "2.0", "id": request_id,
                                     "error": {"code": -32601, "message": "Method not found"}}


@pytest.mark.parametrize("overrides", [
    {"id": None}, {"id": True}, {"id": []}, {"id": 1.5},
    {"method": {}}, {"jsonrpc": "1.0"}, {"params": "private"}, {"result": {}},
])
def test_unknown_method_malformed_envelope_stays_invalid(overrides):
    from agent_recall_mcp_runtime import parse_message

    raw = {"jsonrpc": "2.0", "id": 1, "method": SECRET, **overrides}
    with pytest.raises(MCPPublicError, match="invalid_arguments"):
        parse_message(json.dumps(raw).encode())


@pytest.mark.parametrize("mode", ["complete", "timeout", "cancel", "disconnect"])
def test_http_body_deadline_and_cancellation(mode):
    from mcp.server.transport_security import TransportSecuritySettings

    reached, sent = [], []
    async def app(scope, receive, send):
        reached.append(await receive())
    async def send(message):
        sent.append(message)
    async def run():
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def receive():
            entered.set()
            if mode == "complete":
                return {"type": "http.request", "body": b'{"jsonrpc":"2.0","id":1,"method":"ping"}'}
            if mode == "disconnect":
                return {"type": "http.disconnect"}
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        boundary = HTTPBoundary(app, bearer_token="", body_timeout=.03,
                                security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                    allowed_hosts=["localhost:*"], allowed_origins=["http://localhost:*"]))
        scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": [(b"host", b"localhost:8765")]}
        task = asyncio.create_task(boundary(scope, receive, send))
        if mode == "cancel":
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await asyncio.wait_for(task, 1)
        if mode == "complete":
            assert len(reached) == 1
        else:
            assert not reached
        if mode == "timeout":
            assert sent[0]["status"] == 408
            assert json.loads(sent[1]["body"]) == {"error": "request_timeout"}
        else:
            assert not sent
        if mode in {"cancel", "timeout"}:
            assert cancelled.is_set()
    asyncio.run(run())
