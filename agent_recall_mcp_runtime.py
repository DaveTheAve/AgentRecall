"""Bounded daemon workers: cancellation never frees a running work slot."""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from concurrent.futures import Future
from contextlib import suppress

try:
    from .agent_recall_mcp_contracts import MCPPublicError
except ImportError:
    from agent_recall_mcp_contracts import MCPPublicError


class BoundedRuntime:
    def __init__(self, *, workers=2, queue_capacity=2, operation_timeout=30.0, shutdown_timeout=1.0):
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("Invalid MCP worker limit")
        if type(queue_capacity) is not int or not 0 <= queue_capacity <= 128:
            raise ValueError("Invalid MCP admission limit")
        for value in (operation_timeout, shutdown_timeout):
            if type(value) not in (int, float) or not 0 < value <= 300:
                raise ValueError("Invalid MCP deadline")
        self.timeout = operation_timeout
        self.shutdown_timeout = shutdown_timeout
        self.capacity = workers + queue_capacity
        self._queue = queue.Queue(maxsize=self.capacity)
        self._lock = threading.Lock()
        self._active = 0
        self._closed = False
        self._cleanup_started = False
        self._threads = [threading.Thread(target=self._worker, daemon=True, name="agent-recall-mcp") for _ in range(workers)]
        for thread in self._threads:
            thread.start()

    def status(self):
        with self._lock:
            return {"active": self._active, "capacity": self.capacity, "closed": self._closed}

    def _worker(self):
        while True:
            try:
                future, fn = self._queue.get(timeout=0.05)
            except queue.Empty:
                with self._lock:
                    if self._closed:
                        return
                continue
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(fn())
                    except MCPPublicError as exc:
                        future.set_exception(MCPPublicError(exc.code))
                    except BaseException:
                        # Never retain backend exceptions (or their inputs) in futures.
                        future.set_exception(MCPPublicError("backend_error"))
            finally:
                with self._lock:
                    self._active -= 1
                self._queue.task_done()

    async def run(self, fn):
        with self._lock:
            if self._closed:
                raise MCPPublicError("closed")
            if self._active >= self.capacity:
                raise MCPPublicError("busy")
            future = Future()
            self._active += 1
            self._queue.put_nowait((future, fn))
        wrapped = asyncio.wrap_future(future)
        # A timed out/disconnected client must not leave an unobserved exception.
        wrapped.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), self.timeout)
        except asyncio.TimeoutError:
            future.cancel()  # Succeeds ONLY if work has not started.
            raise MCPPublicError("timeout") from None
        except asyncio.CancelledError:
            future.cancel()
            raise

    def close(self, cleanup):
        """Stop admission; cancel queued work; defer core.close until workers exit.

        Daemon threads deliberately avoid ThreadPoolExecutor's unbounded atexit
        join. Python cannot forcibly abort an in-flight SQLite/HTTP write.
        """
        with self._lock:
            if self._cleanup_started:
                return
            self._closed = True
            self._cleanup_started = True
        # A worker can race with draining; the future's running state decides
        # whether it can be cancelled. Its slot is still released by that worker.
        with self._queue.mutex:
            for future, _ in self._queue.queue:
                future.cancel()
        def finish():
            for thread in self._threads:
                thread.join()
            with suppress(Exception):
                cleanup()
        closer = threading.Thread(target=finish, daemon=True, name="agent-recall-mcp-close")
        closer.start()
        deadline = time.monotonic() + self.shutdown_timeout
        closer.join(max(0, deadline - time.monotonic()))


class UnknownMethod(Exception):
    """A validated request envelope, but not an SDK-supported method."""

    def __init__(self, request_id):
        self.response = {"jsonrpc": "2.0", "id": request_id,
                         "error": {"code": -32601, "message": "Method not found"}}
        super().__init__("Method not found")


def _validate_model(model, value):
    from pydantic import TypeAdapter
    return TypeAdapter(model).validate_python(value)


def _message_root(message):
    return getattr(message, "root", message)


def _jsonrpc_message(value):
    from mcp import types
    return _validate_model(types.JSONRPCMessage, value)


def parse_message(body):
    """Validate before the SDK can echo Pydantic inputs. No raw exception escapes."""
    import json
    from typing import get_args

    from mcp import types
    try:
        from .agent_recall_mcp_contracts import MAX_BODY_BYTES, bounded_json
    except ImportError:
        from agent_recall_mcp_contracts import MAX_BODY_BYTES, bounded_json
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError()
            result[key] = value
        return result
    try:
        if len(body) > MAX_BODY_BYTES:
            raise ValueError()
        raw = json.loads(body, object_pairs_hook=unique_object)
        bounded_json(raw, byte_limit=MAX_BODY_BYTES, string_limit=MAX_BODY_BYTES, depth_limit=12)
        if type(raw) is not dict or raw.keys() - {"jsonrpc", "id", "method", "params", "result", "error"}:
            raise ValueError()
        message = _jsonrpc_message(raw)
        envelope = _message_root(message)
        if isinstance(envelope, types.JSONRPCRequest):
            if raw.keys() - {"jsonrpc", "id", "method", "params"}:
                raise ValueError()
            known_methods = {model.model_fields["method"].default
                             for model in get_args(types.ClientRequest.model_fields["root"].annotation
                                                   if hasattr(types.ClientRequest, "model_fields") else types.ClientRequest)}
            if envelope.method not in known_methods:
                raise UnknownMethod(envelope.id)
            _validate_model(types.ClientRequest, raw)
            if raw["method"] == "tools/call":
                params = raw.get("params", {})
                if params.keys() - {"name", "arguments", "_meta"}:
                    raise ValueError()
        elif isinstance(envelope, types.JSONRPCNotification):
            _validate_model(types.ClientNotification, raw)
        return message
    except UnknownMethod:
        raise
    except Exception:
        raise MCPPublicError("invalid_arguments") from None


async def run_stdio(server):
    """Bounded byte frames, including malformed/oversized input recovery."""
    import json
    import sys

    import anyio
    from mcp import types
    from mcp.shared.message import SessionMessage
    try:
        from .agent_recall_mcp_contracts import MAX_BODY_BYTES
    except ImportError:
        from agent_recall_mcp_contracts import MAX_BODY_BYTES
    incoming, reader = anyio.create_memory_object_stream(0)
    writer, outgoing = anyio.create_memory_object_stream(0)

    async def read_frames():
        async with incoming:
            while True:
                line = await anyio.to_thread.run_sync(sys.stdin.buffer.readline, MAX_BODY_BYTES + 1, abandon_on_cancel=True)
                if not line:
                    return
                oversized = len(line) > MAX_BODY_BYTES
                if oversized:
                    while not line.endswith(b"\n"):
                        line = await anyio.to_thread.run_sync(sys.stdin.buffer.readline, MAX_BODY_BYTES + 1, abandon_on_cancel=True)
                        if not line:
                            break
                try:
                    if oversized:
                        raise ValueError()
                    message = parse_message(line)
                except UnknownMethod as exc:
                    await writer.send(SessionMessage(_jsonrpc_message(exc.response)))
                    continue
                except Exception:
                    code = "request_too_large" if oversized else "invalid_request"
                    # JSON-RPC requires null for an untrusted/unknown request ID;
                    # the SDK's RequestId alias excludes null, so construct only
                    # this local, constant error without coercing a caller's ID.
                    error = types.JSONRPCError.model_construct(jsonrpc="2.0", id=None,
                        error=types.ErrorData(code=-32600, message=code))
                    await writer.send(SessionMessage(_jsonrpc_message(error)))
                    continue
                await incoming.send(SessionMessage(message))

    async def write_frames():
        async with outgoing:
            async for message in outgoing:
                payload = message.message.model_dump(by_alias=True, exclude_none=True, mode="json")
                envelope = _message_root(message.message)
                if isinstance(envelope, types.JSONRPCError) and envelope.id is None:
                    payload["id"] = None
                data = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
                await anyio.to_thread.run_sync(sys.stdout.buffer.write, data, abandon_on_cancel=True)
                await anyio.to_thread.run_sync(sys.stdout.buffer.flush, abandon_on_cancel=True)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(read_frames)
        tasks.start_soon(write_frames)
        try:
            lowlevel = server._lowlevel_server if hasattr(server, "_lowlevel_server") else server._mcp_server
            await lowlevel.run(reader, writer, lowlevel.create_initialization_options())
        finally:
            tasks.cancel_scope.cancel()


class HTTPBoundary:
    """Authenticate, enforce SDK Origin/Host policy, bound/validate request bodies.

    Pure ASGI (not BaseHTTPMiddleware), preserving SSE and cancellation. SDK HTTP
    error responses are replaced, including unsupported protocol header echoes.
    """
    def __init__(self, app, *, bearer_token, security_settings, body_timeout=10.0):
        from mcp.server.transport_security import TransportSecurityMiddleware
        self.app = app
        self.expected = ("Bearer " + bearer_token).encode("utf-8") if bearer_token else b""
        self.security = TransportSecurityMiddleware(security_settings)
        self.body_timeout = body_timeout

    async def __call__(self, scope, receive, send):
        import hmac
        import json

        from starlette.requests import Request
        from starlette.responses import JSONResponse
        try:
            from .agent_recall_mcp_contracts import MAX_BODY_BYTES
        except ImportError:
            from agent_recall_mcp_contracts import MAX_BODY_BYTES
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        async def reject(code, status, headers=None):
            await JSONResponse({"error": code}, status_code=status, headers=headers)(scope, receive, send)
        headers = scope.get("headers", [])
        auth = [value for key, value in headers if key.lower() == b"authorization"]
        supplied = auth[0] if len(auth) == 1 else b""
        if self.expected and not hmac.compare_digest(supplied, self.expected):
            await reject("unauthorized", 401, {"WWW-Authenticate": "Bearer"})
            return
        # Reject duplicate/syntactically ambiguous authority headers before the
        # SDK's wildcard-port matcher (which uses a prefix comparison).
        from urllib.parse import urlsplit
        for key, code, status in ((b"host", "invalid_host", 421), (b"origin", "invalid_origin", 403)):
            values = [value for name, value in headers if name.lower() == key]
            try:
                if len(values) > 1 or (key == b"host" and not values):
                    raise ValueError()
                if values:
                    value = values[0].decode("ascii")
                    if any(c.isspace() for c in value) or "\\" in value:
                        raise ValueError()
                    parsed = urlsplit("http://" + value if key == b"host" else value)
                    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or
                            parsed.username is not None or parsed.password is not None or
                            parsed.path or parsed.query or parsed.fragment):
                        raise ValueError()
                    if parsed.port is not None and not 1 <= parsed.port <= 65535:
                        raise ValueError()
            except (ValueError, UnicodeError):
                await reject(code, status)
                return
        request = Request(scope, receive)
        response = await self.security.validate_request(request, is_post=False)
        if response:
            await reject("invalid_host" if response.status_code == 421 else "invalid_origin", response.status_code)
            return
        replay_receive = receive
        if scope["method"] == "POST":
            lengths = request.headers.getlist("content-length")
            try:
                if len(lengths) > 1 or (lengths and (not lengths[0].isascii() or not lengths[0].isdigit())):
                    raise ValueError()
                if lengths and int(lengths[0]) > MAX_BODY_BYTES:
                    await reject("request_too_large", 413)
                    return
            except ValueError:
                await reject("invalid_request", 400)
                return
            body = bytearray()
            async def read_body():
                while True:
                    chunk = await receive()
                    if chunk["type"] == "http.disconnect":
                        return False
                    part = chunk.get("body", b"")
                    if len(body) + len(part) > MAX_BODY_BYTES:
                        await reject("request_too_large", 413)
                        return False
                    body.extend(part)
                    if not chunk.get("more_body", False):
                        return True
            try:
                if not await asyncio.wait_for(read_body(), self.body_timeout):
                    return
                parse_message(body)
            except UnknownMethod as exc:
                await JSONResponse(exc.response, status_code=200)(scope, receive, send)
                return
            except asyncio.TimeoutError:
                await reject("request_timeout", 408)
                return
            except Exception:
                await reject("invalid_request", 400)
                return
            consumed = False
            async def replay_receive():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

        replaced = started = False
        async def public_send(message):
            nonlocal replaced, started
            if message["type"] == "http.response.start":
                started = True
                if message["status"] >= 400:
                    replaced = True
                    code = {400: "invalid_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
                            413: "request_too_large", 421: "invalid_host"}.get(message["status"], "transport_error")
                    data = json.dumps({"error": code}).encode()
                    await send({"type": "http.response.start", "status": message["status"],
                                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]})
                    await send({"type": "http.response.body", "body": data})
                    return
            if not replaced:
                await send(message)
        try:
            await self.app(scope, replay_receive, public_send)
        except Exception:
            if not started:
                await reject("transport_error", 500)
