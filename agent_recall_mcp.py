from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from .agent_recall_core import AgentIdentity, AgentRecallCore, build_curator, load_config
    from .agent_recall_mcp_contracts import (
        ERRORS,
        INPUT_SCHEMAS,
        OUTPUT_SCHEMAS,
        MCPAccessError,
        MCPPublicError,
        public_result,
        validate_arguments,
    )
    from .agent_recall_mcp_runtime import BoundedRuntime, HTTPBoundary, run_stdio
except ImportError:
    from agent_recall_core import AgentIdentity, AgentRecallCore, build_curator, load_config
    from agent_recall_mcp_contracts import (
        ERRORS,
        INPUT_SCHEMAS,
        OUTPUT_SCHEMAS,
        MCPAccessError,
        MCPPublicError,
        public_result,
        validate_arguments,
    )
    from agent_recall_mcp_runtime import BoundedRuntime, HTTPBoundary, run_stdio


class MCPAdapter:
    """Fixed-identity public boundary. No caller-controlled identity routing."""
    READ_TOOLS = {"search", "prefetch_context", "get_memory", "profile", "stats", "health", "capabilities"}
    WRITE_TOOLS = {"remember", "update", "forget", "curate", "conclude", "profile_synthesize", "review"}
    TOOLS = tuple(sorted(READ_TOOLS | WRITE_TOOLS))

    def __init__(self, core: AgentRecallCore, *, access="read-only", tools=None):
        if access not in {"read-only", "read-write"}:
            raise ValueError("Invalid MCP access policy")
        if tools is not None and (not isinstance(tools, (list, tuple, set, frozenset)) or
                                  any(type(name) is not str or name not in self.TOOLS for name in tools)):
            raise ValueError("Invalid MCP tool allowlist")
        self.core = core
        self._access = access
        self._tools = frozenset(self.TOOLS if tools is None else tools) & (
            self.READ_TOOLS if access == "read-only" else set(self.TOOLS))
        self.runtime = None
        self._closed = False
        # Startup snapshot, not an on-demand SQLite lock acquisition.
        try:
            self._health = public_result("health", {**core.health(), "snapshot": True})
        except Exception:
            self._health = {"success": False, "snapshot": True, "sqlite": {"quick_check": "unavailable"}}

    @property
    def access(self):
        return self._access

    @property
    def tools(self):
        return tuple(sorted(self._tools))

    def _authorize(self, operation):
        if type(operation) is not str or operation not in self._tools:
            raise MCPAccessError()
        if self._closed:
            raise MCPPublicError("closed")

    def prepare(self, operation, args=None):
        self._authorize(operation)
        return validate_arguments(operation, args)

    def call(self, operation: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        values = self.prepare(operation, args)
        return self._execute(operation, values)

    def _execute(self, operation, values):
        self._authorize(operation)
        try:
            track = self.access != "read-only"
            if operation == "search":
                result = self.core.search({**values, "_track_access": track})
            elif operation == "prefetch_context":
                result = self.core.prefetch_context(**values, track_access=track)
            elif operation == "get_memory":
                result = self.core.get_memory(values["id"], include_shared=values.get("include_shared", True))
            elif operation == "profile":
                result = self.core.profile(**values, track_access=track)
            elif operation == "stats":
                result = self.core.stats()
            elif operation == "health":
                result = {**self._health}
                if self.runtime:
                    result["runtime"] = self.runtime.status()
            elif operation == "capabilities":
                result = self.core.capabilities()
                result.update({"access": self.access, "tools": list(self.tools), "operations": list(self.tools)})
                feature_tools = {"curation": "curate", "conclusions": "conclude", "profile_synthesis": "profile_synthesize", "review": "review"}
                result["features"] = {key: bool(enabled and (key not in feature_tools or feature_tools[key] in self._tools))
                                      for key, enabled in result.get("features", {}).items()}
            elif operation == "update":
                result = self.core.update(values["id"], values)
            elif operation == "forget":
                result = self.core.forget(values["id"])
            else:
                result = getattr(self.core, operation)(values)
        except Exception:
            raise MCPPublicError("backend_error") from None
        return public_result(operation, result)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.runtime:
            self.runtime.close(self.core.close)
        else:
            self.core.close()


def build_fastmcp(adapter, *, workers=2, queue_capacity=2, operation_timeout=30.0, shutdown_timeout=1.0):
    """Use explicit SDK handlers, not FastMCP's coercing/echoing function models."""
    try:
        from mcp import types
        try:
            from mcp.server.mcpserver import MCPServer as FastMCP
        except ImportError:
            from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("The optional 'mcp' package is required to run the AgentRecall MCP server") from exc
    if adapter.runtime is not None:
        raise ValueError("An MCP adapter can have only one server")
    adapter.runtime = BoundedRuntime(workers=workers, queue_capacity=queue_capacity,
                                    operation_timeout=operation_timeout, shutdown_timeout=shutdown_timeout)
    external = {"search", "prefetch_context", "profile", "remember", "update", "curate", "conclude", "profile_synthesize", "review"}
    tracked = {"search", "prefetch_context", "profile"}

    class PublicFastMCP(FastMCP):
        async def run_stdio_async(self):
            await run_stdio(self)

        async def list_tools(self):
            return [types.Tool(name=name, description=f"AgentRecall {name} under the server's fixed identity.",
                inputSchema=INPUT_SCHEMAS[name], outputSchema=OUTPUT_SCHEMAS[name],
                annotations=types.ToolAnnotations(
                    readOnlyHint=name in adapter.READ_TOOLS and not (adapter.access == "read-write" and name in tracked),
                    destructiveHint=name in {"remember", "update", "forget", "profile_synthesize", "review"},
                    idempotentHint=name in {"get_memory", "health", "capabilities", "stats", "forget"} or (adapter.access == "read-only" and name in tracked),
                    openWorldHint=name in external)) for name in adapter.tools]

        async def call_tool(self, name, arguments=None, context=None):
            try:
                values = adapter.prepare(name, arguments)
                if name in {"health", "capabilities"}:
                    result = adapter._execute(name, values)
                else:
                    result = await adapter.runtime.run(lambda: adapter._execute(name, values))
                return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(result))],
                                            structuredContent=result, isError=False)
            except MCPPublicError as exc:
                error = {"success": False, "error": {"code": exc.code, "message": ERRORS[exc.code]}}
            except Exception:
                error = {"success": False, "error": {"code": "backend_error", "message": ERRORS["backend_error"]}}
            return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(error))],
                                        structuredContent=error, isError=True)

    server = PublicFastMCP("AgentRecall")
    if hasattr(server, "_lowlevel_server"):
        _protect_sdk2_handlers(server)
        return server
    # Avoid SDK unknown-tool and output-validation exception echo paths too.
    async def handle_call(request):
        return types.ServerResult(await server.call_tool(request.params.name, request.params.arguments))
    server._mcp_server.request_handlers[types.CallToolRequest] = handle_call

    # The SDK's public dispatch table covers prompts/resources/completion too.
    # HTTP status filtering cannot protect errors inside successful SSE streams.
    # ErrorData is not a ServerResult variant: use McpError for RPC failures.
    from mcp.shared.exceptions import McpError

    def public_handler(handler):
        async def handle(request):
            try:
                result = await handler(request)
                root = result.root if isinstance(result, types.ServerResult) else result
                if isinstance(root, types.ErrorData):
                    raise McpError(root)
                if isinstance(root, types.CallToolResult) and root.isError:
                    details = (root.structuredContent or {}).get("error", {})
                    code = details.get("code") if isinstance(details, dict) else None
                    if not isinstance(code, str) or code not in ERRORS:
                        code = "backend_error"
                    error = {"success": False, "error": {"code": code, "message": ERRORS[code]}}
                    safe = types.CallToolResult(isError=True, structuredContent=error,
                        content=[types.TextContent(type="text", text=json.dumps(error))])
                    return types.ServerResult(safe) if isinstance(result, types.ServerResult) else safe
                return result
            except McpError as exc:
                raise McpError(types.ErrorData(code=exc.error.code, message="Operation failed.")) from None
            except Exception:
                raise McpError(types.ErrorData(code=types.INTERNAL_ERROR, message="Operation failed.")) from None
        return handle

    for request_type, handler in tuple(server._mcp_server.request_handlers.items()):
        server._mcp_server.request_handlers[request_type] = public_handler(handler)
    return server


def _protect_sdk2_handlers(server):
    """SDK 2 uses method-keyed, context/params handlers and unwrapped results."""
    from mcp import types
    from mcp.server.lowlevel.server import HandlerEntry
    from mcp.shared.exceptions import MCPError

    def public_handler(handler):
        async def handle(context, params):
            try:
                result = await handler(context, params)
                if isinstance(result, types.ErrorData):
                    raise MCPError(result.code, "Operation failed.")
                if isinstance(result, types.CallToolResult) and result.is_error:
                    details = (result.structured_content or {}).get("error", {})
                    code = details.get("code") if isinstance(details, dict) else None
                    if not isinstance(code, str) or code not in ERRORS:
                        code = "backend_error"
                    error = {"success": False, "error": {"code": code, "message": ERRORS[code]}}
                    return types.CallToolResult(isError=True, structuredContent=error,
                        content=[types.TextContent(type="text", text=json.dumps(error))])
                return result
            except MCPError as exc:
                raise MCPError(exc.error.code, "Operation failed.") from None
            except Exception:
                raise MCPError(types.INTERNAL_ERROR, "Operation failed.") from None
        return handle

    lowlevel = server._lowlevel_server
    for method, entry in tuple(lowlevel._request_handlers.items()):
        lowlevel._request_handlers[method] = HandlerEntry(entry.params_type, public_handler(entry.handler))


def create_adapter_from_config(
    config_path: str | Path,
    *,
    workspace_id: str = "",
    agent_id: str = "",
    session_id: str = "",
    access: str = "",
    tools: list[str] | None = None,
) -> MCPAdapter:
    path = Path(config_path).expanduser().resolve()
    config = load_config(path.parent, path)
    workspace = workspace_id or str(config.get("workspace_id") or "agent-recall")
    agent = agent_id or str(config.get("agent_id") or "mcp")
    session = session_id or os.environ.get("AGENT_RECALL_SESSION", "")
    core = AgentRecallCore(
        config,
        AgentIdentity(workspace, agent, session),
        curator_factory=lambda: build_curator(config),
    )
    return MCPAdapter(core, access=access or str(config.get("mcp_access") or "read-only"),
                      tools=tools if tools is not None else config.get("mcp_tools"))


def run_http_server(server, *, transport: str, host: str, port: int, bearer_token: str = "") -> None:
    """Run HTTP MCP with fail-closed static bearer authentication by default."""
    try:
        import uvicorn

    except ImportError as exc:  # pragma: no cover - provided by the MCP HTTP extra
        raise RuntimeError("MCP HTTP transport requires uvicorn and starlette") from exc

    if hasattr(server, "_lowlevel_server"):
        from mcp.server.transport_security import TransportSecuritySettings
        security = TransportSecuritySettings(enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"])
        app = (server.streamable_http_app(transport_security=security) if transport == "streamable-http"
               else server.sse_app(transport_security=security))
    else:
        security = server.settings.transport_security
        app = server.streamable_http_app() if transport == "streamable-http" else server.sse_app()
    app = HTTPBoundary(app, bearer_token=bearer_token, security_settings=security)

    uvicorn.run(app, host=host, port=port, log_level="info")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentRecall as an optional MCP server")
    parser.add_argument(
        "--config",
        default=os.environ.get("AGENT_RECALL_CONFIG", str(Path.home() / ".agent-recall" / "agent-recall.json")),
    )
    parser.add_argument("--workspace", default=os.environ.get("AGENT_RECALL_WORKSPACE", ""))
    parser.add_argument("--agent", default=os.environ.get("AGENT_RECALL_AGENT", ""))
    parser.add_argument("--session", default=os.environ.get("AGENT_RECALL_SESSION", ""))
    parser.add_argument(
        "--access", choices=["read-only", "read-write"], default=os.environ.get("AGENT_RECALL_MCP_ACCESS", "")
    )
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--auth-token-env",
        default=os.environ.get("AGENT_RECALL_MCP_TOKEN_ENV", "AGENT_RECALL_MCP_TOKEN"),
        help="Environment variable containing the HTTP bearer token",
    )
    parser.add_argument(
        "--allow-unauthenticated-http",
        action="store_true",
        help="Explicitly allow HTTP MCP without bearer authentication (unsafe on public interfaces)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bearer_token = ""
    if args.transport != "stdio":
        bearer_token = os.environ.get(args.auth_token_env, "") if args.auth_token_env else ""
        if not bearer_token and not args.allow_unauthenticated_http:
            raise RuntimeError(
                f"HTTP MCP requires a bearer token in {args.auth_token_env!r}; "
                "set it or explicitly pass --allow-unauthenticated-http"
            )
    adapter = create_adapter_from_config(
        args.config,
        workspace_id=args.workspace,
        agent_id=args.agent,
        session_id=args.session,
        access=args.access,
    )
    server = build_fastmcp(adapter)
    try:
        if args.transport == "stdio":
            server.run(transport="stdio")
        else:
            run_http_server(
                server,
                transport=args.transport,
                host=args.host,
                port=args.port,
                bearer_token=bearer_token,
            )
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
