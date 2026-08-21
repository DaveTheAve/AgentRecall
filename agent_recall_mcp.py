from __future__ import annotations

import argparse
import hmac
import os
from pathlib import Path
from typing import Any

try:
    from .agent_recall_core import AgentIdentity, AgentRecallCore, build_curator, load_config
except ImportError:
    from agent_recall_core import AgentIdentity, AgentRecallCore, build_curator, load_config


class MCPAccessError(PermissionError):
    pass


def _sanitize_public(value: Any) -> Any:
    if isinstance(value, list):
        return [_sanitize_public(item) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        normalized = key.lower().replace("_", "")
        if normalized in {"dbpath", "sessionid"}:
            continue
        if normalized.endswith("path") and isinstance(item, str) and os.path.isabs(item):
            result[key] = "[redacted]"
        else:
            result[key] = _sanitize_public(item)
    return result


class MCPAdapter:
    """Curated public AgentRecall API for MCP and other tool transports."""

    READ_TOOLS = {
        "search",
        "prefetch_context",
        "get_memory",
        "profile",
        "stats",
        "health",
        "capabilities",
    }
    WRITE_TOOLS = {
        "remember",
        "update",
        "forget",
        "curate",
        "conclude",
        "profile_synthesize",
        "review",
    }
    TOOLS = tuple(sorted(READ_TOOLS | WRITE_TOOLS))

    def __init__(self, core: AgentRecallCore, *, access: str = "read-write") -> None:
        if access not in {"read-only", "read-write"}:
            raise ValueError("MCP access must be 'read-only' or 'read-write'")
        self.core = core
        self.access = access

    def _authorize(self, operation: str) -> None:
        if operation not in self.TOOLS:
            raise KeyError(f"Unknown AgentRecall MCP operation: {operation}")
        if self.access == "read-only" and operation in self.WRITE_TOOLS:
            raise MCPAccessError(f"AgentRecall MCP client is read-only; operation {operation!r} is not allowed")

    def call(self, operation: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        self._authorize(operation)
        values = dict(args or {})
        track_access = self.access != "read-only"
        if operation == "remember":
            result = self.core.remember(values)
        elif operation == "search":
            values["_track_access"] = track_access
            result = self.core.search(values)
        elif operation == "prefetch_context":
            result = self.core.prefetch_context(
                str(values.get("query") or ""),
                limit=int(values.get("limit") or self.core.config.get("prefetch_limit", 6)),
                max_chars=int(values.get("max_chars") or self.core.config.get("max_memory_chars", 12_000)),
                explain=bool(values.get("explain", True)),
                include_results=bool(values.get("include_results", False)),
                track_access=track_access,
            )
        elif operation == "get_memory":
            result = self.core.get_memory(
                int(values["id"]),
                include_shared=bool(values.get("include_shared", True)),
            )
        elif operation == "profile":
            result = self.core.profile(
                str(values.get("focus") or ""),
                int(values.get("limit") or 10),
                track_access=track_access,
            )
        elif operation == "stats":
            result = self.core.stats()
        elif operation == "health":
            result = self.core.health()
        elif operation == "capabilities":
            result = self.core.capabilities()
            result.update({"access": self.access, "tools": list(self.TOOLS)})
        elif operation == "update":
            result = self.core.update(int(values["id"]), values)
        elif operation == "forget":
            result = self.core.forget(int(values["id"]))
        elif operation == "curate":
            result = self.core.curate(values)
        elif operation == "conclude":
            result = self.core.conclude(values)
        elif operation == "profile_synthesize":
            result = self.core.profile_synthesize(values)
        elif operation == "review":
            result = self.core.review(values)
        else:
            raise KeyError(f"Unknown AgentRecall MCP operation: {operation}")
        return _sanitize_public(result)

    def close(self) -> None:
        self.core.close()


def build_fastmcp(adapter: MCPAdapter):
    """Build an MCP SDK server lazily so the core/Hermes path has no MCP dependency."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised by CLI smoke with optional dependency
        raise RuntimeError("The optional 'mcp' package is required to run the AgentRecall MCP server") from exc

    server = FastMCP("AgentRecall")

    @server.tool(
        name="remember", description="Store a durable memory under the server's fixed workspace/agent identity."
    )
    def remember(
        content: str,
        title: str = "",
        summary: str = "",
        visibility: str = "agent",
        category: str = "general",
        tags: list[str] | None = None,
        importance: float = 0.5,
        confidence: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return adapter.call(
            "remember",
            {
                "content": content,
                "title": title,
                "summary": summary,
                "visibility": visibility,
                "category": category,
                "tags": tags or [],
                "importance": importance,
                "confidence": confidence,
                "metadata": metadata or {},
            },
        )

    @server.tool(
        name="search",
        description="Hybrid search over memories visible to the fixed server identity, with optional score explanations.",
    )
    def search(
        query: str,
        limit: int = 8,
        include_shared: bool = True,
        category: str = "",
        tags: list[str] | None = None,
        visibility: str = "",
        source_agent_id: str = "",
        min_importance: float | None = None,
        updated_after: float | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        return adapter.call(
            "search",
            {
                "query": query,
                "limit": limit,
                "include_shared": include_shared,
                "category": category,
                "tags": tags or [],
                "visibility": visibility,
                "source_agent_id": source_agent_id,
                "min_importance": min_importance,
                "updated_after": updated_after,
                "explain": explain,
            },
        )

    @server.tool(
        name="prefetch_context",
        description="Return a bounded, ready-to-inject recall block and explain why memories ranked.",
    )
    def prefetch_context(
        query: str,
        limit: int = 6,
        max_chars: int = 12_000,
        explain: bool = True,
        include_results: bool = False,
    ) -> dict[str, Any]:
        return adapter.call(
            "prefetch_context",
            {
                "query": query,
                "limit": limit,
                "max_chars": max_chars,
                "explain": explain,
                "include_results": include_results,
            },
        )

    @server.tool(name="get_memory", description="Inspect one visible memory including provenance and ACL metadata.")
    def get_memory(id: int, include_shared: bool = True) -> dict[str, Any]:
        return adapter.call("get_memory", {"id": id, "include_shared": include_shared})

    @server.tool(name="update", description="Update, retag, promote/demote, or archive an owned visible memory.")
    def update(
        id: int,
        content: str | None = None,
        title: str | None = None,
        summary: str | None = None,
        visibility: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        archived: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        values = {"id": id}
        for key, value in {
            "content": content,
            "title": title,
            "summary": summary,
            "visibility": visibility,
            "category": category,
            "tags": tags,
            "importance": importance,
            "confidence": confidence,
            "archived": archived,
            "metadata": metadata,
        }.items():
            if value is not None:
                values[key] = value
        return adapter.call("update", values)

    @server.tool(name="forget", description="Delete an owned visible memory while enforcing AgentRecall ACLs.")
    def forget(id: int) -> dict[str, Any]:
        return adapter.call("forget", {"id": id})

    @server.tool(
        name="curate",
        description="Extract durable memory candidates with the configured curator; supports dry-run review.",
    )
    def curate(text: str, default_visibility: str = "agent", dry_run: bool = True) -> dict[str, Any]:
        return adapter.call(
            "curate",
            {"text": text, "default_visibility": default_visibility, "dry_run": dry_run},
        )

    @server.tool(
        name="conclude", description="Store an inspectable, source-linked conclusion when conclusions are enabled."
    )
    def conclude(
        content: str,
        scope: str = "general",
        subject: str = "",
        source_ids: list[int] | None = None,
        visibility: str = "agent",
        confidence: float = 0.8,
        supersedes: list[int] | None = None,
    ) -> dict[str, Any]:
        return adapter.call(
            "conclude",
            {
                "content": content,
                "scope": scope,
                "subject": subject,
                "source_ids": source_ids or [],
                "visibility": visibility,
                "confidence": confidence,
                "supersedes": supersedes or [],
            },
        )

    @server.tool(name="profile", description="Return identity, isolation settings, and focused visible memories.")
    def profile(focus: str = "", limit: int = 10) -> dict[str, Any]:
        return adapter.call("profile", {"focus": focus, "limit": limit})

    @server.tool(
        name="profile_synthesize", description="Synthesize a peer/workspace/agent profile, dry-run by default."
    )
    def profile_synthesize(
        scope: str = "peer",
        subject: str = "",
        focus: str = "",
        dry_run: bool = True,
        visibility: str = "agent",
    ) -> dict[str, Any]:
        return adapter.call(
            "profile_synthesize",
            {
                "scope": scope,
                "subject": subject,
                "focus": focus,
                "dry_run": dry_run,
                "visibility": visibility,
            },
        )

    @server.tool(name="review", description="Run bounded conflict/staleness/promotion review, dry-run by default.")
    def review(focus: str = "", dry_run: bool = True, limit: int = 20) -> dict[str, Any]:
        return adapter.call("review", {"focus": focus, "dry_run": dry_run, "limit": limit})

    @server.tool(name="stats", description="Show ACL-filtered memory counts for the server identity.")
    def stats() -> dict[str, Any]:
        return adapter.call("stats", {})

    @server.tool(
        name="health", description="Inspect SQLite integrity/concurrency settings and embedding configuration."
    )
    def health() -> dict[str, Any]:
        return adapter.call("health", {})

    @server.tool(
        name="capabilities",
        description="Describe enabled AgentRecall features, fixed identity, access mode, and tools.",
    )
    def capabilities() -> dict[str, Any]:
        return adapter.call("capabilities", {})

    return server


def create_adapter_from_config(
    config_path: str | Path,
    *,
    workspace_id: str = "",
    agent_id: str = "",
    session_id: str = "",
    access: str = "",
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
    return MCPAdapter(core, access=access or str(config.get("mcp_access") or "read-write"))


def run_http_server(server, *, transport: str, host: str, port: int, bearer_token: str = "") -> None:
    """Run HTTP MCP with fail-closed static bearer authentication by default."""
    try:
        import uvicorn
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - provided by the MCP HTTP extra
        raise RuntimeError("MCP HTTP transport requires uvicorn and starlette") from exc

    app = server.streamable_http_app() if transport == "streamable-http" else server.sse_app()
    if bearer_token:

        class BearerAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                supplied = request.headers.get("authorization", "")
                expected = f"Bearer {bearer_token}"
                if not hmac.compare_digest(supplied, expected):
                    return JSONResponse(
                        {"error": "unauthorized"},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                return await call_next(request)

        app.add_middleware(BearerAuthMiddleware)

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
