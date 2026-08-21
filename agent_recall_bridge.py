from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    from .agent_recall_core import AgentIdentity, AgentRecallCore, build_curator, load_config
except ImportError:
    from agent_recall_core import AgentIdentity, AgentRecallCore, build_curator, load_config


MAX_IDENTITY_CHARS = 200
MAX_SESSION_CHARS = 1000
MAX_REQUEST_BYTES = 1_000_000


class BridgeProtocolError(ValueError):
    pass


def _required_identity(value: Any) -> AgentIdentity:
    if not isinstance(value, dict):
        raise BridgeProtocolError("identity must be an object")
    parts: dict[str, str] = {}
    for key in ("workspace_id", "agent_id", "session_id"):
        item = value.get(key, "")
        if not isinstance(item, str):
            raise BridgeProtocolError(f"identity.{key} must be a string")
        item = item.strip()
        if key != "session_id" and not item:
            raise BridgeProtocolError(f"identity.{key} is required")
        limit = MAX_SESSION_CHARS if key == "session_id" else MAX_IDENTITY_CHARS
        if len(item) > limit:
            raise BridgeProtocolError(f"identity.{key} exceeds {limit} characters")
        parts[key] = item
    user_id = value.get("user_id", "")
    if not isinstance(user_id, str) or len(user_id) > MAX_IDENTITY_CHARS:
        raise BridgeProtocolError(f"identity.user_id must be a string up to {MAX_IDENTITY_CHARS} characters")
    return AgentIdentity(parts["workspace_id"], parts["agent_id"], parts["session_id"], user_id.strip())


class AgentRecallBridge:
    """Bounded in-process core registry behind a local JSONL adapter protocol."""

    def __init__(self, *, config_path: str | Path | None = None, max_contexts: int = 128) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else None
        base_dir = self.config_path.parent if self.config_path else Path.home() / ".agent-recall"
        self.config = load_config(base_dir, self.config_path)
        self.max_contexts = max(1, min(int(max_contexts), 1024))
        self._cores: OrderedDict[tuple[str, str, str, str], AgentRecallCore] = OrderedDict()

    @property
    def context_count(self) -> int:
        return len(self._cores)

    @staticmethod
    def _key(identity: AgentIdentity) -> tuple[str, str, str, str]:
        return (identity.workspace_id, identity.agent_id, identity.session_id, identity.user_id)

    def _core(self, identity: AgentIdentity) -> AgentRecallCore:
        key = self._key(identity)
        existing = self._cores.pop(key, None)
        if existing is not None:
            self._cores[key] = existing
            return existing
        while len(self._cores) >= self.max_contexts:
            _, stale = self._cores.popitem(last=False)
            stale.close()
        core = AgentRecallCore(
            self.config,
            identity,
            curator_factory=lambda: build_curator(self.config),
        )
        self._cores[key] = core
        return core

    def _release(self, identity: AgentIdentity) -> bool:
        core = self._cores.pop(self._key(identity), None)
        if core is None:
            return False
        core.close()
        return True

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise BridgeProtocolError("request must be an object")
        operation = request.get("operation")
        if not isinstance(operation, str) or not operation:
            raise BridgeProtocolError("operation is required")
        identity = _required_identity(request.get("identity"))
        args = request.get("args", {})
        if not isinstance(args, dict):
            raise BridgeProtocolError("args must be an object")

        if operation == "release":
            return {"success": True, "released": self._release(identity)}
        if operation == "shutdown":
            self.close()
            return {"success": True, "shutdown": True}

        core = self._core(identity)
        if operation == "remember":
            return core.remember(args)
        if operation == "search":
            return core.search(args)
        if operation == "get_memory":
            return core.get_memory(int(args["id"]))
        if operation == "prefetch_context":
            return core.prefetch_context(
                str(args.get("query") or ""),
                limit=int(args.get("limit") or core.config.get("prefetch_limit") or 6),
                max_chars=int(args.get("max_chars") or core.config.get("max_memory_chars") or 12_000),
                explain=bool(args.get("explain", True)),
                include_results=bool(args.get("include_results", False)),
                track_access=bool(args.get("track_access", True)),
            )
        if operation == "update":
            return core.update(int(args["id"]), args)
        if operation == "forget":
            return core.forget(int(args["id"]))
        if operation == "curate":
            return core.curate(args)
        if operation == "conclude":
            return core.conclude(args)
        if operation == "profile":
            return core.profile(
                str(args.get("focus") or ""),
                int(args.get("limit") or 10),
                track_access=bool(args.get("track_access", True)),
            )
        if operation == "profile_synthesize":
            return core.profile_synthesize(args)
        if operation == "review":
            return core.review(args)
        if operation == "stats":
            return core.stats()
        if operation == "health":
            return core.health()
        if operation == "capabilities":
            return core.capabilities()
        if operation == "import_markdown":
            return core.import_markdown(args)
        if operation == "capture_turn":
            captured = core.capture_turn(
                str(args.get("user_content") or ""),
                str(args.get("assistant_content") or ""),
                session_id=identity.session_id,
                background=False,
            )
            return {"success": True, "captured": captured}
        if operation == "save_checkpoint":
            messages = args.get("messages")
            if not isinstance(messages, list):
                raise BridgeProtocolError("save_checkpoint requires args.messages as an array")
            memory_id = core.checkpoint_from_messages(messages)
            return {"success": True, "saved": memory_id is not None, "id": memory_id}
        raise BridgeProtocolError(f"unsupported operation: {operation}")

    def close(self) -> None:
        for core in self._cores.values():
            core.close()
        self._cores.clear()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgentRecall local JSONL bridge for native host adapters")
    parser.add_argument("--config", default=None, help="Path to agent-recall.json")
    parser.add_argument("--max-contexts", type=int, default=128)
    return parser


def serve_jsonl(bridge: AgentRecallBridge) -> int:
    stream = sys.stdin.buffer
    while True:
        raw = stream.readline(MAX_REQUEST_BYTES + 1)
        if not raw:
            break
        if len(raw) > MAX_REQUEST_BYTES:
            while raw and not raw.endswith(b"\n"):
                raw = stream.readline(MAX_REQUEST_BYTES + 1)
            response = {"id": None, "ok": False, "error": "request exceeds bridge size limit"}
            print(json.dumps(response), flush=True)
            continue
        request: Any = None
        try:
            line = raw.decode("utf-8")
            request = json.loads(line)
            request_id = request.get("id") if isinstance(request, dict) else None
            result = bridge.dispatch(request)
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:
            response = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)
        if isinstance(request, dict) and request.get("operation") == "shutdown":
            break
    bridge.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bridge = AgentRecallBridge(config_path=args.config, max_contexts=args.max_contexts)
    return serve_jsonl(bridge)


if __name__ == "__main__":
    raise SystemExit(main())
