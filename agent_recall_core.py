from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .agent_recall_curator import ChatCompletionsCurator, CodexCliCurator
    from .agent_recall_store import AgentRecallStore, EmbeddingClient, _json_dumps, normalize_text
except ImportError:
    from agent_recall_curator import ChatCompletionsCurator, CodexCliCurator
    from agent_recall_store import AgentRecallStore, EmbeddingClient, _json_dumps, normalize_text


class AgentRecallError(RuntimeError):
    """Public operation error suitable for host adapters to translate."""


@dataclass(frozen=True)
class AgentIdentity:
    workspace_id: str
    agent_id: str
    session_id: str = ""
    user_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "workspace_id": self.workspace_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except Exception:
        return default


def default_config(base_dir: str | Path) -> dict[str, Any]:
    home = Path(base_dir).expanduser()
    return {
        "db_path": str(home / "agent-recall.db"),
        "workspace_id": os.environ.get("AGENT_RECALL_WORKSPACE", "hermes"),
        "agent_id": os.environ.get("AGENT_RECALL_AGENT", ""),
        "embedding_base_url": os.environ.get("AGENT_RECALL_EMBEDDING_BASE_URL", "http://127.0.0.1:6660/v1"),
        "embedding_model": os.environ.get("AGENT_RECALL_EMBEDDING_MODEL", "qwen3-embedding-4b"),
        "embedding_api_key_env": os.environ.get("AGENT_RECALL_EMBEDDING_API_KEY_ENV", "LLM_OPENAI_API_KEY"),
        "embedding_dimensions": _env_int("AGENT_RECALL_EMBEDDING_DIMENSIONS", 0),
        "embedding_timeout": 20.0,
        "sqlite_busy_timeout_ms": 5_000,
        "default_visibility": "agent",
        "shared_recall": True,
        "raw_memories_enabled": True,
        "curated_memories_enabled": True,
        "conclusions_enabled": False,
        "peer_profiles_enabled": False,
        "workspace_profiles_enabled": False,
        "agent_profiles_enabled": False,
        "dialectic_review_enabled": False,
        "conflict_detection_enabled": False,
        "staleness_detection_enabled": False,
        "promotion_rules_enabled": False,
        "demotion_rules_enabled": False,
        "auto_capture_turns": False,
        "auto_capture_compression_checkpoints": False,
        "auto_capture_visibility": "session",
        "prefetch_limit": 6,
        "max_memory_chars": 12_000,
        "allow_any_agent_to_mutate_shared": False,
        "llm_curator_enabled": True,
        "llm_curator_backend": os.environ.get("AGENT_RECALL_LLM_CURATOR_BACKEND", "codex-cli"),
        "llm_curator_model": os.environ.get("AGENT_RECALL_LLM_CURATOR_MODEL", "gpt-5.3-mini"),
        "llm_curator_command": os.environ.get("AGENT_RECALL_LLM_CURATOR_COMMAND", "codex"),
        "llm_curator_base_url": os.environ.get("AGENT_RECALL_LLM_CURATOR_BASE_URL", ""),
        "llm_curator_api_key_env": os.environ.get("AGENT_RECALL_LLM_CURATOR_API_KEY_ENV", "OPENAI_API_KEY"),
        "llm_curator_timeout": 120.0,
        "import_roots": [str(home)],
        "max_import_bytes": 2_000_000,
        "excluded_terms": [],
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _load_json_strict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"AgentRecall config file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"AgentRecall config is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"AgentRecall config root must be an object: {path}")
    for key in (
        "db_path",
        "workspace_id",
        "agent_id",
        "embedding_base_url",
        "embedding_model",
        "llm_curator_backend",
        "llm_curator_base_url",
        "llm_curator_command",
    ):
        if key in value and not isinstance(value[key], str):
            raise ValueError(f"AgentRecall config {key!r} must be a string")
    if "db_path" in value and not value["db_path"].strip():
        raise ValueError("AgentRecall config 'db_path' must not be empty")
    for key in ("excluded_terms", "import_roots"):
        if key in value and (not isinstance(value[key], list) or not all(isinstance(item, str) for item in value[key])):
            raise ValueError(f"AgentRecall config {key!r} must be an array of strings")
    return value


def load_config(base_dir: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    home = Path(base_dir).expanduser()
    path = Path(config_path).expanduser() if config_path else home / "agent-recall.json"
    config = default_config(home)
    config.update(_load_json_strict(path) if config_path is not None else _load_json(path))
    for marker in ("$HERMES_HOME", "${HERMES_HOME}", "$AGENT_RECALL_HOME", "${AGENT_RECALL_HOME}"):
        config["db_path"] = str(config.get("db_path", "")).replace(marker, str(home))
    return normalize_config(config)


def build_curator(
    config: dict[str, Any],
    *,
    codex_cls=CodexCliCurator,
    chat_cls=ChatCompletionsCurator,
):
    """Build the configured curation backend for any host adapter."""
    backend = str(config.get("llm_curator_backend") or "codex-cli")
    if backend == "codex-cli":
        return codex_cls(
            command=str(config.get("llm_curator_command") or "codex"),
            model=str(config.get("llm_curator_model") or "gpt-5.3-mini"),
            timeout=float(config.get("llm_curator_timeout") or 120.0),
        )
    if backend in {"openai-compatible", "chat-completions"}:
        api_key = os.environ.get(str(config.get("llm_curator_api_key_env") or ""), "")
        return chat_cls(
            base_url=str(config.get("llm_curator_base_url") or ""),
            model=str(config.get("llm_curator_model") or "gpt-5.3-mini"),
            api_key=api_key,
            timeout=float(config.get("llm_curator_timeout") or 120.0),
        )
    raise ValueError("Unsupported llm_curator_backend; use codex-cli or openai-compatible")


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    bool_defaults = {
        "shared_recall": True,
        "raw_memories_enabled": True,
        "curated_memories_enabled": True,
        "conclusions_enabled": False,
        "peer_profiles_enabled": False,
        "workspace_profiles_enabled": False,
        "agent_profiles_enabled": False,
        "dialectic_review_enabled": False,
        "conflict_detection_enabled": False,
        "staleness_detection_enabled": False,
        "promotion_rules_enabled": False,
        "demotion_rules_enabled": False,
        "auto_capture_turns": False,
        "auto_capture_compression_checkpoints": False,
        "allow_any_agent_to_mutate_shared": False,
        "llm_curator_enabled": True,
    }
    for key, default in bool_defaults.items():
        out[key] = _as_bool(out.get(key), default)
    for key in [
        "embedding_dimensions",
        "prefetch_limit",
        "max_memory_chars",
        "max_import_bytes",
        "sqlite_busy_timeout_ms",
    ]:
        if key in out:
            try:
                out[key] = int(out[key] or 0)
            except Exception:
                out[key] = 0
    for key in ["embedding_timeout", "llm_curator_timeout"]:
        fallback = 120.0 if key == "llm_curator_timeout" else 20.0
        try:
            out[key] = float(out.get(key) or fallback)
        except Exception:
            out[key] = fallback
    return out


def normalize_tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("tags must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("tags must be a list of strings")
        text = item.strip()
        if text:
            result.append(text)
    return result


def normalize_metadata(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    return value


class AgentRecallCore:
    """Host-neutral, in-process AgentRecall memory service."""

    OPERATIONS = (
        "remember",
        "search",
        "prefetch_context",
        "get_memory",
        "profile",
        "update",
        "forget",
        "curate",
        "conclude",
        "profile_synthesize",
        "review",
        "stats",
        "import_markdown",
        "health",
        "capabilities",
    )

    def __init__(
        self,
        config: dict[str, Any],
        identity: AgentIdentity,
        *,
        curator_factory: Callable[[], Any] | None = None,
    ) -> None:
        merged = default_config(Path(str(config.get("db_path") or ".")).expanduser().parent)
        merged.update(config)
        self.config = normalize_config(merged)
        self.identity = identity
        self.store = AgentRecallStore(
            self.config["db_path"],
            busy_timeout_ms=int(self.config.get("sqlite_busy_timeout_ms") or 5_000),
        )
        api_key = os.environ.get(str(self.config.get("embedding_api_key_env") or ""), "")
        self.embedder: Any = EmbeddingClient(
            str(self.config.get("embedding_base_url") or ""),
            str(self.config.get("embedding_model") or ""),
            api_key=api_key,
            dimensions=int(self.config.get("embedding_dimensions") or 0),
            timeout=float(self.config.get("embedding_timeout") or 20.0),
        )
        self.curator_factory = curator_factory
        self.last_embedding_error = ""
        self._sync_threads: list[threading.Thread] = []

    def _excluded(self, text: str) -> str | None:
        for term in self.config.get("excluded_terms", []) or []:
            if term and re.search(re.escape(str(term)), text or "", re.IGNORECASE):
                return str(term)
        return None

    def _embedding(self, text: str) -> list[float]:
        self.last_embedding_error = ""
        if not self.embedder:
            return []
        try:
            return self.embedder.embed(text)
        except Exception as exc:
            self.last_embedding_error = str(exc)[-500:]
            return []

    def _make_curator(self):
        if self.curator_factory:
            return self.curator_factory()
        try:
            return build_curator(self.config)
        except ValueError as exc:
            raise AgentRecallError(str(exc)) from exc

    def rotate_session(self, session_id: str) -> None:
        """Update the active session without rebuilding storage or adapters."""
        self.identity = AgentIdentity(
            self.identity.workspace_id,
            self.identity.agent_id,
            session_id or "",
            self.identity.user_id,
        )

    def remember(self, args: dict[str, Any], *, identity: AgentIdentity | None = None) -> dict[str, Any]:
        if not self.config.get("raw_memories_enabled", True):
            raise AgentRecallError("Raw memory storage is disabled by raw_memories_enabled=false")
        scope = identity or self.identity
        visibility = str(args.get("visibility") or self.config.get("default_visibility") or "agent")
        if visibility == "session" and not scope.session_id:
            raise AgentRecallError("Session-scoped memory requires a non-empty session identity")
        content = normalize_text(str(args.get("content") or ""), int(self.config.get("max_memory_chars") or 12_000))
        blocked = self._excluded(content)
        if blocked:
            raise AgentRecallError(f"Refusing to store content matching excluded term: {blocked}")
        embedding = self._embedding("\n".join([str(args.get("title") or ""), str(args.get("summary") or ""), content]))
        memory_id = self.store.add_memory(
            workspace_id=scope.workspace_id,
            agent_id=scope.agent_id,
            source_agent_id=scope.agent_id,
            user_id=scope.user_id,
            session_id=scope.session_id,
            visibility=visibility,
            category=str(args.get("category") or "general"),
            title=str(args.get("title") or ""),
            content=content,
            summary=str(args.get("summary") or ""),
            tags=normalize_tags(args.get("tags")),
            metadata=normalize_metadata(args.get("metadata")),
            embedding=embedding,
            embedding_model=str(self.config.get("embedding_model") or ""),
            importance=float(args.get("importance", 0.5)),
            confidence=float(args.get("confidence", 0.8)),
        )
        result: dict[str, Any] = {
            "success": True,
            "id": memory_id,
            "visibility": visibility,
        }
        if self.last_embedding_error:
            result["embedding_warning"] = self.last_embedding_error
        return result

    def search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        embedding = self._embedding(query) if query else []
        results = self.store.search(
            workspace_id=self.identity.workspace_id,
            agent_id=self.identity.agent_id,
            session_id=self.identity.session_id,
            query=query,
            query_embedding=embedding,
            include_shared=bool(args.get("include_shared", self.config.get("shared_recall", True))),
            category=str(args.get("category") or ""),
            tags=args.get("tags") or [],
            visibility=str(args.get("visibility") or ""),
            source_agent_id=str(args.get("source_agent_id") or ""),
            min_importance=args.get("min_importance"),
            updated_after=args.get("updated_after"),
            min_score=args.get("min_score"),
            explain=bool(args.get("explain", False)),
            track_access=bool(args.get("_track_access", True)),
            limit=int(args.get("limit") or 8),
        )
        result: dict[str, Any] = {"success": True, "results": results, "count": len(results)}
        if self.last_embedding_error:
            result["embedding_warning"] = self.last_embedding_error
        return result

    def get_memory(self, memory_id: int, *, include_shared: bool = True) -> dict[str, Any]:
        memory = self.store.get_memory(
            int(memory_id),
            self.identity.workspace_id,
            self.identity.agent_id,
            self.identity.session_id,
            include_shared=include_shared,
        )
        if not memory:
            return {"success": False, "error": "Memory not found or not visible"}
        return {"success": True, "memory": memory}

    def prefetch_context(
        self,
        query: str,
        *,
        limit: int | None = None,
        max_chars: int | None = None,
        explain: bool = False,
        include_results: bool = True,
        track_access: bool = True,
    ) -> dict[str, Any]:
        if not query:
            return {
                "success": True,
                "context": "",
                "results": [],
                "count": 0,
                "identity": self.identity.as_dict(),
            }
        result = self.search(
            {
                "query": query,
                "limit": limit or self.config.get("prefetch_limit", 6),
                "include_shared": self.config.get("shared_recall", True),
                "explain": explain,
                "_track_access": track_access,
            }
        )
        rows = result["results"]
        lines = ["# AgentRecall Recalled Context"]
        budget = max(0, int(max_chars if max_chars is not None else self.config.get("max_memory_chars", 12_000)))
        for row in rows:
            scope = f"{row['visibility']}:{row['agent_id']}"
            title = f"{row['title']} — " if row.get("title") else ""
            reason = ""
            if explain and row.get("score_explanation"):
                scores = row["score_explanation"]
                reason = f" reason=vector:{scores['vector']},lexical:{scores['lexical']},importance:{scores['importance']},recency:{scores['recency']}"
            line = f"- [{row['id']} {scope} score={row['score']}{reason}] {title}{row['content']}"
            candidate = "\n".join(lines + [line])
            if budget and len(candidate) > budget:
                remaining = budget - len("\n".join(lines)) - 1
                if remaining > 20:
                    lines.append(line[:remaining])
                break
            lines.append(line)
        context = "\n".join(lines) if len(lines) > 1 else ""
        return {
            "success": True,
            "context": context[:budget] if budget else context,
            "results": rows if include_results else [],
            "count": len(rows),
            "identity": self.identity.as_dict(),
            **({"embedding_warning": result["embedding_warning"]} if "embedding_warning" in result else {}),
        }

    def profile(self, focus: str = "", limit: int = 10, *, track_access: bool = True) -> dict[str, Any]:
        query = focus or "user preferences project conventions agent memory"
        recall = self.search({"query": query, "limit": limit, "_track_access": track_access})["results"]
        return {
            "success": True,
            "workspace_id": self.identity.workspace_id,
            "agent_id": self.identity.agent_id,
            "db_path": self.config.get("db_path"),
            "recall": recall,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "success": True,
            "stats": self.store.stats(
                self.identity.workspace_id,
                self.identity.agent_id,
                self.identity.session_id,
            ),
        }

    def forget(self, memory_id: int) -> dict[str, Any]:
        deleted = self.store.delete_memory(
            int(memory_id),
            self.identity.workspace_id,
            self.identity.agent_id,
            self.identity.session_id,
            allow_shared_mutation=bool(self.config.get("allow_any_agent_to_mutate_shared", False)),
        )
        return {"success": deleted, "deleted": deleted}

    def update(self, memory_id: int, args: dict[str, Any]) -> dict[str, Any]:
        if str(args.get("visibility") or "") == "session" and not self.identity.session_id:
            raise AgentRecallError("Session-scoped memory requires a non-empty session identity")
        updates: dict[str, Any] = {}
        for key in ["content", "title", "summary", "category", "visibility", "importance", "confidence"]:
            if key in args:
                updates[key] = normalize_text(str(args[key])) if key == "content" else args[key]
        if updates.get("visibility") == "session":
            updates["target_session_id"] = self.identity.session_id
        if "tags" in args:
            updates["tags_json"] = _json_dumps(normalize_tags(args.get("tags")))
        if "metadata" in args:
            updates["metadata_json"] = _json_dumps(normalize_metadata(args.get("metadata")))
        if "archived" in args:
            updates["archived"] = 1 if args.get("archived") else 0
        text_changed = any(key in updates for key in ("content", "summary", "title"))
        allow_shared_mutation = bool(self.config.get("allow_any_agent_to_mutate_shared", False))
        if text_changed:
            updated = False
            for _attempt in range(5):
                current = self.store.get_memory(
                    int(memory_id),
                    self.identity.workspace_id,
                    self.identity.agent_id,
                    self.identity.session_id,
                    include_shared=True,
                )
                if not current:
                    return {"success": False, "updated": False}
                text = "\n".join(str(updates.get(key, current.get(key, ""))) for key in ["title", "summary", "content"])
                if self._excluded(text):
                    raise AgentRecallError("Updated content matches an excluded term")
                embedding = self._embedding(text)
                merged_updates = {
                    **updates,
                    "embedding_json": _json_dumps(embedding),
                    "embedding_model": str(self.config.get("embedding_model") or ""),
                    "embedding_dimensions": len(embedding),
                }
                updated = self.store.update_memory(
                    int(memory_id),
                    self.identity.workspace_id,
                    self.identity.agent_id,
                    self.identity.session_id,
                    allow_shared_mutation=allow_shared_mutation,
                    expected_updated_at=float(current["updated_at"]),
                    **merged_updates,
                )
                if updated:
                    break
        else:
            updated = self.store.update_memory(
                int(memory_id),
                self.identity.workspace_id,
                self.identity.agent_id,
                self.identity.session_id,
                allow_shared_mutation=allow_shared_mutation,
                **updates,
            )
        return {"success": updated, "updated": updated}

    def import_markdown(self, args: dict[str, Any]) -> dict[str, Any]:
        original_path = Path(str(args["path"])).expanduser()
        if original_path.is_symlink():
            raise AgentRecallError("agent_recall_import_markdown refuses symlinks")
        path = original_path.resolve()
        if path.suffix.lower() != ".md":
            raise AgentRecallError("agent_recall_import_markdown only imports .md files by default")
        root_values = self.config.get("import_roots") or []
        if not root_values or any(not str(root).strip() for root in root_values):
            raise AgentRecallError("No import_roots are configured; markdown import is disabled")
        roots = [Path(str(root)).expanduser().resolve() for root in root_values]
        if not any(path == root or root in path.parents for root in roots):
            raise AgentRecallError("path is outside configured import_roots")
        max_bytes = int(self.config.get("max_import_bytes") or 2_000_000)
        if path.stat().st_size > max_bytes:
            raise AgentRecallError(f"file exceeds max_import_bytes={max_bytes}")
        text = path.read_text(encoding="utf-8")
        blocked = self._excluded(text)
        if blocked:
            raise AgentRecallError(f"Refusing to import content matching excluded term: {blocked}")
        chunk_chars = max(500, min(int(args.get("chunk_chars") or 3000), 8000))
        ids: list[int] = []
        for offset in range(0, len(text), chunk_chars):
            chunk = text[offset : offset + chunk_chars].strip()
            if not chunk:
                continue
            result = self.remember(
                {
                    "content": chunk,
                    "title": f"{path.name} chunk {len(ids) + 1}",
                    "visibility": args.get("visibility") or self.config.get("default_visibility", "agent"),
                    "category": args.get("category") or "markdown_note",
                    "tags": list(args.get("tags") or []) + ["markdown_import", path.stem],
                    "metadata": {"path": str(path), "chunk_index": len(ids)},
                }
            )
            ids.append(int(result["id"]))
        return {"success": True, "imported": len(ids), "ids": ids}

    def curate(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.config.get("curated_memories_enabled", True):
            raise AgentRecallError(
                "Curated memories are disabled. Set curated_memories_enabled=true to use agent_recall_curate."
            )
        if not self.config.get("llm_curator_enabled", False):
            raise AgentRecallError(
                "AgentRecall chat-model curation is disabled. Set llm_curator_enabled=true to use it."
            )
        text = str(args.get("text") or "")
        blocked = self._excluded(text)
        if blocked:
            raise AgentRecallError(f"Refusing to curate content matching excluded term: {blocked}")
        candidates = self._make_curator().curate(
            text,
            default_visibility=str(args.get("default_visibility") or "agent"),
        )
        if args.get("dry_run"):
            return {"success": True, "stored": 0, "candidates": candidates}
        stored = []
        for item in candidates:
            item.setdefault("visibility", args.get("default_visibility") or "agent")
            stored.append(self.remember(item))
        return {"success": True, "stored": len(stored), "results": stored}

    def conclude(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.config.get("conclusions_enabled", False):
            raise AgentRecallError(
                "Conclusions are disabled. Set conclusions_enabled=true to use agent_recall_conclude."
            )
        source_ids = [int(value) for value in (args.get("source_ids") or [])]
        supersedes = [int(value) for value in (args.get("supersedes") or [])]
        metadata = {
            "module": "conclusions",
            "scope": str(args.get("scope") or "general"),
            "subject": str(args.get("subject") or ""),
            "source_ids": source_ids,
            "supersedes": supersedes,
            "provenance": "agent_recall_conclude",
        }
        tags = ["conclusion", metadata["scope"]]
        if metadata["subject"]:
            tags.append(metadata["subject"])
        return self.remember(
            {
                "content": str(args.get("content") or ""),
                "title": str(args.get("title") or "Conclusion"),
                "summary": str(args.get("summary") or ""),
                "visibility": str(args.get("visibility") or self.config.get("default_visibility") or "agent"),
                "category": "conclusion",
                "tags": tags,
                "metadata": metadata,
                "importance": float(args.get("importance", 0.85)),
                "confidence": float(args.get("confidence", 0.8)),
            }
        )

    def _visible_context(self, focus: str, limit: int = 20) -> list[dict[str, Any]]:
        query = focus or "user preferences project conventions decisions procedures environment"
        return self.search({"query": query, "limit": max(1, min(int(limit or 20), 50))})["results"]

    def profile_synthesize(self, args: dict[str, Any]) -> dict[str, Any]:
        scope = str(args.get("scope") or "peer")
        if scope not in {"peer", "workspace", "agent"}:
            raise AgentRecallError("scope must be one of: peer, workspace, agent")
        toggle = f"{scope}_profiles_enabled"
        if not self.config.get(toggle, False):
            raise AgentRecallError(f"Profile synthesis for scope={scope!r} is disabled. Set {toggle}=true to use it.")
        if not self.config.get("llm_curator_enabled", False):
            raise AgentRecallError(
                "Chat-model curation is disabled. Set llm_curator_enabled=true to synthesize profiles."
            )
        focus = str(args.get("focus") or f"{scope} profile preferences conventions durable facts")
        subject = str(
            args.get("subject")
            or (
                self.identity.agent_id
                if scope == "agent"
                else self.identity.workspace_id
                if scope == "workspace"
                else self.identity.user_id or "peer"
            )
        )
        prompt = (
            f"Synthesize an inspectable {scope} profile for subject={subject!r}. "
            "Use only the visible memories below. Preserve uncertainty and provenance. "
            "Return durable profile facts as memory candidates.\n\n"
            + json.dumps(self._visible_context(focus, limit=20), ensure_ascii=False)[:12_000]
        )
        candidates = self._make_curator().curate(prompt, default_visibility=str(args.get("visibility") or "agent"))
        for item in candidates:
            item.setdefault("category", f"{scope}_profile")
            metadata = normalize_metadata(item.get("metadata"))
            metadata.update(
                {
                    "module": f"{scope}_profiles",
                    "scope": scope,
                    "subject": subject,
                    "source": "agent_recall_profile_synthesize",
                }
            )
            item["metadata"] = metadata
        if args.get("dry_run", True):
            return {"success": True, "stored": 0, "scope": scope, "subject": subject, "candidates": candidates}
        stored = [self.remember(item) for item in candidates]
        return {"success": True, "stored": len(stored), "scope": scope, "subject": subject, "results": stored}

    def review(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.config.get("dialectic_review_enabled", False):
            raise AgentRecallError(
                "Dialectic review is disabled. Set dialectic_review_enabled=true to use agent_recall_review."
            )
        if not self.config.get("llm_curator_enabled", False):
            raise AgentRecallError("Chat-model curation is disabled. Set llm_curator_enabled=true to run reviews.")
        enabled_modules = [
            name
            for name, key in [
                ("conflict_detection", "conflict_detection_enabled"),
                ("staleness_detection", "staleness_detection_enabled"),
                ("promotion_rules", "promotion_rules_enabled"),
                ("demotion_rules", "demotion_rules_enabled"),
            ]
            if self.config.get(key, False)
        ]
        focus = str(args.get("focus") or "memory quality conflicts staleness promotion demotion")
        prompt = (
            "Review these visible AgentRecall memories. Produce durable recommendations only. "
            f"Enabled modules: {', '.join(enabled_modules) or 'dialectic_review_only'}. "
            "Do not mutate memory directly; return recommendations with provenance.\n\n"
            + json.dumps(self._visible_context(focus, limit=int(args.get("limit") or 20)), ensure_ascii=False)[:12_000]
        )
        recommendations = self._make_curator().curate(prompt, default_visibility="agent")
        for item in recommendations:
            item.setdefault("category", "review_recommendation")
            metadata = normalize_metadata(item.get("metadata"))
            metadata.update(
                {
                    "module": "dialectic_review",
                    "enabled_modules": enabled_modules,
                    "source": "agent_recall_review",
                }
            )
            item["metadata"] = metadata
        if args.get("dry_run", True):
            return {
                "success": True,
                "stored": 0,
                "enabled_modules": enabled_modules,
                "recommendations": recommendations,
            }
        stored = [self.remember(item) for item in recommendations]
        return {"success": True, "stored": len(stored), "enabled_modules": enabled_modules, "results": stored}

    def capture_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        background: bool = False,
    ) -> bool:
        if not self.config.get("auto_capture_turns", False):
            return False
        text = f"User: {user_content}\nAssistant: {assistant_content}"
        if self._excluded(text):
            return False
        capture_identity = AgentIdentity(
            self.identity.workspace_id,
            self.identity.agent_id,
            session_id or self.identity.session_id,
            self.identity.user_id,
        )

        def write() -> None:
            with suppress(Exception):
                self.remember(
                    {
                        "content": text,
                        "visibility": self.config.get("auto_capture_visibility", "session"),
                        "category": "transcript",
                        "tags": ["auto_capture"],
                        "metadata": {"session_id": capture_identity.session_id},
                        "importance": 0.25,
                        "confidence": 0.6,
                    },
                    identity=capture_identity,
                )

        if background:
            thread = threading.Thread(target=write, daemon=True)
            self._sync_threads = [item for item in self._sync_threads if item.is_alive()]
            self._sync_threads.append(thread)
            thread.start()
        else:
            write()
        return True

    def checkpoint_from_messages(self, messages: list[dict[str, Any]]) -> int | None:
        if not self.config.get("auto_capture_compression_checkpoints", False) or not messages:
            return None
        text_parts = []
        for message in messages[-20:]:
            role = message.get("role", "")
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                text_parts.append(f"{role}: {content[:1000]}")
        text = "\n".join(text_parts)
        if len(text) < 40 or self._excluded(text):
            return None
        result = self.remember(
            {
                "content": text,
                "visibility": "session",
                "category": "compression_checkpoint",
                "tags": ["pre_compress"],
                "importance": 0.35,
                "confidence": 0.5,
            }
        )
        return int(result["id"])

    def mirror_builtin_memory(self, action: str, target: str, content: str, metadata: Any = None) -> bool:
        if action != "add" or not content:
            return False
        self.remember(
            {
                "content": content,
                "visibility": "shared" if target == "user" else "agent",
                "category": "user_pref" if target == "user" else "builtin_memory",
                "tags": ["mirrored_builtin_memory", target],
                "metadata": metadata or {},
                "importance": 0.75,
                "confidence": 0.9,
            }
        )
        return True

    def capabilities(self) -> dict[str, Any]:
        return {
            "success": True,
            "engine": "AgentRecall",
            "native_host_required": False,
            "workspace_id": self.identity.workspace_id,
            "agent_id": self.identity.agent_id,
            "operations": list(self.OPERATIONS),
            "visibility": ["agent", "shared", "session"],
            "features": {
                "semantic_search": True,
                "lexical_fallback": True,
                "score_explanations": True,
                "context_budgets": True,
                "curation": bool(self.config.get("curated_memories_enabled", True)),
                "conclusions": bool(self.config.get("conclusions_enabled", False)),
                "profile_synthesis": any(
                    self.config.get(key, False)
                    for key in ("peer_profiles_enabled", "workspace_profiles_enabled", "agent_profiles_enabled")
                ),
                "review": bool(self.config.get("dialectic_review_enabled", False)),
            },
        }

    def health(self) -> dict[str, Any]:
        sqlite_status = self.store.health()
        return {
            "success": sqlite_status["quick_check"] == "ok",
            "identity": self.identity.as_dict(),
            "sqlite": sqlite_status,
            "embedding": {
                "configured": bool(self.config.get("embedding_base_url")),
                "enabled": bool(self.config.get("embedding_base_url")),
                "model": str(self.config.get("embedding_model") or ""),
                "last_error": self.last_embedding_error,
            },
        }

    def close(self) -> None:
        for thread in list(self._sync_threads):
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join()
        self._sync_threads = []
        self.store.close()
