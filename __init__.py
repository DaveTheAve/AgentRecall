from __future__ import annotations

import json
import os
import re
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider

try:
    from tools.registry import tool_error
except Exception:
    def tool_error(message: str) -> str:
        return json.dumps({"success": False, "error": message})

try:
    from .agent_recall_curator import ChatCompletionsCurator, CodexCliCurator
    from .agent_recall_store import AgentRecallStore, EmbeddingClient, _json_dumps, normalize_text
except ImportError:  # Allows pytest/direct import from the plugin directory.
    from agent_recall_curator import ChatCompletionsCurator, CodexCliCurator
    from agent_recall_store import AgentRecallStore, EmbeddingClient, _json_dumps, normalize_text


PLUGIN_NAME = "agent-recall"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except Exception:
        return default


def _default_config(hermes_home: str | Path) -> dict:
    home = Path(hermes_home).expanduser()
    return {
        "db_path": str(home / "agent-recall.db"),
        "workspace_id": os.environ.get("AGENT_RECALL_WORKSPACE", "hermes"),
        "agent_id": os.environ.get("AGENT_RECALL_AGENT", ""),
        "embedding_base_url": os.environ.get("AGENT_RECALL_EMBEDDING_BASE_URL", "http://127.0.0.1:6660/v1"),
        "embedding_model": os.environ.get("AGENT_RECALL_EMBEDDING_MODEL", "qwen3-embedding-4b"),
        "embedding_api_key_env": os.environ.get("AGENT_RECALL_EMBEDDING_API_KEY_ENV", "LLM_OPENAI_API_KEY"),
        "embedding_dimensions": _env_int("AGENT_RECALL_EMBEDDING_DIMENSIONS", 0),
        "embedding_timeout": 20.0,
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
        "max_memory_chars": 12000,
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


def _load_config(hermes_home: str | Path) -> dict:
    cfg = _default_config(hermes_home)
    path = Path(hermes_home).expanduser() / "agent-recall.json"
    cfg.update(_load_json(path))
    return cfg


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


def _normalize_config(cfg: dict) -> dict:
    out = dict(cfg)
    for key, default in {
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
    }.items():
        out[key] = _as_bool(out.get(key), default)
    for key in ["embedding_dimensions", "prefetch_limit", "max_memory_chars", "max_import_bytes"]:
        if key in out:
            try:
                out[key] = int(out[key] or 0)
            except Exception:
                out[key] = 0
    for key in ["embedding_timeout", "llm_curator_timeout"]:
        try:
            out[key] = float(out.get(key) or (120.0 if key == "llm_curator_timeout" else 20.0))
        except Exception:
            out[key] = 120.0 if key == "llm_curator_timeout" else 20.0
    return out


def _normalize_tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("tags must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("tags must be a list of strings")
        text = item.strip()
        if text:
            out.append(text)
    return out


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    return value


REMEMBER_SCHEMA = {
    "name": "agent_recall_remember",
    "description": "Store a durable memory in AgentRecall. Choose visibility='shared' only for facts useful across agents; use 'agent' for profile-specific facts and 'session' for temporary session recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The factual memory content to store."},
            "title": {"type": "string", "description": "Optional short title."},
            "summary": {"type": "string", "description": "Optional concise summary."},
            "visibility": {"type": "string", "enum": ["agent", "shared", "session"], "description": "Isolation scope."},
            "category": {"type": "string", "description": "Category such as user_pref, project, environment, decision, procedure, transcript."},
            "tags": {"type": "array", "items": {"type": "string"}},
            "importance": {"type": "number", "description": "0.0-1.0 importance."},
            "confidence": {"type": "number", "description": "0.0-1.0 confidence."},
            "metadata": {"type": "object", "description": "Optional JSON metadata."}
        },
        "required": ["content"]
    }
}

SEARCH_SCHEMA = {
    "name": "agent_recall_search",
    "description": "Search visible AgentRecall memories with hybrid embedding + lexical retrieval. Enforces workspace/agent/session ACL isolation.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 8},
            "include_shared": {"type": "boolean", "default": True},
            "category": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["query"]
    }
}

PROFILE_SCHEMA = {
    "name": "agent_recall_profile",
    "description": "Return a compact view of current agent, workspace, isolation settings, and top memories relevant to a focus query.",
    "parameters": {
        "type": "object",
        "properties": {"focus": {"type": "string"}, "limit": {"type": "integer", "default": 10}}
    }
}

UPDATE_SCHEMA = {
    "name": "agent_recall_update",
    "description": "Update, promote/demote, retag, or archive a visible AgentRecall memory. ACL rules prevent editing another agent's private memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "content": {"type": "string"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "visibility": {"type": "string", "enum": ["agent", "shared", "session"]},
            "category": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "importance": {"type": "number"},
            "confidence": {"type": "number"},
            "archived": {"type": "boolean"}
        },
        "required": ["id"]
    }
}

FORGET_SCHEMA = {
    "name": "agent_recall_forget",
    "description": "Delete a visible AgentRecall memory by id, respecting isolation ACLs.",
    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}
}

CURATE_SCHEMA = {
    "name": "agent_recall_curate",
    "description": "Use the configured chat-model curation backend to extract durable memory candidates from text, then store selected candidates in AgentRecall. Set llm_curator_enabled=false to turn this off.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "default_visibility": {"type": "string", "enum": ["agent", "shared", "session"], "default": "agent"},
            "dry_run": {"type": "boolean", "default": False}
        },
        "required": ["text"]
    }
}


STATS_SCHEMA = {
    "name": "agent_recall_stats",
    "description": "Show AgentRecall database counts by workspace/agent/visibility/category.",
    "parameters": {"type": "object", "properties": {}}
}

IMPORT_MD_SCHEMA = {
    "name": "agent_recall_import_markdown",
    "description": "Ingest a markdown note/file into AgentRecall as memory chunks. Use for Obsidian/Hermes vault notes; respects excluded terms.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "visibility": {"type": "string", "enum": ["agent", "shared", "session"]},
            "category": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "chunk_chars": {"type": "integer", "default": 3000}
        },
        "required": ["path"]
    }
}

CONCLUDE_SCHEMA = {
    "name": "agent_recall_conclude",
    "description": "Store an inspectable conclusion with provenance. Controlled by conclusions_enabled.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "scope": {"type": "string", "enum": ["peer", "workspace", "agent", "general"], "default": "general"},
            "subject": {"type": "string"},
            "source_ids": {"type": "array", "items": {"type": "integer"}},
            "visibility": {"type": "string", "enum": ["agent", "shared", "session"], "default": "agent"},
            "confidence": {"type": "number", "default": 0.8},
            "supersedes": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["content"]
    }
}

PROFILE_SYNTH_SCHEMA = {
    "name": "agent_recall_profile_synthesize",
    "description": "Synthesize peer/workspace/agent profiles using the configured chat-model backend. Each scope has its own enable toggle.",
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {"type": "string", "enum": ["peer", "workspace", "agent"], "default": "peer"},
            "subject": {"type": "string"},
            "focus": {"type": "string"},
            "dry_run": {"type": "boolean", "default": True},
            "visibility": {"type": "string", "enum": ["agent", "shared", "session"], "default": "agent"},
        }
    }
}

REVIEW_SCHEMA = {
    "name": "agent_recall_review",
    "description": "Run a bounded dialectic review for conflicts, staleness, and promotion/demotion recommendations using the configured chat-model backend.",
    "parameters": {
        "type": "object",
        "properties": {
            "focus": {"type": "string"},
            "dry_run": {"type": "boolean", "default": True},
            "limit": {"type": "integer", "default": 20},
        }
    }
}


class AgentRecallProvider(MemoryProvider):
    def __init__(self, config: dict | None = None):
        self._initial_config = config or {}
        self._config: dict = {}
        self._store: AgentRecallStore | None = None
        self._embedder: EmbeddingClient | None = None
        self._session_id = ""
        self._workspace_id = "hermes"
        self._agent_id = "hermes"
        self._user_id = ""
        self._sync_threads: list[threading.Thread] = []
        self._last_prefetch = ""
        self._last_embedding_error = ""

    @property
    def name(self) -> str:
        return PLUGIN_NAME

    def is_available(self) -> bool:
        return True

    def get_config_schema(self):
        return [
            {"key": "db_path", "description": "SQLite DB path. Use a shared path for cross-agent shared memory.", "default": "$HERMES_HOME/agent-recall.db"},
            {"key": "workspace_id", "description": "Shared workspace id", "default": "hermes"},
            {"key": "agent_id", "description": "Agent identity for private memory isolation", "default": ""},
            {"key": "embedding_base_url", "description": "OpenAI-compatible embeddings base URL", "default": "http://127.0.0.1:6660/v1"},
            {"key": "embedding_model", "description": "Embedding model name", "default": "qwen3-embedding-4b"},
            {"key": "embedding_api_key_env", "description": "Env var containing embedding API key", "default": "LLM_OPENAI_API_KEY"},
            {"key": "embedding_dimensions", "description": "Embedding dimensions; 0 omits dimensions parameter", "default": "0"},
            {"key": "raw_memories_enabled", "description": "Enable explicit agent_recall_remember storage", "default": "true"},
            {"key": "curated_memories_enabled", "description": "Enable agent_recall_curate feature layer", "default": "true"},
            {"key": "llm_curator_enabled", "description": "Enable chat-model memory curation and synthesis calls", "default": "true"},
            {"key": "llm_curator_backend", "description": "Curation backend: codex-cli or openai-compatible", "default": "codex-cli"},
            {"key": "llm_curator_model", "description": "Chat model used by curation backend", "default": "gpt-5.3-mini"},
            {"key": "llm_curator_command", "description": "CLI command for codex-cli backend", "default": "codex"},
            {"key": "llm_curator_base_url", "description": "OpenAI-compatible chat base URL when using openai-compatible backend", "default": ""},
            {"key": "llm_curator_api_key_env", "description": "Env var containing chat API key for openai-compatible backend", "default": "OPENAI_API_KEY"},
            {"key": "conclusions_enabled", "description": "Enable inspectable conclusion storage", "default": "false"},
            {"key": "peer_profiles_enabled", "description": "Enable peer profile synthesis", "default": "false"},
            {"key": "workspace_profiles_enabled", "description": "Enable workspace profile synthesis", "default": "false"},
            {"key": "agent_profiles_enabled", "description": "Enable agent profile synthesis", "default": "false"},
            {"key": "dialectic_review_enabled", "description": "Enable bounded dialectic review jobs", "default": "false"},
            {"key": "conflict_detection_enabled", "description": "Enable conflict-detection instructions in review jobs", "default": "false"},
            {"key": "staleness_detection_enabled", "description": "Enable stale-memory detection instructions in review jobs", "default": "false"},
            {"key": "promotion_rules_enabled", "description": "Enable promotion recommendations in review jobs", "default": "false"},
            {"key": "demotion_rules_enabled", "description": "Enable demotion/archive recommendations in review jobs", "default": "false"},
            {"key": "auto_capture_turns", "description": "Opt-in completed-turn transcript capture", "default": "false"},
            {"key": "auto_capture_compression_checkpoints", "description": "Opt-in pre-compression checkpoint capture", "default": "false"},
            {"key": "allow_any_agent_to_mutate_shared", "description": "Allow any workspace agent to edit/delete shared rows", "default": "false"},
        ]

    def save_config(self, values, hermes_home):
        path = Path(hermes_home).expanduser() / "agent-recall.json"
        cfg = _default_config(hermes_home)
        clean = dict(values or {})
        db_path = str(clean.get("db_path") or cfg["db_path"]).replace("$HERMES_HOME", str(Path(hermes_home).expanduser()))
        clean["db_path"] = db_path
        for k, v in clean.items():
            if k == "embedding_dimensions":
                try:
                    v = int(v)
                except Exception:
                    v = 0
            cfg[k] = v
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        with suppress(Exception):
            path.chmod(0o600)

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        cfg = _load_config(hermes_home)
        cfg.update(self._initial_config)
        cfg["db_path"] = str(cfg.get("db_path", "")).replace("$HERMES_HOME", str(Path(hermes_home).expanduser()))
        cfg["db_path"] = str(cfg.get("db_path", "")).replace("${HERMES_HOME}", str(Path(hermes_home).expanduser()))
        self._config = _normalize_config(cfg)
        cfg = self._config
        self._session_id = session_id or ""
        self._workspace_id = str(cfg.get("workspace_id") or kwargs.get("agent_workspace") or "hermes")
        self._agent_id = str(cfg.get("agent_id") or kwargs.get("agent_identity") or Path(hermes_home).name or "hermes")
        if self._agent_id == ".hermes":
            self._agent_id = "hermes"
        self._user_id = str(kwargs.get("user_id") or kwargs.get("user_id_alt") or "")
        self._store = AgentRecallStore(cfg["db_path"])
        api_key = os.environ.get(str(cfg.get("embedding_api_key_env") or ""), "")
        self._embedder = EmbeddingClient(
            str(cfg.get("embedding_base_url") or ""),
            str(cfg.get("embedding_model") or ""),
            api_key=api_key,
            dimensions=int(cfg.get("embedding_dimensions") or 0),
            timeout=float(cfg.get("embedding_timeout") or 20.0),
        )

    def system_prompt_block(self) -> str:
        return (
            "# AgentRecall Memory\n"
            f"Active durable memory. workspace={self._workspace_id!r}, agent={self._agent_id!r}.\n"
            "Isolation: automatic recall includes shared memories in this workspace plus this agent's own private/session memories only.\n"
            "Use agent_recall_remember for explicit facts and agent_recall_curate when text should be distilled into durable memory candidates. "
            "Choose visibility='shared' only for cross-agent facts."
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            REMEMBER_SCHEMA, SEARCH_SCHEMA, PROFILE_SCHEMA, UPDATE_SCHEMA, FORGET_SCHEMA,
            CURATE_SCHEMA, CONCLUDE_SCHEMA, PROFILE_SYNTH_SCHEMA, REVIEW_SCHEMA,
            STATS_SCHEMA, IMPORT_MD_SCHEMA,
        ]

    def _excluded(self, text: str) -> str | None:
        for term in self._config.get("excluded_terms", []) or []:
            if term and re.search(re.escape(str(term)), text or "", re.IGNORECASE):
                return str(term)
        return None

    def _embedding(self, text: str) -> list[float]:
        self._last_embedding_error = ""
        if not self._embedder:
            return []
        try:
            return self._embedder.embed(text)
        except Exception as exc:
            # Retrieval remains useful with lexical scoring, and explicit memory
            # writes should not fail just because the embedding endpoint is down.
            self._last_embedding_error = str(exc)[-500:]
            return []

    def _make_curator(self):
        backend = str(self._config.get("llm_curator_backend") or "codex-cli")
        if backend == "codex-cli":
            return CodexCliCurator(
                command=str(self._config.get("llm_curator_command") or "codex"),
                model=str(self._config.get("llm_curator_model") or "gpt-5.3-mini"),
                timeout=float(self._config.get("llm_curator_timeout") or 120.0),
            )
        if backend in {"openai-compatible", "chat-completions"}:
            api_key = os.environ.get(str(self._config.get("llm_curator_api_key_env") or ""), "")
            return ChatCompletionsCurator(
                base_url=str(self._config.get("llm_curator_base_url") or ""),
                model=str(self._config.get("llm_curator_model") or "gpt-5.3-mini"),
                api_key=api_key,
                timeout=float(self._config.get("llm_curator_timeout") or 120.0),
            )
        raise ValueError("Unsupported llm_curator_backend; use codex-cli or openai-compatible")

    def _visible_context(self, focus: str = "", limit: int = 20) -> list[dict[str, Any]]:
        if not self._store:
            return []
        query = focus or "user preferences project conventions decisions procedures environment"
        emb = self._embedding(query) if query else []
        return self._store.search(
            workspace_id=self._workspace_id,
            agent_id=self._agent_id,
            session_id=self._session_id,
            query=query,
            query_embedding=emb,
            include_shared=bool(self._config.get("shared_recall", True)),
            limit=max(1, min(int(limit or 20), 50)),
        )

    def _remember(self, args: dict[str, Any]) -> str:
        if not self._store:
            return tool_error("AgentRecall is not initialized")
        if not self._config.get("raw_memories_enabled", True):
            return tool_error("Raw memory storage is disabled by raw_memories_enabled=false")
        content = normalize_text(str(args.get("content") or ""), int(self._config.get("max_memory_chars") or 12000))
        blocked = self._excluded(content)
        if blocked:
            return tool_error(f"Refusing to store content matching excluded term: {blocked}")
        emb = self._embedding("\n".join([str(args.get("title") or ""), str(args.get("summary") or ""), content]))
        mem_id = self._store.add_memory(
            workspace_id=self._workspace_id,
            agent_id=self._agent_id,
            source_agent_id=self._agent_id,
            user_id=self._user_id,
            session_id=self._session_id,
            visibility=str(args.get("visibility") or self._config.get("default_visibility") or "agent"),
            category=str(args.get("category") or "general"),
            title=str(args.get("title") or ""),
            content=content,
            summary=str(args.get("summary") or ""),
            tags=_normalize_tags(args.get("tags")),
            metadata=_normalize_metadata(args.get("metadata")),
            embedding=emb,
            embedding_model=str(self._config.get("embedding_model") or ""),
            importance=float(args.get("importance", 0.5)),
            confidence=float(args.get("confidence", 0.8)),
        )
        payload = {"success": True, "id": mem_id, "visibility": args.get("visibility") or self._config.get("default_visibility", "agent")}
        if self._last_embedding_error:
            payload["embedding_warning"] = self._last_embedding_error
        return json.dumps(payload)

    def _search(self, args: dict[str, Any]) -> str:
        if not self._store:
            return tool_error("AgentRecall is not initialized")
        query = str(args.get("query") or "")
        emb = self._embedding(query) if query else []
        results = self._store.search(
            workspace_id=self._workspace_id,
            agent_id=self._agent_id,
            session_id=self._session_id,
            query=query,
            query_embedding=emb,
            include_shared=bool(args.get("include_shared", self._config.get("shared_recall", True))),
            category=str(args.get("category") or ""),
            tags=args.get("tags") or [],
            limit=int(args.get("limit") or 8),
        )
        payload = {"success": True, "results": results, "count": len(results)}
        if self._last_embedding_error:
            payload["embedding_warning"] = self._last_embedding_error
        return json.dumps(payload, ensure_ascii=False)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not self._store:
            return ""
        try:
            emb = self._embedding(query)
            results = self._store.search(
                workspace_id=self._workspace_id,
                agent_id=self._agent_id,
                session_id=session_id or self._session_id,
                query=query,
                query_embedding=emb,
                include_shared=bool(self._config.get("shared_recall", True)),
                limit=int(self._config.get("prefetch_limit") or 6),
            )
            if not results:
                return ""
            lines = ["# AgentRecall Recalled Context"]
            for r in results:
                scope = f"{r['visibility']}:{r['agent_id']}"
                title = f"{r['title']} — " if r.get("title") else ""
                lines.append(f"- [{r['id']} {scope} score={r['score']}] {title}{r['content'][:700]}")
            self._last_prefetch = "\n".join(lines)
            return self._last_prefetch
        except Exception:
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Synchronous prefetch is simpler and bounded by embedding_timeout; leave no background state races.
        return None

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages=None) -> None:
        if not self._config.get("auto_capture_turns", False):
            return
        text = f"User: {user_content}\nAssistant: {assistant_content}"
        if self._excluded(text):
            return
        def _run():
            with suppress(Exception):
                self._remember({
                    "content": text,
                    "visibility": self._config.get("auto_capture_visibility", "session"),
                    "category": "transcript",
                    "tags": ["auto_capture"],
                    "metadata": {"session_id": session_id or self._session_id},
                    "importance": 0.25,
                    "confidence": 0.6,
                })
        thread = threading.Thread(target=_run, daemon=True)
        self._sync_threads = [t for t in self._sync_threads if t.is_alive()]
        self._sync_threads.append(thread)
        thread.start()

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        if not self._config.get("auto_capture_compression_checkpoints", False):
            return ""
        if not self._store or not messages:
            return ""
        # Store a compact compression checkpoint when explicitly configured.
        text_parts = []
        for msg in messages[-20:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                text_parts.append(f"{role}: {content[:1000]}")
        text = "\n".join(text_parts)
        if len(text) < 40 or self._excluded(text):
            return ""
        try:
            result = json.loads(self._remember({
                "content": text,
                "visibility": "session",
                "category": "compression_checkpoint",
                "tags": ["pre_compress"],
                "importance": 0.35,
                "confidence": 0.5,
            }))
            return f"AgentRecall saved a compression checkpoint id={result.get('id')} for later recall."
        except Exception:
            return ""

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        if action != "add" or not content:
            return
        with suppress(Exception):
            self._remember({
                "content": content,
                "visibility": "shared" if target == "user" else "agent",
                "category": "user_pref" if target == "user" else "builtin_memory",
                "tags": ["mirrored_builtin_memory", target],
                "metadata": metadata or {},
                "importance": 0.75,
                "confidence": 0.9,
            })

    def _conclude(self, args: dict[str, Any]) -> str:
        if not self._config.get("conclusions_enabled", False):
            return tool_error("Conclusions are disabled. Set conclusions_enabled=true to use agent_recall_conclude.")
        source_ids = args.get("source_ids") or []
        supersedes = args.get("supersedes") or []
        metadata = {
            "module": "conclusions",
            "scope": str(args.get("scope") or "general"),
            "subject": str(args.get("subject") or ""),
            "source_ids": [int(x) for x in source_ids],
            "supersedes": [int(x) for x in supersedes],
            "provenance": "agent_recall_conclude",
        }
        return self._remember({
            "content": str(args.get("content") or ""),
            "title": str(args.get("title") or "Conclusion"),
            "summary": str(args.get("summary") or ""),
            "visibility": str(args.get("visibility") or self._config.get("default_visibility") or "agent"),
            "category": "conclusion",
            "tags": ["conclusion", metadata["scope"], metadata["subject"]] if metadata["subject"] else ["conclusion", metadata["scope"]],
            "metadata": metadata,
            "importance": float(args.get("importance", 0.85)),
            "confidence": float(args.get("confidence", 0.8)),
        })

    def _profile_synthesize(self, args: dict[str, Any]) -> str:
        scope = str(args.get("scope") or "peer")
        toggle = f"{scope}_profiles_enabled"
        if scope not in {"peer", "workspace", "agent"}:
            return tool_error("scope must be one of: peer, workspace, agent")
        if not self._config.get(toggle, False):
            return tool_error(f"Profile synthesis for scope={scope!r} is disabled. Set {toggle}=true to use it.")
        if not self._config.get("llm_curator_enabled", False):
            return tool_error("Chat-model curation is disabled. Set llm_curator_enabled=true to synthesize profiles.")
        focus = str(args.get("focus") or f"{scope} profile preferences conventions durable facts")
        subject = str(args.get("subject") or (self._agent_id if scope == "agent" else self._workspace_id if scope == "workspace" else self._user_id or "peer"))
        context = self._visible_context(focus, limit=20)
        prompt = (
            f"Synthesize an inspectable {scope} profile for subject={subject!r}. "
            "Use only the visible memories below. Preserve uncertainty and provenance. "
            "Return durable profile facts as memory candidates.\n\n"
            + json.dumps(context, ensure_ascii=False)[:12000]
        )
        candidates = self._make_curator().curate(prompt, default_visibility=str(args.get("visibility") or "agent"))
        for item in candidates:
            item.setdefault("category", f"{scope}_profile")
            meta = _normalize_metadata(item.get("metadata"))
            meta.update({"module": f"{scope}_profiles", "scope": scope, "subject": subject, "source": "agent_recall_profile_synthesize"})
            item["metadata"] = meta
        if args.get("dry_run", True):
            return json.dumps({"success": True, "stored": 0, "scope": scope, "subject": subject, "candidates": candidates}, ensure_ascii=False)
        stored = [json.loads(self._remember(item)) for item in candidates]
        return json.dumps({"success": True, "stored": len(stored), "scope": scope, "subject": subject, "results": stored}, ensure_ascii=False)

    def _review(self, args: dict[str, Any]) -> str:
        if not self._config.get("dialectic_review_enabled", False):
            return tool_error("Dialectic review is disabled. Set dialectic_review_enabled=true to use agent_recall_review.")
        if not self._config.get("llm_curator_enabled", False):
            return tool_error("Chat-model curation is disabled. Set llm_curator_enabled=true to run reviews.")
        enabled_modules = [
            name for name, key in [
                ("conflict_detection", "conflict_detection_enabled"),
                ("staleness_detection", "staleness_detection_enabled"),
                ("promotion_rules", "promotion_rules_enabled"),
                ("demotion_rules", "demotion_rules_enabled"),
            ] if self._config.get(key, False)
        ]
        focus = str(args.get("focus") or "memory quality conflicts staleness promotion demotion")
        context = self._visible_context(focus, limit=int(args.get("limit") or 20))
        prompt = (
            "Review these visible AgentRecall memories. Produce durable recommendations only. "
            f"Enabled modules: {', '.join(enabled_modules) or 'dialectic_review_only'}. "
            "Do not mutate memory directly; return recommendations with provenance.\n\n"
            + json.dumps(context, ensure_ascii=False)[:12000]
        )
        recommendations = self._make_curator().curate(prompt, default_visibility="agent")
        for item in recommendations:
            item.setdefault("category", "review_recommendation")
            meta = _normalize_metadata(item.get("metadata"))
            meta.update({"module": "dialectic_review", "enabled_modules": enabled_modules, "source": "agent_recall_review"})
            item["metadata"] = meta
        if args.get("dry_run", True):
            return json.dumps({"success": True, "stored": 0, "enabled_modules": enabled_modules, "recommendations": recommendations}, ensure_ascii=False)
        stored = [json.loads(self._remember(item)) for item in recommendations]
        return json.dumps({"success": True, "stored": len(stored), "enabled_modules": enabled_modules, "results": stored}, ensure_ascii=False)

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "agent_recall_remember":
                return self._remember(args)
            if tool_name == "agent_recall_search":
                return self._search(args)
            if tool_name == "agent_recall_profile":
                focus = str(args.get("focus") or "user preferences project conventions agent memory")
                payload = json.loads(self._search({"query": focus, "limit": int(args.get("limit") or 10)}))
                return json.dumps({"success": True, "workspace_id": self._workspace_id, "agent_id": self._agent_id, "db_path": self._config.get("db_path"), "recall": payload.get("results", [])}, ensure_ascii=False)
            if tool_name == "agent_recall_stats":
                return json.dumps({"success": True, "stats": self._store.stats(self._workspace_id, self._agent_id, self._session_id) if self._store else {}}, ensure_ascii=False)
            if tool_name == "agent_recall_forget":
                ok = self._store.delete_memory(
                    int(args["id"]),
                    self._workspace_id,
                    self._agent_id,
                    self._session_id,
                    allow_shared_mutation=bool(self._config.get("allow_any_agent_to_mutate_shared", False)),
                ) if self._store else False
                return json.dumps({"success": ok, "deleted": ok})
            if tool_name == "agent_recall_review":
                return self._review(args)
            if tool_name == "agent_recall_profile_synthesize":
                return self._profile_synthesize(args)
            if tool_name == "agent_recall_conclude":
                return self._conclude(args)
            if tool_name == "agent_recall_curate":
                if not self._config.get("curated_memories_enabled", True):
                    return tool_error("Curated memories are disabled. Set curated_memories_enabled=true to use agent_recall_curate.")
                if not self._config.get("llm_curator_enabled", False):
                    return tool_error("AgentRecall chat-model curation is disabled. Set llm_curator_enabled=true to use it.")
                text = str(args.get("text") or "")
                blocked = self._excluded(text)
                if blocked:
                    return tool_error(f"Refusing to curate content matching excluded term: {blocked}")
                curator = self._make_curator()
                candidates = curator.curate(text, default_visibility=str(args.get("default_visibility") or "agent"))
                if args.get("dry_run"):
                    return json.dumps({"success": True, "stored": 0, "candidates": candidates}, ensure_ascii=False)
                stored = []
                for item in candidates:
                    item.setdefault("visibility", args.get("default_visibility") or "agent")
                    stored.append(json.loads(self._remember(item)))
                return json.dumps({"success": True, "stored": len(stored), "results": stored}, ensure_ascii=False)
            if tool_name == "agent_recall_update":
                mem_id = int(args["id"])
                updates: dict[str, Any] = {}
                for k in ["content", "title", "summary", "category", "visibility", "importance", "confidence"]:
                    if k in args:
                        updates[k] = normalize_text(str(args[k])) if k == "content" else args[k]
                if "tags" in args:
                    updates["tags_json"] = _json_dumps(args.get("tags") or [])
                if "archived" in args:
                    updates["archived"] = 1 if args.get("archived") else 0
                if "content" in updates or "summary" in updates or "title" in updates:
                    text = "\n".join(str(updates.get(k, "")) for k in ["title", "summary", "content"])
                    if self._excluded(text):
                        return tool_error("Updated content matches an excluded term")
                    emb = self._embedding(text)
                    updates["embedding_json"] = _json_dumps(emb)
                    updates["embedding_model"] = str(self._config.get("embedding_model") or "")
                    updates["embedding_dimensions"] = len(emb)
                ok = self._store.update_memory(
                    mem_id,
                    self._workspace_id,
                    self._agent_id,
                    self._session_id,
                    allow_shared_mutation=bool(self._config.get("allow_any_agent_to_mutate_shared", False)),
                    **updates,
                ) if self._store else False
                return json.dumps({"success": ok, "updated": ok})
            if tool_name == "agent_recall_import_markdown":
                original_path = Path(str(args["path"])).expanduser()
                if original_path.is_symlink():
                    return tool_error("agent_recall_import_markdown refuses symlinks")
                path = original_path.resolve()
                if path.suffix.lower() != ".md":
                    return tool_error("agent_recall_import_markdown only imports .md files by default")
                roots = [Path(str(root)).expanduser().resolve() for root in (self._config.get("import_roots") or [])]
                if roots and not any(path == root or root in path.parents for root in roots):
                    return tool_error("path is outside configured import_roots")
                max_bytes = int(self._config.get("max_import_bytes") or 2_000_000)
                if path.stat().st_size > max_bytes:
                    return tool_error(f"file exceeds max_import_bytes={max_bytes}")
                text = path.read_text(encoding="utf-8")
                blocked = self._excluded(text)
                if blocked:
                    return tool_error(f"Refusing to import content matching excluded term: {blocked}")
                chunk_chars = max(500, min(int(args.get("chunk_chars") or 3000), 8000))
                ids = []
                for i in range(0, len(text), chunk_chars):
                    chunk = text[i:i+chunk_chars].strip()
                    if not chunk:
                        continue
                    res = json.loads(self._remember({
                        "content": chunk,
                        "title": f"{path.name} chunk {len(ids)+1}",
                        "visibility": args.get("visibility") or self._config.get("default_visibility", "agent"),
                        "category": args.get("category") or "markdown_note",
                        "tags": list(args.get("tags") or []) + ["markdown_import", path.stem],
                        "metadata": {"path": str(path), "chunk_index": len(ids)},
                    }))
                    if not res.get("success"):
                        return json.dumps({"success": False, "error": res.get("error", "failed to store markdown chunk"), "imported": len(ids), "ids": ids})
                    ids.append(res.get("id"))
                return json.dumps({"success": True, "imported": len(ids), "ids": ids})
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))
        return tool_error(f"Unknown AgentRecall tool: {tool_name}")

    def shutdown(self) -> None:
        for thread in list(self._sync_threads):
            if thread.is_alive():
                thread.join(timeout=2.0)
        self._sync_threads = []
        if self._store:
            self._store.close()
        self._store = None


def register(ctx) -> None:
    ctx.register_memory_provider(AgentRecallProvider())
