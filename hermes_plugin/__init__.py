from __future__ import annotations

import json
import os
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
    from ..agent_recall_core import (
        AgentIdentity,
        AgentRecallCore,
        AgentRecallError,
        build_curator,
        default_config,
        load_config,
        normalize_config,
        normalize_metadata,
        normalize_tags,
    )
    from ..agent_recall_curator import ChatCompletionsCurator, CodexCliCurator
    from ..agent_recall_schemas import (
        CONCLUDE_SCHEMA,
        CURATE_SCHEMA,
        FORGET_SCHEMA,
        HERMES_SCHEMAS,
        IMPORT_MD_SCHEMA,
        PROFILE_SCHEMA,
        PROFILE_SYNTH_SCHEMA,
        REMEMBER_SCHEMA,
        REVIEW_SCHEMA,
        SEARCH_SCHEMA,
        STATS_SCHEMA,
        UPDATE_SCHEMA,
    )
    from ..agent_recall_store import AgentRecallStore, EmbeddingClient, _json_dumps, normalize_text
except ImportError:
    from agent_recall_core import (
        AgentIdentity,
        AgentRecallCore,
        AgentRecallError,
        build_curator,
        default_config,
        load_config,
        normalize_config,
        normalize_metadata,
        normalize_tags,
    )
    from agent_recall_curator import ChatCompletionsCurator, CodexCliCurator
    from agent_recall_schemas import (
        CONCLUDE_SCHEMA,
        CURATE_SCHEMA,
        FORGET_SCHEMA,
        HERMES_SCHEMAS,
        IMPORT_MD_SCHEMA,
        PROFILE_SCHEMA,
        PROFILE_SYNTH_SCHEMA,
        REMEMBER_SCHEMA,
        REVIEW_SCHEMA,
        SEARCH_SCHEMA,
        STATS_SCHEMA,
        UPDATE_SCHEMA,
    )
    from agent_recall_store import AgentRecallStore, EmbeddingClient, _json_dumps, normalize_text

PLUGIN_NAME = "agent-recall"

__all__ = [
    "AgentRecallProvider",
    "AgentRecallStore",
    "EmbeddingClient",
    "REMEMBER_SCHEMA",
    "SEARCH_SCHEMA",
    "PROFILE_SCHEMA",
    "UPDATE_SCHEMA",
    "FORGET_SCHEMA",
    "CURATE_SCHEMA",
    "CONCLUDE_SCHEMA",
    "PROFILE_SYNTH_SCHEMA",
    "REVIEW_SCHEMA",
    "STATS_SCHEMA",
    "IMPORT_MD_SCHEMA",
    "normalize_text",
    "_json_dumps",
    "register",
]

# Backward-compatible private helpers used by existing installations/tests.
_default_config = default_config
_normalize_config = normalize_config
_normalize_tags = normalize_tags
_normalize_metadata = normalize_metadata


def _load_config(hermes_home: str | Path) -> dict[str, Any]:
    return load_config(hermes_home)


class AgentRecallProvider(MemoryProvider):
    """Native Hermes adapter over the host-neutral in-process core."""

    def __init__(self, config: dict | None = None):
        self._initial_config = config or {}
        self._config: dict[str, Any] = {}
        self._core: AgentRecallCore | None = None
        self._pending_embedder: Any = None
        self._session_id = ""
        self._workspace_id = "hermes"
        self._agent_id = "hermes"
        self._user_id = ""
        self._last_prefetch = ""

    @property
    def name(self) -> str:
        return PLUGIN_NAME

    @property
    def _store(self):
        return self._core.store if self._core else None

    @property
    def _embedder(self):
        return self._core.embedder if self._core else self._pending_embedder

    @_embedder.setter
    def _embedder(self, value):
        self._pending_embedder = value
        if self._core:
            self._core.embedder = value

    @property
    def _last_embedding_error(self) -> str:
        return self._core.last_embedding_error if self._core else ""

    def is_available(self) -> bool:
        return True

    def get_config_schema(self):
        return [
            {
                "key": "db_path",
                "description": "SQLite DB path. Use a shared path for cross-agent shared memory.",
                "default": "$HERMES_HOME/agent-recall.db",
            },
            {"key": "workspace_id", "description": "Shared workspace id", "default": "hermes"},
            {"key": "agent_id", "description": "Agent identity for private memory isolation", "default": ""},
            {
                "key": "embedding_base_url",
                "description": "OpenAI-compatible embeddings base URL",
                "default": "",
            },
            {"key": "embedding_model", "description": "Embedding model name", "default": ""},
            {
                "key": "embedding_api_key_env",
                "description": "Env var containing embedding API key",
                "default": "LLM_OPENAI_API_KEY",
            },
            {
                "key": "embedding_dimensions",
                "description": "Embedding dimensions; 0 omits dimensions parameter",
                "default": "0",
            },
            {
                "key": "sqlite_busy_timeout_ms",
                "description": "SQLite lock wait for safe multi-process access",
                "default": "5000",
            },
            {
                "key": "raw_memories_enabled",
                "description": "Enable explicit agent_recall_remember storage",
                "default": "true",
            },
            {
                "key": "curated_memories_enabled",
                "description": "Enable agent_recall_curate feature layer",
                "default": "true",
            },
            {
                "key": "llm_curator_enabled",
                "description": "Enable chat-model memory curation and synthesis calls",
                "default": "true",
            },
            {
                "key": "llm_curator_backend",
                "description": "Curation backend: codex-cli or openai-compatible",
                "default": "codex-cli",
            },
            {
                "key": "llm_curator_model",
                "description": "Chat model used by curation backend",
                "default": "gpt-5.3-mini",
            },
            {"key": "llm_curator_command", "description": "CLI command for codex-cli backend", "default": "codex"},
            {
                "key": "llm_curator_base_url",
                "description": "OpenAI-compatible chat base URL when using openai-compatible backend",
                "default": "",
            },
            {
                "key": "llm_curator_api_key_env",
                "description": "Env var containing chat API key for openai-compatible backend",
                "default": "OPENAI_API_KEY",
            },
            {"key": "conclusions_enabled", "description": "Enable inspectable conclusion storage", "default": "false"},
            {"key": "peer_profiles_enabled", "description": "Enable peer profile synthesis", "default": "false"},
            {
                "key": "workspace_profiles_enabled",
                "description": "Enable workspace profile synthesis",
                "default": "false",
            },
            {"key": "agent_profiles_enabled", "description": "Enable agent profile synthesis", "default": "false"},
            {
                "key": "dialectic_review_enabled",
                "description": "Enable bounded dialectic review jobs",
                "default": "false",
            },
            {
                "key": "conflict_detection_enabled",
                "description": "Enable conflict-detection instructions in review jobs",
                "default": "false",
            },
            {
                "key": "staleness_detection_enabled",
                "description": "Enable stale-memory detection instructions in review jobs",
                "default": "false",
            },
            {
                "key": "promotion_rules_enabled",
                "description": "Enable promotion recommendations in review jobs",
                "default": "false",
            },
            {
                "key": "demotion_rules_enabled",
                "description": "Enable demotion/archive recommendations in review jobs",
                "default": "false",
            },
            {
                "key": "auto_capture_turns",
                "description": "Opt-in completed-turn transcript capture",
                "default": "false",
            },
            {
                "key": "auto_capture_compression_checkpoints",
                "description": "Opt-in pre-compression checkpoint capture",
                "default": "false",
            },
            {
                "key": "allow_any_agent_to_mutate_shared",
                "description": "Allow any workspace agent to edit/delete shared rows",
                "default": "false",
            },
        ]

    def save_config(self, values, hermes_home):
        path = Path(hermes_home).expanduser() / "agent-recall.json"
        config = default_config(hermes_home)
        clean = dict(values or {})
        db_path = str(clean.get("db_path") or config["db_path"]).replace(
            "$HERMES_HOME", str(Path(hermes_home).expanduser())
        )
        clean["db_path"] = db_path
        config.update(clean)
        config = normalize_config(config)
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        with suppress(Exception):
            path.chmod(0o600)

    def initialize(self, session_id: str, **kwargs) -> None:
        if self._core:
            self._core.close()
            self._core = None
        hermes_home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        config = load_config(hermes_home)
        config.update(self._initial_config)
        for marker in ("$HERMES_HOME", "${HERMES_HOME}"):
            config["db_path"] = str(config.get("db_path", "")).replace(marker, str(Path(hermes_home).expanduser()))
        config = normalize_config(config)
        self._session_id = session_id or ""
        self._workspace_id = str(config.get("workspace_id") or kwargs.get("agent_workspace") or "hermes")
        self._agent_id = str(
            config.get("agent_id") or kwargs.get("agent_identity") or Path(hermes_home).name or "hermes"
        )
        if self._agent_id == ".hermes":
            self._agent_id = "hermes"
        self._user_id = str(kwargs.get("user_id") or kwargs.get("user_id_alt") or "")
        identity = AgentIdentity(self._workspace_id, self._agent_id, self._session_id, self._user_id)
        self._core = AgentRecallCore(config, identity, curator_factory=self._make_curator)
        self._config = self._core.config
        if self._pending_embedder is not None:
            self._core.embedder = self._pending_embedder

    def _require_core(self) -> AgentRecallCore:
        if not self._core:
            raise AgentRecallError("AgentRecall is not initialized")
        return self._core

    def _make_curator(self):
        return build_curator(
            self._config,
            codex_cls=CodexCliCurator,
            chat_cls=ChatCompletionsCurator,
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
        return list(HERMES_SCHEMAS)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not self._core:
            return ""
        try:
            result = self._core.prefetch_context(
                query,
                limit=int(self._config.get("prefetch_limit") or 6),
                max_chars=int(self._config.get("max_memory_chars") or 12_000),
                explain=False,
                include_results=False,
                track_access=True,
                session_id=session_id or self._session_id,
            )
            self._last_prefetch = str(result.get("context") or "")
            return self._last_prefetch
        except Exception:
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        self._session_id = new_session_id or ""
        if self._core:
            self._core.rotate_session(self._session_id)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages=None) -> None:
        if self._core:
            self._core.capture_turn(
                user_content,
                assistant_content,
                session_id=session_id or self._session_id,
                background=True,
            )

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        if not self._core:
            return ""
        try:
            memory_id = self._core.checkpoint_from_messages(messages)
            return f"AgentRecall saved a compression checkpoint id={memory_id} for later recall." if memory_id else ""
        except Exception:
            return ""

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        if not self._core:
            return
        with suppress(Exception):
            self._core.mirror_builtin_memory(action, target, content, metadata)

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        try:
            core = self._require_core()
            if tool_name == "agent_recall_remember":
                result = core.remember(args)
            elif tool_name == "agent_recall_search":
                result = core.search(args)
            elif tool_name == "agent_recall_profile":
                result = core.profile(str(args.get("focus") or ""), int(args.get("limit") or 10))
            elif tool_name == "agent_recall_stats":
                result = core.stats()
            elif tool_name == "agent_recall_forget":
                result = core.forget(int(args["id"]))
            elif tool_name == "agent_recall_update":
                result = core.update(int(args["id"]), args)
            elif tool_name == "agent_recall_curate":
                result = core.curate(args)
            elif tool_name == "agent_recall_conclude":
                result = core.conclude(args)
            elif tool_name == "agent_recall_profile_synthesize":
                result = core.profile_synthesize(args)
            elif tool_name == "agent_recall_review":
                result = core.review(args)
            elif tool_name == "agent_recall_import_markdown":
                result = core.import_markdown(args)
            else:
                return tool_error(f"Unknown AgentRecall tool: {tool_name}")
            return json.dumps(result, ensure_ascii=False)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def shutdown(self) -> None:
        if self._core:
            self._core.close()
        self._core = None
        self._pending_embedder = None


def register(ctx) -> None:
    ctx.register_memory_provider(AgentRecallProvider())
