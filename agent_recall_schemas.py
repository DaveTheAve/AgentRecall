from __future__ import annotations

from copy import deepcopy
from typing import Any

REMEMBER_SCHEMA: dict[str, Any] = {
    "name": "agent_recall_remember",
    "description": "Store a durable memory in AgentRecall. Choose visibility='shared' only for facts useful across agents; use 'agent' for profile-specific facts and 'session' for temporary session recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The factual memory content to store."},
            "title": {"type": "string", "description": "Optional short title."},
            "summary": {"type": "string", "description": "Optional concise summary."},
            "visibility": {"type": "string", "enum": ["agent", "shared", "session"], "description": "Isolation scope."},
            "category": {
                "type": "string",
                "description": "Category such as user_pref, project, environment, decision, procedure, transcript.",
            },
            "tags": {"type": "array", "items": {"type": "string"}},
            "importance": {"type": "number", "description": "0.0-1.0 importance."},
            "confidence": {"type": "number", "description": "0.0-1.0 confidence."},
            "metadata": {"type": "object", "description": "Optional JSON metadata."},
        },
        "required": ["content"],
    },
}

SEARCH_SCHEMA: dict[str, Any] = {
    "name": "agent_recall_search",
    "description": "Search visible AgentRecall memories with hybrid embedding + lexical retrieval. Enforces workspace/agent/session ACL isolation.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 8},
            "include_shared": {"type": "boolean", "default": True},
            "category": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "visibility": {"type": "string", "enum": ["agent", "shared", "session"]},
            "source_agent_id": {"type": "string"},
            "min_importance": {"type": "number"},
            "updated_after": {"type": "number", "description": "Unix timestamp lower bound."},
            "explain": {"type": "boolean", "default": False, "description": "Include score component explanations."},
        },
        "required": ["query"],
    },
}

PROFILE_SCHEMA: dict[str, Any] = {
    "name": "agent_recall_profile",
    "description": "Return a compact view of current agent, workspace, isolation settings, and top memories relevant to a focus query.",
    "parameters": {
        "type": "object",
        "properties": {"focus": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
    },
}

UPDATE_SCHEMA: dict[str, Any] = {
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
            "metadata": {"type": "object"},
            "importance": {"type": "number"},
            "confidence": {"type": "number"},
            "archived": {"type": "boolean"},
        },
        "required": ["id"],
    },
}

FORGET_SCHEMA: dict[str, Any] = {
    "name": "agent_recall_forget",
    "description": "Delete a visible AgentRecall memory by id, respecting isolation ACLs.",
    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
}

CURATE_SCHEMA: dict[str, Any] = {
    "name": "agent_recall_curate",
    "description": "Use the configured chat-model curation backend to extract durable memory candidates from text, then store selected candidates in AgentRecall. Set llm_curator_enabled=false to turn this off.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "default_visibility": {"type": "string", "enum": ["agent", "shared", "session"], "default": "agent"},
            "dry_run": {"type": "boolean", "default": False},
        },
        "required": ["text"],
    },
}

STATS_SCHEMA: dict[str, Any] = {
    "name": "agent_recall_stats",
    "description": "Show AgentRecall database counts by workspace/agent/visibility/category.",
    "parameters": {"type": "object", "properties": {}},
}

IMPORT_MD_SCHEMA: dict[str, Any] = {
    "name": "agent_recall_import_markdown",
    "description": "Ingest a markdown note/file into AgentRecall as memory chunks. Use for Obsidian/Hermes vault notes; respects excluded terms.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "visibility": {"type": "string", "enum": ["agent", "shared", "session"]},
            "category": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "chunk_chars": {"type": "integer", "default": 3000},
        },
        "required": ["path"],
    },
}

CONCLUDE_SCHEMA: dict[str, Any] = {
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
        "required": ["content"],
    },
}

PROFILE_SYNTH_SCHEMA: dict[str, Any] = {
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
        },
    },
}

REVIEW_SCHEMA: dict[str, Any] = {
    "name": "agent_recall_review",
    "description": "Run a bounded dialectic review for conflicts, staleness, and promotion/demotion recommendations using the configured chat-model backend.",
    "parameters": {
        "type": "object",
        "properties": {
            "focus": {"type": "string"},
            "dry_run": {"type": "boolean", "default": True},
            "limit": {"type": "integer", "default": 20},
        },
    },
}

HERMES_SEARCH_SCHEMA = deepcopy(SEARCH_SCHEMA)
for _key in ("visibility", "source_agent_id", "min_importance", "updated_after", "explain"):
    HERMES_SEARCH_SCHEMA["parameters"]["properties"].pop(_key)

HERMES_UPDATE_SCHEMA = deepcopy(UPDATE_SCHEMA)
HERMES_UPDATE_SCHEMA["parameters"]["properties"].pop("metadata")

HERMES_SCHEMAS = [
    REMEMBER_SCHEMA,
    HERMES_SEARCH_SCHEMA,
    PROFILE_SCHEMA,
    HERMES_UPDATE_SCHEMA,
    FORGET_SCHEMA,
    CURATE_SCHEMA,
    CONCLUDE_SCHEMA,
    PROFILE_SYNTH_SCHEMA,
    REVIEW_SCHEMA,
    STATS_SCHEMA,
    IMPORT_MD_SCHEMA,
]
