"""Dependency-free MCP contracts. Public data is projected, never regex-redacted."""
from __future__ import annotations

import json
import math
from typing import Any

MAX_ARGUMENT_BYTES = 65_536
MAX_BODY_BYTES = 131_072
MAX_DEPTH = 8
MAX_ITEMS = 1024

ERRORS = {
    "invalid_arguments": "Invalid tool arguments.",
    "forbidden": "Operation not allowed by read-only policy or tool allowlist.",
    "backend_error": "Operation failed.",
    "not_found": "Memory not found or not visible.",
    "busy": "Server capacity exhausted; retry later.",
    "timeout": "Operation deadline exceeded; in-flight work may complete. No rollback is implied.",
    "closed": "Server is shutting down.",
}


class MCPPublicError(ValueError):
    def __init__(self, code: str):
        self.code = code if code in ERRORS else "backend_error"
        super().__init__(f"{self.code}: {ERRORS[self.code]}")


class MCPAccessError(MCPPublicError, PermissionError):
    def __init__(self):
        super().__init__("forbidden")


def string(limit=512, minimum=0, **extra):
    return {"type": "string", "minLength": minimum, "maxLength": limit, **extra}


def number(low=0, high=1):
    return {"type": "number", "minimum": low, "maximum": high}


def integer(low=0, high=2**53 - 1):
    return {"type": "integer", "minimum": low, "maximum": high}


def array(items, limit=64):
    return {"type": "array", "items": items, "maxItems": limit}


def obj(properties, required=()):
    return {"type": "object", "properties": properties, "additionalProperties": False, "required": list(required)}


BOOL = {"type": "boolean"}
TEXT = string(12_000)
ID = integer(1)
VIS = string(enum=["agent", "shared", "session"])
TAGS = array(string(128))
SCOPE = string(enum=["general", "peer", "workspace", "agent"])
# Arbitrary metadata may be stored, but not returned. Runtime limits apply to
# this JSON subtree too; identity/provenance override keys are forbidden.
METADATA_INPUT = {"type": "object", "maxProperties": 64, "additionalProperties": True}
MEMORY_INPUT = {
    "content": string(12_000, 1), "title": string(512), "summary": string(4096),
    "visibility": VIS, "category": string(128), "tags": TAGS,
    "importance": number(), "confidence": number(), "expires_at": number(0, 253402300799),
    "metadata": METADATA_INPUT,
}
INPUT_SCHEMAS = {
    "remember": obj({**MEMORY_INPUT, "canonical_key": string(512)}, ["content"]),
    "update": obj({**MEMORY_INPUT, "id": ID, "archived": BOOL}, ["id"]),
    "forget": obj({"id": ID}, ["id"]),
    "get_memory": obj({"id": ID, "include_shared": BOOL}, ["id"]),
    "search": obj({
        "query": TEXT, "limit": integer(1, 50), "include_shared": BOOL,
        "category": string(128), "tags": TAGS, "visibility": string(enum=["", "agent", "shared", "session"]),
        "source_agent_id": string(512), "min_importance": number(),
        "updated_after": number(0, 253402300799), "explain": BOOL,
    }, ["query"]),
    "prefetch_context": obj({"query": TEXT, "limit": integer(1, 50), "max_chars": integer(1, 12_000),
                             "explain": BOOL, "include_results": BOOL}, ["query"]),
    "profile": obj({"focus": TEXT, "limit": integer(1, 50)}),
    "curate": obj({"text": string(12_000, 1), "default_visibility": VIS, "dry_run": BOOL}, ["text"]),
    "conclude": obj({"content": string(12_000, 1), "scope": SCOPE, "subject": string(512),
                     "source_ids": array(ID), "visibility": VIS, "confidence": number(), "supersedes": array(ID)}, ["content"]),
    "profile_synthesize": obj({"scope": string(enum=["peer", "workspace", "agent"]), "subject": string(512),
                               "focus": TEXT, "dry_run": BOOL, "visibility": VIS}),
    "review": obj({"focus": TEXT, "dry_run": BOOL, "limit": integer(1, 50)}),
    "stats": obj({}), "health": obj({}), "capabilities": obj({}),
}
DEFAULTS = {
    "remember": {"visibility": "agent"},
    "search": {"limit": 8, "include_shared": True},
    "prefetch_context": {"limit": 6, "max_chars": 12_000, "explain": True, "include_results": False},
    "profile": {"focus": "", "limit": 10},
    "curate": {"dry_run": True, "default_visibility": "agent"},
    "profile_synthesize": {"dry_run": True, "scope": "peer", "visibility": "agent"},
    "review": {"dry_run": True, "limit": 20},
}


def bounded_json(value: Any, *, byte_limit=MAX_ARGUMENT_BYTES, identities=False,
                 string_limit=12_000, depth_limit=MAX_DEPTH):
    """Check complexity before serialization, schema validation, or backend work."""
    pending = [(value, 0)]
    count = size = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > depth_limit or count > MAX_ITEMS:
            raise MCPPublicError("invalid_arguments")
        if type(item) is dict:
            if len(item) > 64:
                raise MCPPublicError("invalid_arguments")
            for key, child in item.items():
                if type(key) is not str or len(key) > 128:
                    raise MCPPublicError("invalid_arguments")
                normalized = key.lower().replace("_", "").replace("-", "")
                if identities and normalized in {"identity", "workspaceid", "agentid", "sessionid", "userid", "sourcesessionid", "targetsessionid"}:
                    raise MCPPublicError("invalid_arguments")
                size += len(key.encode("utf-8"))
                pending.append((child, depth + 1))
        elif type(item) is list:
            if len(item) > 64:
                raise MCPPublicError("invalid_arguments")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > string_limit:
                raise MCPPublicError("invalid_arguments")
            size += len(item.encode("utf-8"))
        elif type(item) in (int, float):
            if not math.isfinite(item) or abs(item) > 2**53 - 1:
                raise MCPPublicError("invalid_arguments")
        elif item is not None and type(item) is not bool:
            raise MCPPublicError("invalid_arguments")
        if size > byte_limit:
            raise MCPPublicError("invalid_arguments")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) > byte_limit:
        raise MCPPublicError("invalid_arguments")


def validate(value, schema):
    kind = schema["type"]
    valid_type = {
        "object": type(value) is dict, "array": type(value) is list,
        "string": type(value) is str, "integer": type(value) is int,
        "number": type(value) in (int, float), "boolean": type(value) is bool,
    }[kind]
    if not valid_type:
        raise MCPPublicError("invalid_arguments")
    if "enum" in schema and value not in schema["enum"]:
        raise MCPPublicError("invalid_arguments")
    if kind == "object":
        props = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            raise MCPPublicError("invalid_arguments")
        if not schema.get("additionalProperties", True) and value.keys() - props.keys():
            raise MCPPublicError("invalid_arguments")
        if len(value) > schema.get("maxProperties", 64):
            raise MCPPublicError("invalid_arguments")
        for key in value.keys() & props.keys():
            validate(value[key], props[key])
    elif kind == "array":
        if len(value) > schema["maxItems"]:
            raise MCPPublicError("invalid_arguments")
        for item in value:
            validate(item, schema["items"])
    elif kind == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema["maxLength"]:
            raise MCPPublicError("invalid_arguments")
    elif kind in ("number", "integer"):
        if not math.isfinite(value) or not schema["minimum"] <= value <= schema["maximum"]:
            raise MCPPublicError("invalid_arguments")


def validate_arguments(operation, args):
    values = {} if args is None else args
    try:
        bounded_json(values)
        validate(values, INPUT_SCHEMAS[operation])
        if "metadata" in values:
            bounded_json(values["metadata"], identities=True)
        return {**DEFAULTS.get(operation, {}), **json.loads(json.dumps(values))}
    except (MCPPublicError, TypeError, ValueError, OverflowError, RecursionError):
        raise MCPPublicError("invalid_arguments") from None


MODULES = ["conclusions", "peer_profiles", "workspace_profiles", "agent_profiles", "dialectic_review"]
REVIEW_MODULES = ["conflict_detection", "staleness_detection", "promotion_rules", "demotion_rules"]
PUBLIC_METADATA = obj({
    "module": string(enum=MODULES), "scope": SCOPE,
    "source_ids": array(ID), "supersedes": array(ID),
    "enabled_modules": array(string(enum=REVIEW_MODULES)),
    "provenance": string(enum=["agent_recall_conclude"]),
    "source": string(enum=["agent_recall_profile_synthesize", "agent_recall_review"]),
})
MEMORY = obj({
    **{key: value for key, value in MEMORY_INPUT.items() if key != "metadata"},
    "id": ID, "metadata": PUBLIC_METADATA, "workspace_id": string(512), "agent_id": string(512),
    "source_agent_id": string(512), "canonical_key": string(512), "archived": integer(0, 1),
    "created_at": number(0, 253402300799), "updated_at": number(0, 253402300799),
    "last_accessed_at": number(0, 253402300799), "access_count": integer(),
    "score": number(-1e6, 1e6),
    "score_explanation": obj({key: number(-1e6, 1e6) for key in
                              ["vector", "lexical", "bm25", "exact_lexical_boost", "importance", "recency"]}),
})
IDENTITY = obj({"workspace_id": string(512), "agent_id": string(512)})
WRITE_RESULT = obj({"success": BOOL, "id": ID, "action": string(enum=["added", "updated", "unchanged", "created"]), "visibility": VIS}, ["success"])
OUTPUT_SCHEMAS = {
    "remember": WRITE_RESULT, "conclude": WRITE_RESULT,
    "search": obj({"success": BOOL, "results": array(MEMORY, 50), "count": integer(0, 50)}, ["success"]),
    "get_memory": obj({"success": BOOL, "memory": MEMORY}, ["success"]),
    "prefetch_context": obj({"success": BOOL, "context": TEXT, "results": array(MEMORY, 50),
                             "count": integer(0, 50), "identity": IDENTITY}, ["success"]),
    "profile": obj({"success": BOOL, **IDENTITY["properties"], "recall": array(MEMORY, 50)}, ["success"]),
    "update": obj({"success": BOOL, "updated": BOOL}, ["success"]),
    "forget": obj({"success": BOOL, "deleted": BOOL}, ["success"]),
    "stats": obj({"success": BOOL, "stats": obj({"total": integer(), "buckets": array(obj({
        "visibility": VIS, "category": string(128), "agent_id": string(512), "count": integer()}), 64)})}, ["success"]),
    "health": obj({"success": BOOL, "identity": IDENTITY,
        "sqlite": obj({"quick_check": string(enum=["ok", "unavailable"]), "journal_mode": string(enum=["wal", "delete", "memory", "truncate", "persist", "off"]),
                       "busy_timeout_ms": integer(), "foreign_keys": BOOL}),
        "embedding": obj({"configured": BOOL, "enabled": BOOL}), "snapshot": BOOL,
        "runtime": obj({"active": integer(), "capacity": integer(), "closed": BOOL})}, ["success"]),
    "capabilities": obj({"success": BOOL, "engine": string(enum=["AgentRecall"]), "native_host_required": BOOL,
        **IDENTITY["properties"], "operations": array(string(enum=list(INPUT_SCHEMAS))),
        "visibility": array(VIS), "access": string(enum=["read-only", "read-write"]),
        "tools": array(string(enum=list(INPUT_SCHEMAS))),
        "features": obj({key: BOOL for key in ["semantic_search", "lexical_fallback", "score_explanations",
            "context_budgets", "curation", "conclusions", "profile_synthesis", "review"]})}, ["success"]),
}
for _operation in ("curate", "profile_synthesize", "review"):
    OUTPUT_SCHEMAS[_operation] = obj({"success": BOOL, "stored": integer(), "scope": SCOPE,
        "subject": string(512), "candidates": array(MEMORY), "recommendations": array(MEMORY),
        "results": array(WRITE_RESULT), "enabled_modules": array(string(enum=REVIEW_MODULES))}, ["success"])


def project(value, schema):
    """Only approved keys/types leave the boundary; opaque content is unchanged.

    Fail on over-budget approved data rather than silently truncating memory text
    or returning counts inconsistent with clipped lists. Unknown fields are never
    traversed, including arbitrary metadata and internal diagnostics.
    """
    if schema["type"] == "object" and type(value) is dict:
        result = {key: project(value[key], sub) for key, sub in schema["properties"].items() if key in value}
    elif schema["type"] == "array" and type(value) is list:
        if len(value) > schema["maxItems"]:
            raise MCPPublicError("backend_error")
        result = [project(item, schema["items"]) for item in value]
    else:
        result = value
    validate(result, schema)
    return result


def public_result(operation, value):
    try:
        if type(value) is not dict:
            raise MCPPublicError("backend_error")
        if "error" in value:
            raise MCPPublicError("not_found" if operation == "get_memory" else "backend_error")
        result = project(value, OUTPUT_SCHEMAS[operation])
        if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 1_048_576:
            raise MCPPublicError("backend_error")
        return result
    except Exception:
        if operation == "get_memory" and type(value) is dict and value.get("success") is False:
            raise MCPPublicError("not_found") from None
        raise MCPPublicError("backend_error") from None
