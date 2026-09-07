from __future__ import annotations

import json
import math
import os
import stat
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

MAX_CURATION_RESPONSE_CHARS = 128_000
MAX_CURATION_TRANSPORT_BYTES = 512_000
MAX_CURATED_MEMORIES = 32
MAX_CURATED_CONTENT_CHARS = 1_200
MAX_CURATED_TAGS = 16
MAX_CURATED_METADATA_BYTES = 4_096
MAX_CURATED_METADATA_PROPERTIES = 32
_OPTIONAL_STRING_LIMITS = {
    "title": 200,
    "summary": 500,
    "category": 80,
    "canonical_key_hint": 256,
}
_MAX_CURATED_TAG_CHARS = 64

CURATION_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "maxItems": MAX_CURATED_MEMORIES,
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "maxLength": MAX_CURATED_CONTENT_CHARS},
                    "title": {"type": "string", "maxLength": _OPTIONAL_STRING_LIMITS["title"]},
                    "summary": {"type": "string", "maxLength": _OPTIONAL_STRING_LIMITS["summary"]},
                    "visibility": {"type": "string", "enum": ["agent", "shared", "session"]},
                    "category": {"type": "string", "maxLength": _OPTIONAL_STRING_LIMITS["category"]},
                    "tags": {
                        "type": "array",
                        "maxItems": MAX_CURATED_TAGS,
                        "items": {"type": "string", "maxLength": _MAX_CURATED_TAG_CHARS},
                    },
                    "importance": {"type": "number"},
                    "confidence": {"type": "number"},
                    "metadata": {"type": "object", "maxProperties": MAX_CURATED_METADATA_PROPERTIES},
                    "canonical_key_hint": {
                        "type": "string",
                        "maxLength": _OPTIONAL_STRING_LIMITS["canonical_key_hint"],
                        "description": "Optional stable topical hint; never an authoritative persistence key",
                    },
                },
                "required": ["content"]
            }
        }
    },
    "required": ["memories"]
}


def _candidate_number(item: dict[str, Any], key: str, default: float) -> float | None:
    if key not in item:
        return default
    value = item[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        return None
    return max(0.0, min(parsed, 1.0)) if math.isfinite(parsed) else None


def _clean_candidate(item: dict[str, Any]) -> dict[str, Any] | None:
    content = item.get("content")
    if not isinstance(content, str) or not content.strip() or len(content) > MAX_CURATED_CONTENT_CHARS:
        return None
    cleaned: dict[str, Any] = {"content": content.strip()}
    for key, max_chars in _OPTIONAL_STRING_LIMITS.items():
        if key not in item:
            continue
        value = item[key]
        if not isinstance(value, str) or len(value) > max_chars:
            return None
        if value.strip():
            cleaned[key] = value.strip()
    if "visibility" in item:
        visibility = item["visibility"]
        if not isinstance(visibility, str) or visibility not in {"agent", "shared", "session"}:
            return None
    cleaned["visibility"] = item.get("visibility", "agent")
    tags = item.get("tags", [])
    if not isinstance(tags, list) or len(tags) > MAX_CURATED_TAGS:
        return None
    cleaned_tags: list[str] = []
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip() or len(tag) > _MAX_CURATED_TAG_CHARS:
            return None
        cleaned_tags.append(tag.strip())
    cleaned["tags"] = cleaned_tags
    importance = _candidate_number(item, "importance", 0.5)
    confidence = _candidate_number(item, "confidence", 0.8)
    if importance is None or confidence is None:
        return None
    cleaned["importance"] = importance
    cleaned["confidence"] = confidence
    metadata = item.get("metadata")
    if "metadata" in item and not isinstance(metadata, dict):
        return None
    if isinstance(metadata, dict):
        serialized_metadata = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        if (
            len(metadata) <= MAX_CURATED_METADATA_PROPERTIES
            and len(serialized_metadata.encode("utf-8")) <= MAX_CURATED_METADATA_BYTES
        ):
            cleaned["metadata"] = metadata
    return cleaned


def parse_curation_json(text: str) -> list[dict[str, Any]]:
    """Parse a curation response and return memory dicts.

    Accepts either raw JSON or JSON surrounded by prose/code fences.
    """
    if not isinstance(text, str) or len(text) > MAX_CURATION_RESPONSE_CHARS:
        return []
    raw = text.strip()
    if not raw:
        return []
    candidates = [raw]
    if "```" in raw:
        parts = raw.split("```")
        candidates.extend(part.strip().removeprefix("json").strip() for part in parts)
    if "{" in raw and "}" in raw:
        candidates.append(raw[raw.find("{"): raw.rfind("}") + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        memories = data.get("memories") if isinstance(data, dict) else data
        if not isinstance(memories, list):
            continue
        cleaned: list[dict[str, Any]] = []
        for item in memories[:MAX_CURATED_MEMORIES]:
            if isinstance(item, dict):
                candidate = _clean_candidate(item)
                if candidate:
                    cleaned.append(candidate)
        return cleaned
    return []


def build_curation_prompt(text: str, *, default_visibility: str = "agent") -> str:
    return f"""You extract durable AI-agent memories. Return ONLY JSON matching this schema:
{{"memories":[{{"content":"durable fact", "title":"short", "summary":"optional", "visibility":"agent|shared|session", "category":"user_pref|project|environment|decision|procedure|general", "tags":["tag"], "importance":0.0, "confidence":0.0, "canonical_key_hint":"optional.stable.topic"}}]}}

Rules:
- Keep only durable facts useful after this conversation.
- Do not store secrets, credentials, command output dumps, or transient progress.
- canonical_key_hint is an optional stable topical hint and is NOT authoritative; persistence assigns safe keys.
- Use visibility='{default_visibility}' unless the text explicitly says the fact is useful across agents, then use 'shared'.
- If there is nothing durable, return {{"memories":[]}}.

Text to curate:
---
{text[:12000]}
---
"""


class ChatCompletionsCurator:
    """OpenAI-compatible chat-completions memory curation backend."""

    def __init__(self, base_url: str, model: str, *, api_key: str = "", timeout: float = 120.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or "gpt-5.3-mini"
        self.api_key = api_key or ""
        self.timeout = float(timeout or 120.0)

    def curate(self, text: str, *, default_visibility: str = "agent") -> list[dict[str, Any]]:
        if not self.base_url:
            raise RuntimeError("llm_curator_base_url is required for openai-compatible curation")
        prompt = build_curation_prompt(text, default_visibility=default_visibility)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON for durable AI-agent memory curation."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - user-configured endpoint
            response_bytes = resp.read(MAX_CURATION_TRANSPORT_BYTES + 1)
            if len(response_bytes) > MAX_CURATION_TRANSPORT_BYTES:
                raise RuntimeError("chat curator response exceeds transport limit")
            payload = json.loads(response_bytes.decode("utf-8"))
        message = payload.get("choices", [{}])[0].get("message", {})
        content = message.get("content") if isinstance(message, dict) else ""
        if not isinstance(content, str):
            content = ""
        return parse_curation_json(content)


class CodexCliCurator:
    """Configurable chat-model memory curation backend.

    The current implementation shells out to a local CLI so deployments can use
    whatever authenticated model/runtime that command exposes. AgentRecall keeps
    the backend, command, model, and timeout in config so users can swap them or
    disable curation without changing code.
    """

    def __init__(self, command: str = "codex", model: str = "gpt-5.3-mini", timeout: float = 120.0) -> None:
        self.command = command or "codex"
        self.model = model or "gpt-5.3-mini"
        self.timeout = float(timeout or 120.0)

    def curate(self, text: str, *, default_visibility: str = "agent") -> list[dict[str, Any]]:
        prompt = build_curation_prompt(text, default_visibility=default_visibility)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "codex-memory.json"
            cmd = [
                self.command,
                "--ask-for-approval",
                "never",
                "exec",
                "--model",
                self.model,
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--output-last-message",
                str(out),
                "-",
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"codex curator timed out after {exc.timeout} seconds") from None
            except (OSError, ValueError):
                raise RuntimeError("codex curator failed to start") from None
            if proc.returncode != 0:
                raise RuntimeError(f"codex curator failed with exit code {proc.returncode}")
            try:
                flags = os.O_RDONLY
                flags |= getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NONBLOCK", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(out, flags)
                try:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise RuntimeError("codex curator output unavailable or invalid")
                    response_bytes = bytearray()
                    while len(response_bytes) <= MAX_CURATION_TRANSPORT_BYTES:
                        chunk = os.read(
                            fd,
                            MAX_CURATION_TRANSPORT_BYTES + 1 - len(response_bytes),
                        )
                        if not chunk:
                            break
                        response_bytes.extend(chunk)
                finally:
                    os.close(fd)
                if len(response_bytes) > MAX_CURATION_TRANSPORT_BYTES:
                    raise RuntimeError("codex curator output unavailable or invalid")
                response = response_bytes.decode("utf-8")
            except RuntimeError:
                raise
            except (OSError, UnicodeError):
                raise RuntimeError("codex curator output unavailable or invalid") from None
        return parse_curation_json(response)
