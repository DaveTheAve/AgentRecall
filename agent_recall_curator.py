from __future__ import annotations

import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

CURATION_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "visibility": {"type": "string", "enum": ["agent", "shared", "session"]},
                    "category": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "number"},
                    "confidence": {"type": "number"}
                },
                "required": ["content"]
            }
        }
    },
    "required": ["memories"]
}


def _as_number(value: Any, default: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except Exception:
        return default


def _clean_candidate(item: dict[str, Any]) -> dict[str, Any] | None:
    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    cleaned: dict[str, Any] = {"content": content.strip()}
    for key in ["title", "summary", "category"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            cleaned[key] = value.strip()
    visibility = item.get("visibility")
    cleaned["visibility"] = visibility if visibility in {"agent", "shared", "session"} else "agent"
    tags = item.get("tags")
    cleaned["tags"] = [t.strip() for t in tags if isinstance(t, str) and t.strip()] if isinstance(tags, list) else []
    cleaned["importance"] = _as_number(item.get("importance"), 0.5)
    cleaned["confidence"] = _as_number(item.get("confidence"), 0.8)
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        cleaned["metadata"] = metadata
    return cleaned


def parse_curation_json(text: str) -> list[dict[str, Any]]:
    """Parse a curation response and return memory dicts.

    Accepts either raw JSON or JSON surrounded by prose/code fences.
    """
    raw = (text or "").strip()
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
        for item in memories:
            if isinstance(item, dict):
                candidate = _clean_candidate(item)
                if candidate:
                    cleaned.append(candidate)
        return cleaned
    return []


def build_curation_prompt(text: str, *, default_visibility: str = "agent") -> str:
    return f"""You extract durable AI-agent memories. Return ONLY JSON matching this schema:
{{"memories":[{{"content":"durable fact", "title":"short", "summary":"optional", "visibility":"agent|shared|session", "category":"user_pref|project|environment|decision|procedure|general", "tags":["tag"], "importance":0.0, "confidence":0.0}}]}}

Rules:
- Keep only durable facts useful after this conversation.
- Do not store secrets, credentials, command output dumps, or transient progress.
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
            payload = json.loads(resp.read().decode("utf-8"))
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
                "exec",
                "--model",
                self.model,
                "--sandbox",
                "read-only",
                "--ask-for-approval",
                "never",
                "--skip-git-repo-check",
                "--ephemeral",
                "--output-last-message",
                str(out),
                prompt,
            ]
            try:
                proc = subprocess.run(cmd, text=True, capture_output=True, timeout=self.timeout, check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"codex curator timed out after {exc.timeout} seconds") from None
            if proc.returncode != 0:
                raise RuntimeError(f"codex curator failed with exit code {proc.returncode}")
            response = out.read_text(encoding="utf-8") if out.exists() else proc.stdout
        return parse_curation_json(response)
