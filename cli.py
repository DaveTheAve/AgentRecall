from __future__ import annotations

import json
import re
from pathlib import Path


def _config_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "agent-recall.json"
    except Exception:
        return Path.home() / ".hermes" / "agent-recall.json"


def _redact_config(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if re.search(r"(?i)(api[_-]?key|secret|password|token|bearer|credential)", str(key)):
                out[key] = "<redacted>"
            else:
                out[key] = _redact_config(item)
        return out
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


def cmd(args):
    sub = getattr(args, "agent_recall_command", None)
    path = _config_path()
    if sub == "status":
        print(f"Config: {path}")
        if path.exists():
            cfg = json.loads(path.read_text(encoding="utf-8"))
            print(json.dumps(_redact_config(cfg), indent=2))
        else:
            print("No config file yet. Run: hermes memory setup agent-recall")
    else:
        print("Usage: hermes agent-recall status")


def register_cli(subparser) -> None:
    subs = subparser.add_subparsers(dest="agent_recall_command")
    subs.add_parser("status", help="Show AgentRecall config/status")
    subparser.set_defaults(func=cmd)
