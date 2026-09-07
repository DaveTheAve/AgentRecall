from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Install the native AgentRecall memory plugin for OpenClaw")
    value.add_argument("--openclaw", default="openclaw", help="OpenClaw executable")
    value.add_argument(
        "--root", default=str(Path(__file__).resolve().parents[1]), help="AgentRecall repository/package root"
    )
    value.add_argument("--config-path", default=str(Path.home() / ".agent-recall" / "agent-recall.json"))
    value.add_argument("--workspace-id", default="default")
    value.add_argument("--agent-id-prefix", default="openclaw")
    value.add_argument("--python-command", default="python3", help="Python >=3.10 executable for the local bridge")
    value.add_argument("--copy", action="store_true", help="Copy the plugin instead of linking the development tree")
    value.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return value


def commands(args: argparse.Namespace) -> list[list[str]]:
    install = [args.openclaw, "plugins", "install"]
    if not args.copy:
        install.append("--link")
    # The bridge is intentionally a local child process. OpenClaw requires an explicit
    # operator override for external plugins containing node:child_process.
    install.extend(["--dangerously-force-unsafe-install", str(Path(args.root).expanduser().resolve())])
    settings = {
        "plugins.entries.agent-recall.hooks.allowConversationAccess": True,
        "plugins.entries.agent-recall.hooks.allowPromptInjection": True,
        "plugins.entries.agent-recall.config.configPath": str(Path(args.config_path).expanduser().resolve()),
        "plugins.entries.agent-recall.config.workspaceId": args.workspace_id,
        "plugins.entries.agent-recall.config.agentIdPrefix": args.agent_id_prefix,
        "plugins.entries.agent-recall.config.pythonCommand": args.python_command,
    }
    result = [install]
    for key, value in settings.items():
        result.append([args.openclaw, "config", "set", key, json.dumps(value), "--json"])
    result.append([args.openclaw, "plugins", "inspect", "agent-recall", "--runtime", "--json"])
    return result


def normalized_python_command(value: str) -> str:
    expanded = Path(value).expanduser()
    if expanded.is_absolute() or value.startswith("~") or "/" in value or "\\" in value:
        # Resolving symlinks would turn a venv interpreter into its base Python.
        return str(expanded.absolute())
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.python_command = normalized_python_command(args.python_command)
    if not args.dry_run and shutil.which(args.openclaw) is None and not Path(args.openclaw).exists():
        raise SystemExit(f"OpenClaw executable not found: {args.openclaw}")
    if not args.dry_run and shutil.which(args.python_command) is None:
        raise SystemExit(f"Python executable not found: {args.python_command}")
    if not args.dry_run:
        config_path = Path(args.config_path).expanduser()
        if not config_path.is_file():
            raise SystemExit(f"AgentRecall config file not found: {config_path}")
        try:
            config_value = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"AgentRecall config is not valid JSON: {config_path}") from exc
        if not isinstance(config_value, dict):
            raise SystemExit(f"AgentRecall config root must be an object: {config_path}")
    for command in commands(args):
        if args.dry_run:
            print(json.dumps(command))
            continue
        subprocess.run(command, check=True)
    if not args.dry_run:
        print("AgentRecall is selected as OpenClaw's native memory plugin. Restart the OpenClaw gateway.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
