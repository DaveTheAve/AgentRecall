#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# Only the native plugin's supported release surface belongs in a copied install.
# Do not traverse arbitrary checkout contents (secrets, archives, or runtime data).
COPY_FILES = frozenset({
    "__init__.py", "cli.py", "plugin.yaml",
    "agent_recall_core.py", "agent_recall_curator.py", "agent_recall_store.py",
    "agent_recall_schemas.py", "agent_recall_mcp.py", "agent_recall_bridge.py",
    "hermes_plugin/__init__.py", "hermes_plugin/plugin.yaml",
    "LICENSE", "README.md", "INSTALL.md", "CHANGELOG.md", "SECURITY.md",
    "CONTRIBUTING.md", "pyproject.toml", "uv.lock",
    "docs/ARCHITECTURE.md", "docs/MCP.md", "docs/OPENCLAW.md",
    "scripts/install_user_plugin.py", "scripts/install_openclaw_plugin.py",
})


def _copy_ignore(source: Path):
    directories = {parent.as_posix() for name in COPY_FILES for parent in Path(name).parents}
    allowed = COPY_FILES | directories
    for relative in sorted(allowed):
        if (source / relative).is_symlink():
            raise SystemExit(f"Refusing to copy release path through a symlink: {relative}")

    def ignore(directory: str, names: list[str]) -> list[str]:
        excluded = []
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(source).as_posix()
            # Skip all symlinks, including allowed names pointing outside the tree.
            if path.is_symlink() or relative not in COPY_FILES | directories:
                excluded.append(name)
        return excluded

    return ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Install AgentRecall as a Hermes user memory plugin")
    parser.add_argument("--hermes-home", default=os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = parser.parse_args()

    src = Path(__file__).resolve().parents[1]
    dest = Path(args.hermes_home).expanduser() / "plugins" / "agent-recall"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        raise SystemExit(f"Destination already exists: {dest}")
    if args.copy:
        shutil.copytree(src, dest, ignore=_copy_ignore(src))
    else:
        dest.symlink_to(src, target_is_directory=True)
    print(f"Installed AgentRecall at {dest}")
    print("Restart Hermes, then run: hermes memory setup agent-recall")


if __name__ == "__main__":
    main()
