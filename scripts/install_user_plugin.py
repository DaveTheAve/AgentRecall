#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


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
        ignore = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "$tmp", "*.db", "*.db-wal", "*.db-shm")
        shutil.copytree(src, dest, ignore=ignore)
    else:
        dest.symlink_to(src, target_is_directory=True)
    print(f"Installed AgentRecall at {dest}")
    print("Restart Hermes, then run: hermes memory setup agent-recall")


if __name__ == "__main__":
    main()
