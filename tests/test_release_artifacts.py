"""Regressions for source distribution and native user-plugin installation."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_source_wrapper_matches_hermes_discovery_contract():
    source = (ROOT / "__init__.py").read_text(encoding="utf-8")[:8192]
    assert "MemoryProvider" in source or "register_memory_provider" in source


def _source_copy(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info",
        "build", "dist", "node_modules", "Archive.tar.gz", ".env", ".venv",
    ))
    return source


@pytest.mark.parametrize("payload", [".env", "Archive.tar.gz", "runtime/session.json", "hermes_plugin/local.db", "docs/private.tar.gz", "build/generated.py"])
def test_copy_install_excludes_local_artifacts(tmp_path, payload):
    source = _source_copy(tmp_path)
    secret = source / payload
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("fake-private-sentinel", encoding="utf-8")
    home = tmp_path / "home"
    subprocess.run([sys.executable, str(source / "scripts/install_user_plugin.py"),
                    "--copy", "--hermes-home", str(home)], check=True)
    dest = home / "plugins/agent-recall"
    assert not (dest / payload).exists()
    assert secret.read_text() == "fake-private-sentinel"
    assert (dest / "__init__.py").is_file()
    assert (dest / "hermes_plugin/__init__.py").is_file()
    assert (dest / "agent_recall_store.py").is_file()
    assert (dest / "agent_recall_session_archive.py").is_file()

    marker = dest / "keep.txt"
    marker.write_text("existing installation")
    result = subprocess.run([sys.executable, str(source / "scripts/install_user_plugin.py"),
                             "--copy", "--hermes-home", str(home)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Destination already exists" in result.stderr
    assert marker.read_text() == "existing installation"


@pytest.mark.parametrize("relative", ["agent_recall_core.py", "hermes_plugin"])
def test_copy_install_does_not_follow_symlinks(tmp_path, relative):
    source = _source_copy(tmp_path)
    target = source / relative
    outside = tmp_path / "outside"
    target.rename(outside)
    target.symlink_to(outside, target_is_directory=outside.is_dir())
    home = tmp_path / "home"
    result = subprocess.run([sys.executable, str(source / "scripts/install_user_plugin.py"),
                             "--copy", "--hermes-home", str(home)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert not (home / "plugins/agent-recall").exists()
    assert outside.exists()
    assert target.is_symlink()


def test_sdist_contains_testable_release_surface_only(tmp_path):
    source = _source_copy(tmp_path)
    for name in [".env", "Archive.tar.gz", "hermes_plugin/private.db", "docs/private.tar.gz"]:
        (source / name).write_text("fake-private-sentinel")
    result = subprocess.run([sys.executable, "-c",
                             "from setuptools.build_meta import build_sdist; build_sdist('dist')"],
                            cwd=source, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(next((source / "dist").glob("*.tar.gz"))) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    required = {"tests/conftest.py", "tests/test_session_archive.py", "__init__.py",
                "agent_recall_session_archive.py", "plugin.yaml", "package.json",
                "openclaw.plugin.json", "openclaw_plugin/test/plugin.test.mjs",
                "scripts/install_user_plugin.py", "scripts/benchmark_retrieval.py",
                "scripts/smoke_packed_openclaw.mjs", "docs/OPENCLAW.md", "CHANGELOG.md",
                "scripts/benchmark_version_comparison.py", "benchmarks/version_comparison/README.md",
                "tests/test_version_comparison.py", "tests/test_mcp_release.py",
                "tests/test_explicit_memory.py", "MANIFEST.in", ".gitignore", ".npmignore"}
    assert required <= names, sorted(required - names)
    retired = {
        "scripts/benchmark_learning_quality.py",
        "tests/test_learning_benchmark.py",
        "tests/test_session_learning.py",
        "tests/test_session_end_learning.py",
        "tests/test_learning_core_extraction.py",
    }
    assert not retired & names, sorted(retired & names)
    assert not any(name.startswith("benchmarks/learning_quality") for name in names)
    assert not {".env", "Archive.tar.gz", "hermes_plugin/private.db", "docs/private.tar.gz"} & names


@pytest.mark.skipif(not shutil.which("npm") or not shutil.which("node"), reason="npm and node are required")
def test_npm_pack_contains_session_modules_and_bridge_starts(tmp_path):
    source = _source_copy(tmp_path)
    pack_dir = tmp_path / "npm-pack"
    pack_dir.mkdir()
    result = subprocess.run(
        ["npm", "pack", "--pack-destination", str(pack_dir)],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    archive_path = next(pack_dir.glob("*.tgz"))
    extracted = tmp_path / "extracted"
    with tarfile.open(archive_path) as archive:
        names = set(archive.getnames())
        archive.extractall(extracted, filter="data")
    required = {
        "package/agent_recall_session_archive.py",
    }
    assert required <= names, sorted(required - names)

    smoke = subprocess.run(
        ["node", str(source / "scripts/smoke_packed_openclaw.mjs"), str(extracted / "package")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
    assert "packed-openclaw-bridge-smoke-ok" in smoke.stdout


@pytest.mark.skipif(not os.environ.get("AGENT_RECALL_HERMES_PYTHON"), reason="opt-in real Hermes runtime")
@pytest.mark.parametrize("copy", [False, True])
def test_real_hermes_discovery(tmp_path, copy):
    home = tmp_path / "home"
    args = [sys.executable, str(ROOT / "scripts/install_user_plugin.py"), "--hermes-home", str(home)]
    if copy:
        args.append("--copy")
    subprocess.run(args, check=True)
    env = dict(os.environ, HERMES_HOME=str(home), HERMES_ENABLE_PROJECT_PLUGINS="false")
    code = """
from pathlib import Path
import os
from plugins.memory import find_provider_dir, list_memory_provider_names, load_memory_provider
from agent.memory_provider import MemoryProvider
assert 'agent-recall' in list_memory_provider_names()
assert find_provider_dir('agent-recall') == Path(os.environ['HERMES_HOME']) / 'plugins/agent-recall'
provider = load_memory_provider('agent-recall', register_skills=False)
assert isinstance(provider, MemoryProvider), provider
print(type(provider).__name__)
"""
    result = subprocess.run([os.environ["AGENT_RECALL_HERMES_PYTHON"], "-c", code],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
