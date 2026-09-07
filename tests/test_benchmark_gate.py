from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import benchmark_retrieval
from scripts.benchmark_retrieval import _relevant_rank, run_synthetic_benchmark


def test_duplicate_equivalent_memory_counts_as_relevant():
    assert _relevant_rank([9, 12, 15], {12, 99}) == 2
    assert _relevant_rank([9, 12, 15], {98, 99}) == 0


def test_synthetic_benchmark_refuses_to_overwrite_an_existing_path(tmp_path):
    existing = tmp_path / "do-not-delete.db"
    sentinel = b"existing database sentinel"
    existing.write_bytes(sentinel)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_synthetic_benchmark(existing)

    assert existing.read_bytes() == sentinel


def test_synthetic_benchmark_atomically_refuses_a_racing_database_creator(tmp_path, monkeypatch):
    db_path = tmp_path / "racing.db"
    sentinel = b"database created by another process"
    real_link = benchmark_retrieval.os.link

    def racing_link(source, target):
        Path(target).write_bytes(sentinel)
        return real_link(source, target)

    monkeypatch.setattr(benchmark_retrieval.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_synthetic_benchmark(db_path)

    assert db_path.read_bytes() == sentinel


def test_synthetic_benchmark_sidecar_race_cannot_publish_an_unusable_target(tmp_path, monkeypatch):
    db_path = tmp_path / "sidecar-race.db"
    wal_path = Path(f"{db_path}-wal")
    real_link = benchmark_retrieval.os.link

    def racing_link(source, target):
        wal_path.mkdir()
        return real_link(source, target)

    monkeypatch.setattr(benchmark_retrieval.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        run_synthetic_benchmark(db_path)

    assert not db_path.exists()


def test_synthetic_benchmark_copy_failure_never_publishes_a_partial_target(tmp_path, monkeypatch):
    db_path = tmp_path / "partial.db"

    def fail_write(_fd, _data):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(benchmark_retrieval.os, "write", fail_write)
    with pytest.raises(OSError, match="simulated copy failure"):
        run_synthetic_benchmark(db_path)

    assert not db_path.exists()


def test_synthetic_benchmark_generic_link_failure_rolls_back_reservations_and_staging(tmp_path, monkeypatch):
    db_path = tmp_path / "link-failure.db"

    def fail_link(_source, _target):
        raise OSError("simulated hard-link failure")

    monkeypatch.setattr(benchmark_retrieval.os, "link", fail_link)
    with pytest.raises(OSError, match="simulated hard-link failure"):
        run_synthetic_benchmark(db_path)

    assert not db_path.exists()
    assert not Path(f"{db_path}-wal").exists()
    assert not Path(f"{db_path}-shm").exists()
    assert not Path(f"{db_path}-journal").exists()
    assert list(tmp_path.glob(f".{db_path.name}.publish-*")) == []


def test_synthetic_benchmark_restores_a_sidecar_replaced_during_post_link_release(tmp_path, monkeypatch):
    db_path = tmp_path / "post-link-sidecar-race.db"
    wal_path = Path(f"{db_path}-wal")
    sentinel = b"replacement WAL must stay at the requested sidecar path"
    real_rename = benchmark_retrieval.os.rename
    raced = False

    def racing_rename(source, target):
        nonlocal raced
        if Path(source) == wal_path and db_path.exists() and not raced:
            raced = True
            wal_path.unlink()
            wal_path.write_bytes(sentinel)
        return real_rename(source, target)

    monkeypatch.setattr(benchmark_retrieval.os, "rename", racing_rename)
    with pytest.raises(RuntimeError, match="sidecar reservation was replaced"):
        run_synthetic_benchmark(db_path)

    assert raced is True
    assert wal_path.read_bytes() == sentinel
    assert not db_path.exists()
    assert list(tmp_path.glob(f".{db_path.name}.publish-*")) == []


@pytest.mark.parametrize("outcome", ["success", "replacement", "retry", "persistent-failure", "link-failure"])
def test_synthetic_benchmark_pins_reservation_inodes_until_cleanup(tmp_path, monkeypatch, outcome):
    db_path = tmp_path / "pinned.db"
    wal_path = Path(f"{db_path}-wal")
    sidecars = {Path(f"{db_path}{suffix}") for suffix in ("-wal", "-shm", "-journal")}
    real_open = benchmark_retrieval.os.open
    real_close = benchmark_retrieval.os.close
    real_rename = benchmark_retrieval.os.rename
    real_unlink = Path.unlink
    descriptors = {}
    live = set()
    attempts = 0
    sentinel = b"replacement must survive reservation cleanup"

    def track_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) in sidecars:
            descriptors[Path(path)] = fd
            live.add(fd)
        return fd

    def track_close(fd):
        real_close(fd)
        live.discard(fd)

    def assert_pinned(path):
        fd = descriptors[path]
        assert fd in live, "reservation descriptor closed before identity-sensitive cleanup"
        benchmark_retrieval.os.fstat(fd)

    def racing_rename(source, target):
        nonlocal attempts
        source = Path(source)
        if source in sidecars:
            assert_pinned(source)
        if source == wal_path:
            attempts += 1
            if outcome == "persistent-failure" or (outcome == "retry" and attempts == 1):
                raise OSError("injected reservation release failure")
            if outcome == "replacement":
                wal_path.unlink()
                wal_path.write_bytes(sentinel)
                assert_pinned(wal_path)
        return real_rename(source, target)

    def checked_unlink(path, *args, **kwargs):
        if path.name.startswith("reserved-sidecar-") and path.exists():
            identity = path.lstat()
            # Pin through the final identity-sensitive unlink, not just rename.
            assert any(
                fd in live and benchmark_retrieval.os.fstat(fd).st_ino == identity.st_ino
                for fd in descriptors.values()
            ) or (outcome == "replacement" and path.read_bytes() == sentinel)
        return real_unlink(path, *args, **kwargs)

    def fail_link(_source, _target):
        raise OSError("injected link failure")

    monkeypatch.setattr(benchmark_retrieval.os, "open", track_open)
    monkeypatch.setattr(benchmark_retrieval.os, "close", track_close)
    monkeypatch.setattr(benchmark_retrieval.os, "rename", racing_rename)
    monkeypatch.setattr(Path, "unlink", checked_unlink)
    if outcome == "link-failure":
        monkeypatch.setattr(benchmark_retrieval.os, "link", fail_link)
    try:
        if outcome == "success":
            run_synthetic_benchmark(db_path)
            assert db_path.exists()
        else:
            error = RuntimeError if outcome in {"replacement", "persistent-failure"} else OSError
            message = {
                "replacement": "sidecar reservation was replaced",
                "persistent-failure": "rollback could not safely restore",
                "retry": "injected reservation release failure",
                "link-failure": "injected link failure",
            }[outcome]
            with pytest.raises(error, match=message):
                run_synthetic_benchmark(db_path)
            assert not db_path.exists()
        assert len(descriptors) == 3
        assert live == set(), "reservation descriptors leaked after publication"
        if outcome in {"retry", "persistent-failure"}:
            assert attempts == 2
        if outcome == "replacement":
            assert wal_path.read_bytes() == sentinel
        elif outcome != "persistent-failure":
            assert not wal_path.exists()
    finally:
        # Keep a failing regression from leaking descriptors into later tests.
        for fd in live:
            real_close(fd)


def test_synthetic_benchmark_fails_if_published_target_is_replaced_during_sidecar_release(tmp_path, monkeypatch):
    db_path = tmp_path / "post-link-target-race.db"
    wal_path = Path(f"{db_path}-wal")
    sentinel = b"competing target must be preserved"
    real_rename = benchmark_retrieval.os.rename
    raced = False

    def racing_rename(source, target):
        nonlocal raced
        if Path(source) == wal_path and db_path.exists() and not raced:
            raced = True
            db_path.unlink()
            db_path.write_bytes(sentinel)
        return real_rename(source, target)

    monkeypatch.setattr(benchmark_retrieval.os, "rename", racing_rename)
    with pytest.raises(RuntimeError, match="published benchmark database was replaced"):
        run_synthetic_benchmark(db_path)

    assert raced is True
    assert db_path.read_bytes() == sentinel
    assert not wal_path.exists()
    assert not Path(f"{db_path}-shm").exists()
    assert not Path(f"{db_path}-journal").exists()
    assert list(tmp_path.glob(f".{db_path.name}.publish-*")) == []


def test_synthetic_benchmark_refuses_orphan_sidecars_and_broken_symlinks(tmp_path):
    db_path = tmp_path / "sidecar-only.db"
    wal_path = Path(f"{db_path}-wal")
    wal_path.write_bytes(b"orphan WAL sentinel")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_synthetic_benchmark(db_path)
    assert wal_path.read_bytes() == b"orphan WAL sentinel"
    assert not db_path.exists()

    journal_db_path = tmp_path / "journal-only.db"
    journal_path = Path(f"{journal_db_path}-journal")
    journal_path.write_bytes(b"orphan rollback journal sentinel")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_synthetic_benchmark(journal_db_path)
    assert journal_path.read_bytes() == b"orphan rollback journal sentinel"
    assert not journal_db_path.exists()

    link_path = tmp_path / "broken-link.db"
    link_path.symlink_to(tmp_path / "missing-target.db")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_synthetic_benchmark(link_path)
    assert link_path.is_symlink()
    assert not (tmp_path / "missing-target.db").exists()


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
@pytest.mark.parametrize("kind", ["file", "symlink", "broken-symlink"])
def test_synthetic_cli_refuses_existing_main_and_sidecars(tmp_path, suffix, kind):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_retrieval.py"
    db_path = tmp_path / "bench.db"
    protected_path = Path(f"{db_path}{suffix}")
    referent = tmp_path / "referent.db"
    sentinel = b"caller-owned bytes must remain unchanged"
    if kind == "file":
        protected_path.write_bytes(sentinel)
    else:
        if kind == "symlink":
            referent.write_bytes(sentinel)
        protected_path.symlink_to(referent.name)
    before = protected_path.lstat()

    completed = subprocess.run(
        [sys.executable, str(script), "--json", "--db", db_path.name],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode != 0, completed.stdout
    assert "refusing to overwrite" in completed.stderr
    after = protected_path.lstat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    if kind == "file":
        assert protected_path.read_bytes() == sentinel
    else:
        assert protected_path.is_symlink()
        assert protected_path.readlink() == Path(referent.name)
        if kind == "symlink":
            assert referent.read_bytes() == sentinel
        else:
            assert not referent.exists()
            assert not referent.is_symlink()
    for candidate in [db_path, *(Path(f"{db_path}{s}") for s in ("-wal", "-shm", "-journal"))]:
        if candidate != protected_path:
            assert not candidate.exists()
            assert not candidate.is_symlink()
    assert not list(tmp_path.glob(".bench.db.publish-*"))
    assert not list(tmp_path.glob("referent.db-*"))


def test_real_copy_cli_still_resolves_existing_database_symlink(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_retrieval.py"
    db_path = tmp_path / "disposable-copy.db"
    run_synthetic_benchmark(db_path)
    link = tmp_path / "copy-link.db"
    link.symlink_to(db_path.name)

    completed = subprocess.run(
        [sys.executable, str(script), "--mode", "real-copy", "--json",
         "--db", link.name, "--workspace-id", "bench"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )

    payload = json.loads(completed.stdout)
    assert payload["mode"] == "real-copy"
    assert payload["db_path"] == str(db_path.resolve())
    assert payload["metadata"]["cases_built"] > 0
    assert link.is_symlink()
    assert link.readlink() == Path(db_path.name)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_retrieval_benchmark_gate_runs_against_temp_database_without_live_profile_paths(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_retrieval.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--gate", "--json", "--db", str(tmp_path / "bench.db")],
        cwd=script.parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["gate_passed"] is True
    assert payload["db_path"] == str(tmp_path / "bench.db")
    assert "/.hermes/" not in payload["db_path"]
    with sqlite3.connect(payload["db_path"]) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 11
    assert payload["legacy"]["top1_accuracy"] < payload["current"]["top1_accuracy"]
    assert payload["current"]["top1_accuracy"] >= 0.9
    assert payload["current"]["acl_leaks"] == 0
    assert payload["current"]["expired_hits"] == 0
    assert payload["current"]["p95_ms"] < 50
    assert payload["delta"]["top1_accuracy"] > 0
    assert payload["fusion_config"] == {
        "bm25": 0.25,
        "lexical": 0.15,
        "vector": 0.55,
        "importance": 0.03,
        "recency": 0.02,
        "exact_lexical_boost": 0.16,
    }
