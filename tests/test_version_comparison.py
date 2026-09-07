"""Benchmark harness tests; synthetic examples below are FIXTURES, not measurements."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Version-comparison harness requires Python 3.11+ (tomllib); runtime supports 3.10",
                allow_module_level=True)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_version_comparison.py"


def harness():
    assert SCRIPT.is_file(), "Reusable version-comparison harness is not implemented"
    spec = importlib.util.spec_from_file_location("version_comparison_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("suffix", [".md", ".txt", ""])
def test_publication_requires_json_before_worker(tmp_path, monkeypatch, suffix):
    mod = harness()
    monkeypatch.setattr(mod, "checkout_identity", lambda root: {})
    monkeypatch.setattr(mod, "run_worker", lambda *args: pytest.fail("worker started before output preflight"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--baseline-root", str(tmp_path),
                                    "--candidate-root", str(tmp_path / "candidate"),
                                    "--json-output", str(tmp_path / ("report" + suffix)),
                                    "--markdown-output", str(tmp_path / "summary.md")])
    with pytest.raises((ValueError, SystemExit), match="json"):
        mod.main()


@pytest.mark.parametrize("suffix", [".json", ".md"])
@pytest.mark.parametrize("kind", ["file", "symlink", "broken_symlink"])
def test_publication_preserves_existing_targets_before_worker(tmp_path, monkeypatch, suffix, kind):
    mod = harness()
    target = tmp_path / ("report" + suffix)
    original = tmp_path / "original"
    if kind != "broken_symlink":
        original.write_bytes(b"FIXTURE preserve")
    if kind == "file":
        target.write_bytes(b"FIXTURE preserve")
    else:
        target.symlink_to(original)
    monkeypatch.setattr(mod, "checkout_identity", lambda root: {})
    monkeypatch.setattr(mod, "run_worker", lambda *args: pytest.fail("worker started before output preflight"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--baseline-root", str(tmp_path),
                                    "--candidate-root", str(tmp_path / "candidate"),
                                    "--json-output", str(tmp_path / "report.json"),
                                    "--markdown-output", str(tmp_path / "report.md")])
    with pytest.raises((FileExistsError, SystemExit)):
        mod.main()
    if kind == "file":
        assert target.read_bytes() == b"FIXTURE preserve"
    else:
        assert target.is_symlink() and target.readlink() == original
    if kind != "broken_symlink":
        assert original.read_bytes() == b"FIXTURE preserve"


@pytest.mark.parametrize("suffix", [".json", ".md"])
@pytest.mark.parametrize("kind", ["file", "symlink", "broken_symlink"])
def test_atomic_publication_loses_race_without_clobber(tmp_path, monkeypatch, suffix, kind):
    mod = harness()
    output = tmp_path / "report.json"
    original = tmp_path / "original"
    if kind != "broken_symlink":
        original.write_text("FIXTURE racer")
    target = output.with_suffix(suffix)
    real_link = os.link
    def race(src, dst, *args, **kwargs):
        if Path(dst).name == target.name:
            if kind == "file":
                target.write_text("FIXTURE racer")
            else:
                target.symlink_to(original)
        return real_link(src, dst, *args, **kwargs)
    monkeypatch.setattr(mod.os, "link", race)
    with pytest.raises(FileExistsError):
        mod.publish_reports(output, output.with_suffix(".md"), '{"fixture":true}\n', "# FIXTURE\n")
    if kind == "file":
        assert target.read_text() == "FIXTURE racer"
    else:
        assert target.is_symlink() and target.readlink() == original
    if kind != "broken_symlink":
        assert original.read_text() == "FIXTURE racer"


def test_atomic_publication_links_complete_distinct_artifacts(tmp_path, monkeypatch):
    mod = harness()
    output = tmp_path / "nested" / "report.json"
    linked = []
    real_link = os.link
    def inspect_link(src, dst, *args, **kwargs):
        assert Path(src).read_text() in ('{"fixture":true}\n', "# FIXTURE\n")
        linked.append(Path(dst).name)
        return real_link(src, dst, *args, **kwargs)
    monkeypatch.setattr(mod.os, "link", inspect_link)
    mod.publish_reports(output, output.with_suffix(".md"), '{"fixture":true}\n', "# FIXTURE\n")
    assert linked == ["report.md", "report.json"]  # JSON is published last.
    assert json.loads(output.read_text()) == {"fixture": True}
    assert output.with_suffix(".md").read_text() == "# FIXTURE\n"


def test_fixture_ranking_metrics_and_nearest_rank_p95():
    mod = harness()
    assert mod.ranking_metrics([[1, 2], [4, 3], []], [1, 3, 9]) == {
        "queries": 3, "recall_at_1": 1 / 3, "recall_at_5": 2 / 3, "mrr": 0.5,
    }
    assert mod.latency_summary([1, 2, 3, 4, 100]) == {
        "samples": 5, "median_ms": 3, "p95_ms": 100,
    }
    assert mod.latency_summary([]) is None


def test_fixture_schedule_is_reproducible_paired_and_interleaved():
    mod = harness()
    schedule = mod.make_schedule(["baseline", "candidate"], rounds=8, seed=41)
    assert schedule == mod.make_schedule(["baseline", "candidate"], rounds=8, seed=41)
    assert len(schedule) == 8
    assert all(sorted(pair) == ["baseline", "candidate"] for pair in schedule)
    assert len({tuple(pair) for pair in schedule}) == 2


def test_fixture_corpus_is_deterministic_and_covers_acl_partitions():
    mod = harness()
    corpus, queries = mod.synthetic_corpus()
    assert (corpus, queries) == mod.synthetic_corpus()
    assert len(queries) >= 12
    assert {row["partition"] for row in corpus} >= {
        "own_agent", "shared", "own_session", "other_agent", "other_session", "other_workspace",
    }
    assert all(q["expected_key"] in {row["key"] for row in corpus} for q in queries)
    assert all(row["content"].startswith("FIXTURE") for row in corpus)


def test_fixture_sdk_result_shapes_and_error_detection():
    mod = harness()
    class TextFixture:
        text = '{"success": true, "results": []}'
    class SDKFixture:
        isError = True
        content = [TextFixture()]
        structuredContent = {"success": False, "error": {"code": "invalid_arguments"}}
    assert mod.public_observation(([TextFixture()], {"success": True}))["rejected"] is False
    assert mod.public_observation(SDKFixture())["rejected"] is True
    assert mod.public_observation(ValueError("FIXTURE_BACKEND_ERROR_SENTINEL"))["sentinel_exposed"] is True


def boundary_fixture(mod, monkeypatch, behavior="working"):
    """A minimal counted contract fixture; never a performance measurement."""
    from types import SimpleNamespace
    class MCPAccessError(PermissionError):
        pass
    class Core:
        config = {}
        def remember(self, args):
            return {"id": 1}
        def close(self):
            pass
    class Adapter:
        WRITE_TOOLS = {"remember"}
        def __init__(self, core, *, access="read-write"):
            self.core, self.access = core, access
        def call(self, operation, args):
            if behavior == "broken":
                raise RuntimeError("FIXTURE private failure")
            if behavior == "rejected_valid":
                return {"success": False, "error": {"code": "invalid_arguments"}}
            if operation == "remember":
                raise MCPAccessError("FIXTURE read-only denial")
            if isinstance(self.core, mod.FaultInjectionBackend):
                if self.core.fail and behavior == "no_fault_call":
                    return {"success": False, "error": {"code": "backend_error"}}
                if self.core.fail and behavior == "double_fault_call":
                    self.core.calls += 1
                if args != {"query": "FIXTURE"}:
                    return {"success": False, "error": {"code": "invalid_arguments"}}
                return self.core.search(args)
            return {"success": True}
        def close(self):
            pass
    monkeypatch.setattr(mod, "new_core", lambda *args: Core())
    monkeypatch.setattr(mod, "database_digest", lambda core: "FIXTURE digest")
    return SimpleNamespace(MCPAdapter=Adapter, MCPAccessError=MCPAccessError)


@pytest.mark.parametrize("failure", [RuntimeError("FIXTURE"), ImportError("FIXTURE"),
                                     ModuleNotFoundError("FIXTURE dependency", name="fixture_dependency")])
def test_sdk_construction_failure_is_error_not_unavailable(tmp_path, monkeypatch, failure):
    import asyncio
    pytest.importorskip("mcp")
    mod = harness()
    mcp = boundary_fixture(mod, monkeypatch)
    def broken(adapter):
        raise failure
    mcp.build_fastmcp = broken
    result = asyncio.run(mod.mcp_boundary_probe(None, mcp, tmp_path, "sdk"))
    assert result["status"] == "error"
    assert result["error_type"] == type(failure).__name__
    assert "FIXTURE" not in json.dumps(result)


def test_only_absent_top_level_sdk_is_unavailable(tmp_path, monkeypatch):
    import asyncio
    mod = harness()
    mcp = boundary_fixture(mod, monkeypatch)
    mcp.build_fastmcp = lambda adapter: pytest.fail("must not build absent SDK")
    real_find = mod.importlib.util.find_spec
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: None if name == "mcp" else real_find(name))
    assert asyncio.run(mod.mcp_boundary_probe(None, mcp, tmp_path, "sdk"))["status"] == "N/A"


@pytest.mark.parametrize("value", [RuntimeError("invalid_arguments"), ValueError("invalid_arguments"),
                                  KeyError("invalid_arguments"), {"isError": True}, {"success": False}])
def test_arbitrary_failures_never_count_as_contract_rejection(value):
    observed = harness().public_observation(value)
    assert observed["rejected"] is False
    assert observed["status"] == "error"
    assert observed["succeeded"] is False


def test_typed_validation_and_legacy_access_rejections(monkeypatch):
    pytest.importorskip("pydantic")
    from pydantic import BaseModel, ValidationError
    mod = harness()
    mcp = boundary_fixture(mod, monkeypatch)
    class Arguments(BaseModel):
        limit: int
    with pytest.raises(ValidationError) as raised:
        Arguments(limit="FIXTURE not an integer")
    observed = mod.public_observation(raised.value)
    assert observed["rejected"] is True and observed["error_code"] == "invalid_arguments"
    observed = mod.public_observation(mcp.MCPAccessError("FIXTURE"), access_error_type=mcp.MCPAccessError)
    assert observed["rejected"] is True and observed["error_code"] == "forbidden"
    assert observed["status"] != "error"


def test_generic_exception_chained_from_validation_is_not_contract_rejection():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel, ValidationError
    mod = harness()
    class Arguments(BaseModel):
        limit: int
    try:
        Arguments(limit="FIXTURE")
    except ValidationError as validation:
        failure = RuntimeError("FIXTURE construction failed after validation")
        failure.__cause__ = validation
    assert mod.public_observation(failure)["rejected"] is False


def test_invalid_rejection_counts_require_working_valid_control(tmp_path, monkeypatch):
    import asyncio
    mod = harness()
    mcp = boundary_fixture(mod, monkeypatch, "rejected_valid")
    result = asyncio.run(mod.mcp_boundary_probe(None, mcp, tmp_path, "adapter"))
    assert not any(case["rejected_before_backend"] for case in result["invalid_inputs"].values())


@pytest.mark.parametrize("behavior", ["broken", "rejected_valid", "no_fault_call", "double_fault_call"])
def test_broken_boundary_or_failed_valid_control_is_error(tmp_path, monkeypatch, behavior):
    import asyncio
    mod = harness()
    mcp = boundary_fixture(mod, monkeypatch, behavior)
    result = asyncio.run(mod.mcp_boundary_probe(None, mcp, tmp_path, "adapter"))
    assert result["status"] == "error"
    assert result["errors"]
    if behavior == "broken":
        assert not any(v["rejected_before_backend"] for v in result["invalid_inputs"].values())
        assert not result["read_only"]["all_reads_succeeded"]
    if behavior in {"broken", "rejected_valid"}:
        assert result["valid_control"]["succeeded"] is False
    else:
        assert result["backend_error"]["status"] == "error"


def test_legitimate_legacy_denial_and_injected_exception_are_measured(tmp_path, monkeypatch):
    import asyncio
    mod = harness()
    mcp = boundary_fixture(mod, monkeypatch)
    result = asyncio.run(mod.mcp_boundary_probe(None, mcp, tmp_path, "adapter"))
    assert result["status"] == "measured"
    assert result["valid_control"]["succeeded"] is True
    assert result["valid_control"]["backend_calls"] == 1
    assert result["backend_error"]["backend_calls"] == 1
    assert result["backend_error"]["sentinel_exposed"] is True
    assert result["read_only"]["write_rejected"] is True


def test_sdk_typed_validation_is_not_confused_with_generic_tool_error(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    pytest.importorskip("mcp")

    from mcp import types
    from mcp.server.fastmcp import FastMCP
    mod = harness()
    server = FastMCP("FIXTURE")
    @server.tool()
    def search(limit: int):
        raise RuntimeError("invalid_arguments")
    async def check():
        invalid = await mod.invoke_mcp(None, server, "search", {"limit": "FIXTURE"}, SimpleNamespace(), types)
        broken = await mod.invoke_mcp(None, server, "search", {"limit": 1}, SimpleNamespace(), types)
        assert invalid["rejected"] is True
        assert invalid["error_code"] == "invalid_arguments"
        assert broken["rejected"] is False and broken["status"] == "error"
    asyncio.run(check())


def test_report_surfaces_errors_and_failed_valid_controls_across_rounds(tmp_path, monkeypatch):
    import asyncio
    mod = harness()
    mcp = boundary_fixture(mod, monkeypatch, "broken")
    data = asyncio.run(mod.mcp_boundary_probe(None, mcp, tmp_path, "adapter"))
    report = {"parameters": {"rounds": 2, "warmups": 1, "samples": 1, "seed": 41},
              "summary": {"baseline": {"ranking": {"recall_at_1": 1, "recall_at_5": 1, "mrr": 1},
                  "search_latency": {"median_ms": 1, "p95_ms": 1},
                  "write_latency": {"median_ms": 1, "p95_ms": 1},
                  "cross_agent_search_leaks": 0, "cross_agent_direct_id_leaks": 0}},
              "runs": [{"round": 0, "label": "baseline", "measurement": {"mcp": mod.unavailable("fixture")}},
                       {"round": 1, "label": "baseline", "measurement": {"mcp": {"adapter": data,
                        "sdk": {"status": "error", "error_type": "RuntimeError", "reason": "SDK construction failed"}}}}]}
    text = mod.markdown_report(report)
    assert "ERROR" in text and "RuntimeError" in text
    assert "valid control succeeded=False" in text and "all reads succeeded=False" in text
    assert "errors=" in text


def test_fixture_environment_drops_secrets_and_redirects_home(tmp_path, monkeypatch):
    mod = harness()
    monkeypatch.setenv("OPENAI_API_KEY", "FIXTURE_SECRET_NOT_REAL")
    monkeypatch.setenv("AGENT_RECALL_DB_PATH", "/fixture/not-used.db")
    env = mod.worker_environment(tmp_path)
    assert "OPENAI_API_KEY" not in env
    assert "AGENT_RECALL_DB_PATH" not in env
    assert env["HOME"] == str(tmp_path)
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_fixture_sandbox_blocks_network_live_database_and_config(tmp_path):
    harness()
    (tmp_path / "fixture-code").mkdir()
    (tmp_path / "fixture-code" / "config.py").write_text("# FIXTURE library source, not application state\n")
    code = f'''import importlib.util, pathlib, socket, sqlite3
spec = importlib.util.spec_from_file_location("fixture_harness", {str(SCRIPT)!r})
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.install_sandbox(pathlib.Path({str(tmp_path / 'sandbox')!r}))
pathlib.Path({str(tmp_path / 'sandbox')!r}).mkdir()
assert pathlib.Path({str(tmp_path / 'fixture-code' / 'config.py')!r}).read_text().startswith("# FIXTURE")
checks = [lambda: socket.create_connection(("127.0.0.1", 9)),
          lambda: sqlite3.connect("/tmp/FIXTURE_FORBIDDEN_LIVE.db"),
          lambda: open("/tmp/FIXTURE_FORBIDDEN/config.yaml")]
for fn in checks:
    try: fn()
    except PermissionError: pass
    else: raise AssertionError("sandbox did not deny forbidden effect")
with sqlite3.connect({str(tmp_path / 'sandbox' / 'fixture.db')!r}) as db:
    db.execute("CREATE TABLE fixture (id INTEGER)")
print("fixture-sandbox-ok")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fixture-sandbox-ok"


def test_fixture_cli_rejects_unbounded_sample_count(tmp_path):
    harness()
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--baseline-root", str(tmp_path),
                             "--candidate-root", str(tmp_path / "candidate"), "--samples", "10000000",
                             "--json-output", str(tmp_path / "fixture.json"),
                             "--markdown-output", str(tmp_path / "fixture.md")],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "samples" in result.stderr


def test_fixture_worker_failure_has_safe_diagnostic_not_error_payload(tmp_path):
    mod = harness()
    (tmp_path / "agent_recall_core.py").write_text('raise ValueError("FIXTURE_SECRET_NOT_REAL")\n')
    with pytest.raises(RuntimeError) as raised:
        mod.run_worker(tmp_path, warmups=1, samples=1, seed=41)
    assert "ValueError" in str(raised.value)
    assert "FIXTURE_SECRET_NOT_REAL" not in str(raised.value)


def test_stale_unchecked_pyc_cannot_override_fingerprinted_source(tmp_path):
    import hashlib
    import py_compile
    mod = harness()
    source = tmp_path / "agent_recall_core.py"
    source.write_text('raise RuntimeError("FIXTURE stale bytecode")\n')
    cached = Path(py_compile.compile(str(source), invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH))
    cache_hash = hashlib.sha256(cached.read_bytes()).hexdigest()
    source.write_text('raise ValueError("FIXTURE current source")\n')
    with pytest.raises(RuntimeError) as raised:
        mod.run_worker(tmp_path, warmups=1, samples=1, seed=41)
    assert "ValueError" in str(raised.value)
    assert hashlib.sha256(cached.read_bytes()).hexdigest() == cache_hash


def test_worker_cache_prefix_is_empty_before_source_import(tmp_path):
    mod = harness()
    (tmp_path / "agent_recall_core.py").write_text('''import pathlib, sys
assert sys.pycache_prefix, "FIXTURE missing cache prefix"
prefix = pathlib.Path(sys.pycache_prefix)
assert prefix.is_dir() and not list(prefix.iterdir()), "FIXTURE cache not empty"
assert not prefix.is_relative_to(pathlib.Path(__file__).parent), "FIXTURE root cache"
raise ValueError("FIXTURE verified startup cache")
''')
    with pytest.raises(RuntimeError) as raised:
        mod.run_worker(tmp_path, warmups=1, samples=1, seed=41)
    assert "ValueError" in str(raised.value)


@pytest.mark.parametrize("phase", ["import", "measurement"])
def test_worker_checks_source_manifest_before_import_and_after(tmp_path, monkeypatch, phase):
    from types import SimpleNamespace
    mod = harness()
    source = tmp_path / "agent_recall_core.py"
    source.write_text("# FIXTURE initial source\n")
    before = {"agent_recall_core": {"file": str(source), "sha256": "fixture"}, "agent_recall_store": {}}
    monkeypatch.setattr(mod, "install_sandbox", lambda home: {})
    monkeypatch.setattr(mod, "module_fingerprints", lambda root: before)
    monkeypatch.setattr(sys, "path", list(sys.path))
    prefix = tmp_path / "home" / "pycache"
    prefix.mkdir(parents=True)
    monkeypatch.setattr(sys, "pycache_prefix", str(prefix))
    def load(name):
        if phase == "import":
            source.write_text("# FIXTURE changed during import\n")
        return SimpleNamespace()
    monkeypatch.setattr(mod.importlib, "import_module", load)
    def retrieval(*args):
        if phase == "import":
            pytest.fail("measurement started despite source changing during import")
        source.write_text("# FIXTURE changed during measurement\n")
        return {}
    monkeypatch.setattr(mod, "retrieval_probe", retrieval)
    with pytest.raises(RuntimeError, match="[Ss]ource"):
        mod.worker(tmp_path, tmp_path / "home", 1, 1, 41)


def test_parent_checks_source_again_before_publication(tmp_path, monkeypatch):
    mod = harness()
    source = tmp_path / "agent_recall_core.py"
    source.write_text("# FIXTURE initial\n")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    monkeypatch.setattr(mod, "checkout_identity", lambda root: {})
    def changed(root, *args):
        manifest = mod.source_manifest(root)
        if root == tmp_path:
            source.write_text("# FIXTURE changed after worker\n")
        return {"identity": {"source_manifest": manifest}}
    monkeypatch.setattr(mod, "run_worker", changed)
    monkeypatch.setattr(mod, "aggregate", lambda runs: pytest.fail("source mismatch reached report construction"))
    output = tmp_path / "report.json"
    markdown_output = tmp_path / "summary.md"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--baseline-root", str(tmp_path),
                                    "--candidate-root", str(candidate), "--rounds", "1",
                                    "--json-output", str(output),
                                    "--markdown-output", str(markdown_output)])
    with pytest.raises(RuntimeError, match="[Ss]ource"):
        mod.main()
    assert not output.exists() and not markdown_output.exists()


def test_new_core_uses_only_cross_version_safe_configuration(tmp_path):
    from types import SimpleNamespace
    mod = harness()
    captured = {}
    def core(config, identity):
        captured.update(config)
        return SimpleNamespace()
    mod.new_core(SimpleNamespace(AgentRecallCore=core, AgentIdentity=lambda *args: None), tmp_path / "fixture.db")
    assert captured == {
        "db_path": str(tmp_path / "fixture.db"),
        "embedding_base_url": "",
        "embedding_model": "",
        "embedding_api_key_env": "",
    }


def test_benchmark_module_skips_on_python310_before_tomllib():
    code = f'''import runpy, sys, pytest
sys.version_info = (3, 10, 0)
try:
    runpy.run_path({str(Path(__file__).resolve())!r})
except pytest.skip.Exception:
    print("fixture-skipped")
else:
    raise AssertionError("benchmark test module did not skip Python 3.10")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fixture-skipped"


def test_main_propagates_boundary_errors_to_report_and_exit(tmp_path, monkeypatch, capsys):
    mod = harness()
    monkeypatch.setattr(mod, "checkout_identity", lambda root: {})
    measurement = {"identity": {"source_manifest": {}, "modules": {}}, "retrieval": {
        "corpus_sha256": "FIXTURE", "ranking": {"recall_at_1": 1, "recall_at_5": 1, "mrr": 1},
        "search_ms": [1], "write_ms": [1],
        "acl": {"cross_agent_search_leaks": 0, "cross_agent_direct_id_leaks": 0}},
        "mcp": {"sdk": {"status": "error", "reason": "SDK construction failed", "error_type": "RuntimeError"}}}
    monkeypatch.setattr(mod, "run_worker", lambda *args: measurement)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    output = tmp_path / "report.json"
    markdown_output = tmp_path / "summary.md"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--baseline-root", str(tmp_path),
                                    "--candidate-root", str(candidate), "--rounds", "1",
                                    "--json-output", str(output),
                                    "--markdown-output", str(markdown_output)])
    assert mod.main() == 1
    report = json.loads(output.read_text())
    assert report["status"] == "error" and report["summary"]["baseline"]["status"] == "error"
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert "**ERROR**" in markdown_output.read_text()


def test_source_change_during_report_rendering_prevents_publication(tmp_path, monkeypatch):
    mod = harness()
    monkeypatch.setattr(mod, "checkout_identity", lambda root: {})
    monkeypatch.setattr(mod, "run_worker", lambda *args: {"identity": {"source_manifest": {}}})
    monkeypatch.setattr(mod, "aggregate", lambda runs: {
        "baseline": {"status": "measured"},
        "candidate": {"status": "measured"},
    })
    def changed(report):
        (tmp_path / "agent_recall_late.py").write_text("# FIXTURE late addition\n")
        return "# FIXTURE\n"
    monkeypatch.setattr(mod, "markdown_report", changed)
    output = tmp_path / "report.json"
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--baseline-root", str(tmp_path),
                                    "--candidate-root", str(tmp_path / "candidate"),
                                    "--rounds", "1", "--json-output", str(output),
                                    "--markdown-output", str(summary)])
    with pytest.raises(RuntimeError, match="[Ss]ource"):
        mod.main()
    assert not output.exists() and not summary.exists()


def test_fixture_metrics_reject_mismatched_query_counts():
    with pytest.raises(ValueError):
        harness().ranking_metrics([[1]], [1, 2])


def test_fixture_aggregate_rejects_different_version_corpora():
    mod = harness()
    def fixture(label, fingerprint):
        return {"label": label, "measurement": {"identity": {"modules": {}}, "retrieval": {
            "corpus_sha256": fingerprint, "ranking": {"mrr": 1.0}, "search_ms": [1], "write_ms": [2],
            "acl": {"cross_agent_search_leaks": 0, "cross_agent_direct_id_leaks": 0}}}}
    with pytest.raises(RuntimeError, match="corpus"):
        mod.aggregate([fixture("baseline", "fixture-hash-a"), fixture("candidate", "fixture-hash-b")])


def test_fixture_report_explains_absent_mcp_without_zero_score():
    mod = harness()
    report = {"parameters": {"rounds": 1, "warmups": 1, "samples": 1, "seed": 41},
              "summary": {"baseline": {"ranking": {"recall_at_1": 1, "recall_at_5": 1, "mrr": 1},
                  "search_latency": {"median_ms": 1, "p95_ms": 1},
                  "write_latency": {"median_ms": 1, "p95_ms": 1},
                  "cross_agent_search_leaks": 0, "cross_agent_direct_id_leaks": 0}},
              "runs": [{"label": "baseline", "measurement": {"mcp": mod.unavailable("fixture MCP absent")}}]}
    assert "N/A (fixture MCP absent)" in mod.markdown_report(report)


@pytest.mark.skipif(not os.environ.get("AGENT_RECALL_BENCHMARK_BASELINE_TEST"),
                    reason="opt-in actual-root baseline smoke; fixture unit tests do not represent performance")
def test_actual_baseline_worker_imports_exact_root_and_uses_sqlite(tmp_path):
    mod = harness()
    root = Path(os.environ["AGENT_RECALL_BENCHMARK_BASELINE_TEST"]).resolve()
    result = mod.run_worker(root, warmups=1, samples=2, seed=41)
    assert Path(result["identity"]["modules"]["agent_recall_core"]["file"]).parent == root
    assert result["retrieval"]["backend"] == "real AgentRecallCore + SQLite; embeddings disabled; lexical fallback"
    assert result["retrieval"]["search_latency"]["samples"] == 2
    assert result["retrieval"]["write_latency"]["samples"] == 2
    assert result["retrieval"]["ranking"]["queries"] >= 12
    assert result["retrieval"]["ranking_depth"] == 50
    assert result["isolation"]["network_attempts"] == 0
    assert result["mcp"]["adapter"]["read_only"]["database_unchanged"] is True
    assert result["mcp"]["sdk"]["status"] == "measured"
    assert result["mcp"]["sdk"]["read_only"]["all_reads_succeeded"] is True
    serialized = json.dumps(result, allow_nan=False)
    assert "FIXTURE_BACKEND_ERROR_SENTINEL" not in serialized
    assert "FIXTURE_INTERNAL_SESSION_SENTINEL" not in serialized


def test_publication_uses_two_explicit_caller_paths(tmp_path):
    mod = harness()
    json_output = tmp_path / "json" / "comparison.json"
    markdown_output = tmp_path / "notes" / "comparison-summary.md"
    mod.publish_reports(json_output, markdown_output, '{"fixture":true}\n', "# FIXTURE\n")
    assert json.loads(json_output.read_text()) == {"fixture": True}
    assert markdown_output.read_text() == "# FIXTURE\n"
    assert not json_output.with_suffix(".md").exists()


def test_readme_has_no_retired_learning_material_and_documents_explicit_outputs():
    text = (SCRIPT.parents[1] / "benchmarks/version_comparison/README.md").read_text(encoding="utf-8").lower()
    for retired in ("automatic-learning", "held-out", "heldout", "session-end learning",
                    "session_end_learning", "session_learning"):
        assert retired not in text
    assert "--json-output" in text
    assert "--markdown-output" in text
