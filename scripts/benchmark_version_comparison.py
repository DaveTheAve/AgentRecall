#!/usr/bin/env python3
"""Actual-checkout lexical/ACL and MCP comparison, never a legacy simulation.

Each scheduled batch imports one exact root in an isolated subprocess and uses
only a disposable synthetic SQLite database. See benchmarks/version_comparison/.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import tomllib

ERROR_SENTINEL = "FIXTURE_BACKEND_ERROR_SENTINEL"
SESSION_SENTINEL = "FIXTURE_INTERNAL_SESSION_SENTINEL"
MAX_OUTPUT_BYTES = 2_000_000


def ranking_metrics(rankings, expected):
    if len(rankings) != len(expected) or not expected:
        raise ValueError("Expected one nonempty query specification per ranking")
    ranks = [rows.index(want) + 1 if want in rows else None for rows, want in zip(rankings, expected, strict=True)]
    n = len(ranks)
    return {"queries": n, "recall_at_1": sum(r == 1 for r in ranks) / n,
            "recall_at_5": sum(r is not None and r <= 5 for r in ranks) / n,
            "mrr": sum(1 / r if r else 0 for r in ranks) / n}


def latency_summary(values):
    if not values:
        return None
    return {"samples": len(values), "median_ms": statistics.median(values),
            "p95_ms": sorted(values)[math.ceil(0.95 * len(values)) - 1]}


def make_schedule(labels, rounds, seed):
    rng = random.Random(seed)
    result = []
    for _ in range(rounds):
        pair = list(labels)
        rng.shuffle(pair)
        result.append(pair)
    return result


def synthetic_corpus():
    """Public, intentionally simple FIXTURE corpus; no heldout or live data."""
    rows, queries = [], []
    partitions = {
        "own_agent": ("fixture-workspace", "alpha", "s1", "agent"),
        "shared": ("fixture-workspace", "beta", "s2", "shared"),
        "own_session": ("fixture-workspace", "alpha", "s1", "session"),
        "other_agent": ("fixture-workspace", "beta", "s2", "agent"),
        # Same owner, different session and same session, different owner.
        "other_session": ("fixture-workspace", "alpha", "s2", "session"),
        "other_agent_session": ("fixture-workspace", "beta", "s1", "session"),
        "other_workspace": ("fixture-elsewhere", "alpha", "s1", "shared"),
    }
    for i in range(24):
        query = f"topic{i:03d} answer{i:03d}"
        visible = ["own_agent", "shared", "own_session"][i % 3]
        for partition in [visible, "other_agent", "other_session", "other_agent_session", "other_workspace"]:
            key = f"q{i}-{partition}"
            rows.append({"key": key, "partition": partition, "identity": partitions[partition][:3],
                         "visibility": partitions[partition][3], "content": f"FIXTURE {query} durable project instruction",
                         "importance": 0.5 if partition == visible else 1.0})
        queries.append({"query": query, "expected_key": f"q{i}-{visible}"})
        for d in range(4):
            rows.append({"key": f"d{i}-{d}", "partition": "own_agent", "identity": partitions["own_agent"][:3],
                         "visibility": "agent", "content": f"FIXTURE topic{i:03d} unrelated distractor{d}",
                         "importance": 0.5})
    return rows, queries


def worker_environment(home):
    # Deliberately do not inherit provider keys, PYTHONPATH, host/profile config,
    # proxies, or application environment. Python executable is passed explicitly.
    return {"PATH": os.defpath, "HOME": str(home), "TMPDIR": str(home),
            "XDG_CONFIG_HOME": str(home / "config"), "XDG_CACHE_HOME": str(home / "cache"),
            "HERMES_HOME": str(home / "hermes"), "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0", "LANG": "C.UTF-8", "TZ": "UTC"}


def install_sandbox(home):
    """Fail closed on Python network/process and non-disposable DB/config I/O.

    Defense in depth for trusted source, not an OS sandbox for hostile code.
    asyncio's local socketpair is allowed; outbound connects/binds/DNS are not.
    """
    home = home.resolve()
    counters = {"network_attempts": 0, "blocked_state_accesses": 0}

    def inside(value):
        return Path(os.fsdecode(value)).resolve().is_relative_to(home)

    def audit(event, args):
        if event in {"socket.connect", "socket.connect_ex", "socket.bind", "socket.getaddrinfo",
                     "socket.gethostbyname", "socket.gethostbyaddr", "socket.sendto", "socket.sendmsg"}:
            counters["network_attempts"] += 1
            raise PermissionError("Benchmark network disabled")
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "os.fork"}:
            raise PermissionError("Benchmark child processes disabled")
        if event == "sqlite3.connect" and (str(args[0]) == ":memory:" or not inside(args[0])):
            counters["blocked_state_accesses"] += 1
            raise PermissionError("Only disposable benchmark SQLite files are allowed")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            sensitive = path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".yaml", ".yml", ".env"} or (
                path.name.startswith(("config.", "auth.", ".env")) and path.suffix.lower() not in {".py", ".pyc", ".so"})
            if (writing or sensitive) and not inside(path):
                counters["blocked_state_accesses"] += 1
                raise PermissionError("Benchmark non-disposable state access disabled")
    sys.addaudithook(audit)
    return counters


def public_observation(value, *, access_error_type=(), public_error_type=(), diagnostic=None):
    """Normalize legacy FastMCP tuples and current MCP ServerResult envelopes.

    Return only bounded booleans; never emit raw backend text or public payloads.
    """
    def plain(item):
        if isinstance(item, BaseException):
            return {"isError": True, "exception": str(item)}
        if hasattr(item, "model_dump"):
            return plain(item.model_dump(mode="json"))
        if hasattr(item, "isError"):
            return {"isError": item.isError, "content": plain(item.content),
                    "structuredContent": plain(getattr(item, "structuredContent", None))}
        if isinstance(item, dict):
            return {str(k): plain(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(v) for v in item]
        if hasattr(item, "text"):
            return plain(item.text)
        if isinstance(item, str):
            try:
                return plain(json.loads(item))
            except (ValueError, TypeError):
                return item
        return item

    normalized = plain(value)

    safe_codes = {"invalid_arguments", "forbidden", "backend_error", "not_found", "busy", "timeout", "closed"}

    def typed_code(exc):
        seen = set()
        while isinstance(exc, BaseException) and id(exc) not in seen:
            seen.add(id(exc))
            if isinstance(exc, access_error_type):
                return "forbidden"
            if isinstance(exc, public_error_type) and getattr(exc, "code", None) in safe_codes:
                return exc.code
            for name in ("pydantic", "pydantic.v1", "jsonschema"):
                module = sys.modules.get(name)
                cls = getattr(module, "ValidationError", ())
                if isinstance(exc, cls):
                    return "invalid_arguments"
            # Only the SDK's known wrapper preserves contract provenance.
            # An arbitrary RuntimeError caused by ValidationError is still a
            # runtime failure, not an argument rejection.
            wrapper = tuple(cls for module in ("mcp.server.fastmcp.exceptions", "mcp.server.mcpserver.exceptions")
                            if isinstance(cls := getattr(sys.modules.get(module), "ToolError", None), type))
            if not isinstance(exc, wrapper):
                break
            exc = exc.__cause__
        return None

    def flags(item):
        if isinstance(item, dict):
            yield item
            for child in item.values():
                yield from flags(child)
        elif isinstance(item, list):
            for child in item:
                yield from flags(child)

    objects = list(flags(normalized))
    failed = isinstance(value, BaseException) or any(o.get("isError") is True or o.get("success") is False for o in objects)
    code = typed_code(value) or typed_code(diagnostic)
    if code is None and failed:
        code = next((o["error"]["code"] for o in objects if isinstance(o.get("error"), dict)
                     and o["error"].get("code") in safe_codes), None)
    rejected = failed and code in {"invalid_arguments", "forbidden"}
    succeeded = not failed and any(o.get("success") is True for o in objects)
    text = json.dumps(normalized, default=str)
    return {"rejected": rejected, "succeeded": succeeded, "failed": failed,
            "status": "measured" if rejected or succeeded else "error", "error_code": code,
            "error_type": type(value if isinstance(value, BaseException) else diagnostic).__name__ if isinstance(value, BaseException) or diagnostic is not None else None,
            "sentinel_exposed": ERROR_SENTINEL in text,
            "internal_session_exposed": SESSION_SENTINEL in text}


async def invoke_mcp(adapter, sdk, operation, arguments, mcp_module, types=None):
    """Observe the real handler, retaining typed errors before SDK flattening.

    Legacy SDK handlers stringify ValidationError/ToolError to opaque text. The
    local observer records the active exception, never changes the returned
    response and never classifies text resembling a validation message.
    """
    diagnostic = None
    sdk2 = sdk is not None and hasattr(sdk, "_lowlevel_server")
    low = (sdk._lowlevel_server if sdk2 else sdk._mcp_server) if sdk is not None else None
    original = getattr(low, "_make_error_result", None)
    had_own = low is not None and "_make_error_result" in vars(low)
    def observe_error(*args, **kwargs):
        nonlocal diagnostic
        diagnostic = sys.exc_info()[1]
        return original(*args, **kwargs)
    try:
        if low is None:
            result = adapter.call(operation, arguments)
        elif sdk2:
            # SDK 2's high-level call preserves typed exceptions, unlike the
            # legacy low-level handler's flattening path observed below.
            result = await sdk.call_tool(operation, arguments)
        else:
            if original is not None:
                low._make_error_result = observe_error
            request = types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(
                name=operation, arguments=arguments))
            result = await low.request_handlers[types.CallToolRequest](request)
    except Exception as exc:
        result = exc
    finally:
        if original is not None:
            if had_own:
                low._make_error_result = original
            else:
                del low._make_error_result
    return public_observation(result, access_error_type=getattr(mcp_module, "MCPAccessError", ()),
                              public_error_type=getattr(mcp_module, "MCPPublicError", ()), diagnostic=diagnostic)


def new_core(module, database):
    core = module.AgentRecallCore({"db_path": str(database), "embedding_base_url": "", "embedding_model": "",
                                   "embedding_api_key_env": ""},
                                  module.AgentIdentity("fixture-workspace", "alpha", "s1"))
    core.embedder = None  # Explicitly lexical fallback, NOT semantic retrieval.
    return core


def database_digest(core):
    # serialize reads actual pages without querying legacy external-content FTS
    # column aliases (which need not match the memories table's JSON columns).
    return hashlib.sha256(core.store.conn.serialize()).hexdigest()


def elapsed(fn):
    start = time.perf_counter_ns()
    value = fn()
    return (time.perf_counter_ns() - start) / 1_000_000, value


def retrieval_probe(module, home, warmups, samples, seed):
    corpus, queries = synthetic_corpus()
    core = new_core(module, home / "retrieval.db")
    writer = new_core(module, home / "writes.db")
    ids, forbidden = {}, set()
    try:
        for row in corpus:
            result = core.remember({k: row[k] for k in ("content", "visibility", "importance")},
                                   identity=module.AgentIdentity(*row["identity"]))
            ids[row["key"]] = result["id"]
            if row["partition"].startswith("other_"):
                forbidden.add(result["id"])
        # Normalize timestamps to make ranking independent of machine/seed speed.
        core.store.conn.execute("UPDATE memories SET created_at=?, updated_at=?", (1_700_000_000.0, 1_700_000_000.0))
        core.store.conn.commit()
        # Broader retrieval ACL probes plus direct-id access, not just top-5.
        acl_rankings = [[r["id"] for r in core.search({"query": q["query"], "limit": 50,
                                                       "_track_access": False})["results"]] for q in queries]
        acl_leaks = sum(mid in forbidden for rows in acl_rankings for mid in rows)
        direct_leaks = sum(core.get_memory(mid).get("success") is True for mid in forbidden)
        expected = [ids[q["expected_key"]] for q in queries]
        rng = random.Random(seed)
        query_order = [rng.randrange(len(queries)) for _ in range(warmups + samples)]
        search_times, write_times = [], []
        write_adapter_module = importlib.import_module("agent_recall_mcp") if (Path(module.__file__).parent / "agent_recall_mcp.py").is_file() else None
        write_adapter = write_adapter_module.MCPAdapter(writer, access="read-write") if write_adapter_module else None
        for i, query_index in enumerate(query_order):
            def search(query_index=query_index):
                return core.search({"query": queries[query_index]["query"], "limit": 5, "_track_access": False})
            def write(i=i):
                args = {"content": f"FIXTURE latency write number {i}", "visibility": "agent"}
                return write_adapter.call("remember", args) if write_adapter else writer.remember(args)
            # Alternate operation order; version ordering happens in the parent.
            operations = [(search, search_times), (write, write_times)]
            if i % 2:
                operations.reverse()
            for operation, values in operations:
                duration, value = elapsed(operation)
                if value.get("success") is not True:
                    raise RuntimeError("Real SQLite benchmark operation failed")
                if i >= warmups:
                    values.append(duration)
        count = writer.store.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count != warmups + samples:
            raise RuntimeError("Write benchmark persistence verification failed")
        return {"backend": "real AgentRecallCore + SQLite; embeddings disabled; lexical fallback",
                "write_boundary": "MCPAdapter explicit read-write" if write_adapter else "core (MCP unavailable)",
                "corpus_sha256": hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest(),
                "corpus_rows": len(corpus), "ranking": ranking_metrics(acl_rankings, expected), "ranking_depth": 50,
                "acl": {"search_queries": len(acl_rankings), "search_limit": 50,
                        "cross_agent_search_leaks": acl_leaks, "direct_id_probes": len(forbidden),
                        "cross_agent_direct_id_leaks": direct_leaks},
                "warmups_per_operation": warmups, "query_order_including_warmups": query_order,
                "search_ms": search_times, "write_ms": write_times,
                "search_latency": latency_summary(search_times), "write_latency": latency_summary(write_times),
                "verified_write_rows": count}
    finally:
        core.close()
        writer.close()


class FaultInjectionBackend:
    """FIXTURE only: count validation admission / deliberately throw an error.

    Retrieval, metadata output and read-only durability use the real core instead.
    """
    def __init__(self, core):
        self.real = core
        self.config = core.config
        self.calls = 0
        self.fail = False

    def health(self):
        return self.real.health()

    def search(self, args):
        self.calls += 1
        if self.fail:
            raise RuntimeError(ERROR_SENTINEL)
        return {"success": True, "count": 0, "results": []}

    def close(self):
        pass


def unavailable(reason):
    return {"status": "N/A", "reason": reason}


async def mcp_boundary_probe(core_module, mcp_module, home, boundary):
    core = new_core(core_module, home / f"mcp-{boundary}.db")
    adapters = []
    try:
        memory_id = core.remember({"content": "FIXTURE metadata probe", "visibility": "agent",
                                  "metadata": {"nested": {"internal_session": {"archive_id": SESSION_SENTINEL,
                                                                              "session_id": SESSION_SENTINEL}}}})["id"]
        fault = FaultInjectionBackend(core)
        fault_adapter = mcp_module.MCPAdapter(fault, access="read-write")
        real_adapter = mcp_module.MCPAdapter(core, access="read-only")
        adapters.extend([fault_adapter, real_adapter])
        server, fault_server = None, None
        types = None
        if boundary == "sdk":
            if not hasattr(mcp_module, "build_fastmcp"):
                return unavailable("MCP SDK factory absent")
            try:
                if importlib.util.find_spec("mcp") is None:
                    return unavailable("Optional MCP SDK not installed")
                from mcp import types
                server = mcp_module.build_fastmcp(real_adapter)
                fault_server = mcp_module.build_fastmcp(fault_adapter)
            except Exception as exc:
                return {"status": "error", "reason": "MCP SDK import or construction failed",
                        "error_type": type(exc).__name__}

        async def invoke(adapter, sdk, operation, arguments):
            return await invoke_mcp(adapter, sdk, operation, arguments, mcp_module, types)

        errors = []
        before = fault.calls
        valid = await invoke(fault_adapter, fault_server, "search", {"query": "FIXTURE"})
        valid["backend_calls"] = fault.calls - before
        valid_control_ok = valid["succeeded"] and valid["backend_calls"] == 1
        if not valid_control_ok:
            errors.append("valid_control_failed")
        cases = {
            "string_integer": {"query": "FIXTURE", "limit": "3"},
            "bool_integer": {"query": "FIXTURE", "limit": True},
            "nan_number": {"query": "FIXTURE", "min_importance": float("nan")},
            "wrong_key": {"query": "FIXTURE", "unknown_fixture_key": "fixture"},
            "string_boolean": {"query": "FIXTURE", "include_shared": "false"},
            "boolean_query": {"query": True},
        }
        invalid = {}
        for name, args in cases.items():
            before = fault.calls
            start = time.perf_counter_ns()
            observed = await invoke(fault_adapter, fault_server, "search", args)
            invalid[name] = {**observed, "rejected_before_backend": valid_control_ok and observed["rejected"] and observed["error_code"] == "invalid_arguments" and fault.calls == before,
                             "backend_calls": fault.calls - before,
                             "admission_ms": (time.perf_counter_ns() - start) / 1_000_000}
            if observed["status"] == "error":
                errors.append(f"invalid_input:{name}:unexpected_error")
        fault.fail = True
        before = fault.calls
        error = await invoke(fault_adapter, fault_server, "search", {"query": "FIXTURE"})
        error["backend_calls"] = fault.calls - before
        injected = error["backend_calls"] == 1 and error["failed"] and not error["rejected"]
        error["status"] = "measured fault injection" if injected else "error"
        if not injected:
            errors.append("backend_error_injection_failed")
        fault.fail = False
        digest = database_digest(core)
        read_results = {}
        for operation, args in [("search", {"query": "FIXTURE"}),
                                ("get_memory", {"id": memory_id}),
                                ("prefetch_context", {"query": "FIXTURE", "include_results": True}),
                                ("profile", {"focus": "FIXTURE"})]:
            read_results[operation] = await invoke(real_adapter, server, operation, args)
            if not read_results[operation]["succeeded"]:
                errors.append(f"valid_read:{operation}:failed")
        denied = await invoke(real_adapter, server, "remember", {"content": "FIXTURE forbidden write"})
        if denied["status"] == "error":
            errors.append("read_only_denial:unexpected_error")
        advertised = None
        if server is not None:
            advertised = [tool.name for tool in await server.list_tools()]
        default = inspect.signature(mcp_module.MCPAdapter).parameters["access"].default
        return {"status": "error" if errors else "measured", "errors": errors, "valid_control": valid,
                "boundary": "MCP registered tools/call handler (in-process; no wire transport)" if server else "MCPAdapter.call",
                "invalid_inputs": invalid, "invalid_input_method": "fault-injection counting backend; single admission timings, not throughput",
                "backend_error": error, "metadata_reads": read_results,
                "read_only": {"write_rejected": denied["rejected"] and denied["error_code"] == "forbidden", "database_unchanged": digest == database_digest(core),
                              "all_reads_succeeded": all(r["succeeded"] for r in read_results.values()),
                              "advertises_write_tools": bool(set(advertised) & set(mcp_module.MCPAdapter.WRITE_TOOLS)) if advertised is not None else None},
                "constructor_default_access": default,
                "bounded_admission_runtime": {"status": "present, not load-tested"} if getattr(real_adapter, "runtime", None) is not None else unavailable("No bounded runtime attached at this boundary")}
    finally:
        for adapter in adapters:
            adapter.close()
        if not adapters:
            core.close()


def source_manifest(root):
    """Fingerprint all available AgentRecall source, including lazy imports."""
    manifest = {}
    for path in sorted(root.glob("agent_recall_*.py")):
        if path.resolve().parent != root or not path.is_file():
            raise RuntimeError("Source manifest escaped exact requested root")
        manifest[path.stem] = {"file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return manifest


def verify_source_manifest(root, expected):
    if source_manifest(root) != expected:
        raise RuntimeError("Source changed during benchmark; rerun after freeze")


def module_fingerprints(root):
    modules = {}
    for name, module in sorted(sys.modules.copy().items()):
        if not name.startswith("agent_recall_") or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if path.parent != root:
            raise RuntimeError("AgentRecall import escaped exact requested root")
        modules[name] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return modules


def worker(root, home, warmups, samples, seed):
    counters = install_sandbox(home)
    prefix = Path(sys.pycache_prefix).resolve() if sys.pycache_prefix else None
    if prefix is None or not prefix.is_relative_to(home) or not prefix.is_dir() or any(prefix.iterdir()):
        raise RuntimeError("Worker requires an empty disposable startup pycache prefix")
    sources = source_manifest(root)
    sys.path.insert(0, str(root))
    core_module = importlib.import_module("agent_recall_core")
    mcp_module = importlib.import_module("agent_recall_mcp") if (root / "agent_recall_mcp.py").is_file() else None
    verify_source_manifest(root, sources)
    before = module_fingerprints(root)
    if "agent_recall_core" not in before or "agent_recall_store" not in before:
        raise RuntimeError("Real core/store imports were not verified")
    retrieval = retrieval_probe(core_module, home, warmups, samples, seed)
    if mcp_module is None:
        mcp = unavailable("MCP adapter absent in this checkout")
    else:
        mcp = {boundary: asyncio.run(mcp_boundary_probe(core_module, mcp_module, home, boundary))
               for boundary in ("adapter", "sdk")}
    after = module_fingerprints(root)
    verify_source_manifest(root, sources)
    if any(after.get(name) != value for name, value in before.items()):
        raise RuntimeError("Source changed during benchmark; rerun after freeze")
    if any(sources.get(name) != value for name, value in after.items()):
        raise RuntimeError("Imported module differs from pre-import source manifest")
    return {"identity": {"root": str(root), "modules": after, "source_manifest": sources}, "retrieval": retrieval, "mcp": mcp,
            "isolation": {**counters, "environment": "allowlisted; disposable HOME", "storage": "temporary SQLite only",
                          "bytecode": "-B -X pycache_prefix=<empty disposable directory> before source imports",
                          "guard": "Python audit hook; trusted source, not hostile-code OS sandbox"}}


def run_worker(root, warmups, samples, seed):
    with tempfile.TemporaryDirectory(prefix="agentrecall-version-fixture-") as directory:
        home = Path(directory)
        prefix = home / "pycache"
        prefix.mkdir()
        result_path = home / "result.json"
        process = subprocess.run([sys.executable, "-B", "-X", f"pycache_prefix={prefix}", str(Path(__file__).resolve()), "--worker", str(root.resolve()),
                                  "--worker-home", str(home), "--warmups", str(warmups), "--samples", str(samples),
                                  "--seed", str(seed), "--worker-output", str(result_path)], cwd=home,
                                 env=worker_environment(home), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 timeout=180)
        if process.returncode or not result_path.is_file():
            # Never relay unbounded backend traces/config/error strings.
            diagnostic = json.loads(result_path.read_text()) if result_path.is_file() else {}
            raise RuntimeError(f"Isolated worker failed (exit {process.returncode}); safe diagnostic: {diagnostic}")
        if result_path.stat().st_size > MAX_OUTPUT_BYTES:
            raise RuntimeError("Worker output exceeded bound")
        return json.loads(result_path.read_text())


def checkout_identity(root):
    project = tomllib.loads((root / "pyproject.toml").read_text())
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise RuntimeError("Unable to verify checkout identity")
        return result.stdout.strip()
    return {"root": str(root), "package_version": project["project"]["version"], "git_head": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def aggregate(runs):
    result = {}
    if len({run["measurement"]["retrieval"]["corpus_sha256"] for run in runs}) != 1:
        raise RuntimeError("Different corpus across versions or rounds")
    for label in sorted({run["label"] for run in runs}):
        group = [run["measurement"] for run in runs if run["label"] == label]
        if any(g["identity"]["modules"] != group[0]["identity"]["modules"] for g in group):
            raise RuntimeError("Source changed between rounds; rerun after freeze")
        if any(g["retrieval"]["ranking"] != group[0]["retrieval"]["ranking"] for g in group):
            raise RuntimeError("Ranking changed between deterministic repetitions")
        failed = any(g.get("mcp", {}).get("status") == "error" or any(
            isinstance(boundary, dict) and boundary.get("status") == "error"
            for boundary in g.get("mcp", {}).values()) for g in group)
        result[label] = {"status": "error" if failed else "measured", "rounds": len(group), "ranking": group[0]["retrieval"]["ranking"],
                         "search_latency": latency_summary([v for g in group for v in g["retrieval"]["search_ms"]]),
                         "write_latency": latency_summary([v for g in group for v in g["retrieval"]["write_ms"]]),
                         "cross_agent_search_leaks": sum(g["retrieval"]["acl"]["cross_agent_search_leaks"] for g in group),
                         "cross_agent_direct_id_leaks": sum(g["retrieval"]["acl"]["cross_agent_direct_id_leaks"] for g in group)}
    return result


def markdown_report(report):
    lines = ["# Actual-root version comparison", "", "Synthetic lexical fallback only; embeddings disabled. No semantic-quality claim.",
             "Fresh isolated process/store per batch; paired randomized version order; warmups excluded from timings.",
             "p95 uses nearest rank. Write timing includes explicit read-write MCP admission when available.",
             "MRR uses the returned top-50 ranking (a missing target contributes zero); timed searches request top-5.",
             "No startup/cold-start timing, statistical significance claim, production-load claim or network transport benchmark.", "",
             f"Rounds: {report['parameters']['rounds']}; warmups/operation/batch: {report['parameters']['warmups']}; samples/operation/batch: {report['parameters']['samples']}; seed: {report['parameters']['seed']}.", "",
             "| Version | recall@1 | recall@5 | MRR | search median/p95 ms | write median/p95 ms | search/direct ACL leaks |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    if report.get("status") == "error":
        lines.insert(2, "**ERROR: one or more MCP boundaries or valid controls failed; counts are not a passing security result.**\n")
    for label, data in report["summary"].items():
        rank, search, write = data["ranking"], data["search_latency"], data["write_latency"]
        lines.append(f"| {label} | {rank['recall_at_1']:.3f} | {rank['recall_at_5']:.3f} | {rank['mrr']:.3f} | {search['median_ms']:.3f}/{search['p95_ms']:.3f} | {write['median_ms']:.3f}/{write['p95_ms']:.3f} | {data['cross_agent_search_leaks']}/{data['cross_agent_direct_id_leaks']} |")
    lines.extend(["", "## MCP security observations", "", "Metadata/read-only probes use the real core and SQLite. Invalid-input admission and backend exceptions use a labeled fault-injection backend. Each boundary is exercised separately; missing capability is N/A, never a zero score."])
    for run in report["runs"]:
        label = f"{run['label']}/round {run.get('round', 0) + 1}"
        measurement = run["measurement"]
        if measurement["mcp"].get("status") == "N/A":
            lines.append(f"- {label}/MCP: N/A ({measurement['mcp']['reason']}).")
            continue
        for boundary, data in measurement["mcp"].items():
            if not isinstance(data, dict):
                continue
            if data["status"] == "N/A":
                lines.append(f"- {label}/{boundary}: N/A ({data['reason']}).")
                continue
            if "invalid_inputs" not in data:
                lines.append(f"- {label}/{boundary}: **ERROR** ({data.get('reason', 'Probe failed')}; {data.get('error_type', 'unknown')}).")
                continue
            rejected = sum(v["rejected_before_backend"] for v in data["invalid_inputs"].values())
            exposed = any(v["internal_session_exposed"] for v in data["metadata_reads"].values())
            ro = data["read_only"]
            state = "**ERROR**" if data["status"] == "error" else data["status"]
            lines.append(f"- {label}/{boundary}: {state}; valid control succeeded={data['valid_control']['succeeded']} (backend calls={data['valid_control']['backend_calls']}); all reads succeeded={ro['all_reads_succeeded']}; invalid rejected before backend {rejected}/{len(data['invalid_inputs'])}; backend injection={data['backend_error']['status']} (calls={data['backend_error']['backend_calls']}); error sentinel exposed={data['backend_error']['sentinel_exposed']}; nested internal session exposed={exposed}; explicit read-only write rejected={ro['write_rejected']}, DB unchanged={ro['database_unchanged']}; default={data['constructor_default_access']}; errors={','.join(data['errors']) or 'none'}.")
    lines.extend(["", "JSON contains every round, actual module __file__/SHA-256, code identity, schedule, raw bounded latency samples and per-case observations. No response bodies, config, secrets or live database content are included.", ""])
    return "\n".join(lines)


def bounded_int(name, low, high):
    def parse(value):
        result = int(value)
        if not low <= result <= high:
            raise argparse.ArgumentTypeError(f"{name} must be between {low} and {high}")
        return result
    return parse


def publication_targets(json_output, markdown_output):
    # Resolve the parent, never the leaf: resolving a leaf would follow symlinks
    # and turn the protected destination into its (possibly absent) referent.
    if json_output.suffix != ".json":
        raise ValueError("JSON publication output must have a .json suffix")
    if markdown_output.suffix != ".md":
        raise ValueError("Markdown publication output must have a .md suffix")
    json_output = json_output.parent.resolve() / json_output.name
    markdown_output = markdown_output.parent.resolve() / markdown_output.name
    if json_output == markdown_output:
        raise ValueError("JSON and Markdown publication outputs must be distinct")
    for target in (json_output, markdown_output):
        if os.path.lexists(target):
            raise FileExistsError(f"Publication target already exists: {target}")
    return json_output, markdown_output


def publish_reports(json_output, markdown_output, payload, markdown):
    """Publish complete files exclusively; JSON is the last commit marker.

    Hard links on the same filesystem are atomic and fail on any existing leaf,
    including broken symlinks. Never roll back a public path: a racer could have
    replaced it. A collision on JSON can leave our complete Markdown companion;
    neither a concurrent file nor an earlier report is ever removed/overwritten.
    """
    json_output, markdown_output = publication_targets(json_output, markdown_output)
    for target in (json_output, markdown_output):
        target.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        staged = []
        for target, content in ((markdown_output, markdown), (json_output, payload)):
            directory = stack.enter_context(tempfile.TemporaryDirectory(
                prefix=".comparison-stage-", dir=target.parent))
            source = Path(directory) / target.name
            with source.open("x", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((source, target))
        for source, target in staged:
            os.link(source, target, follow_symlinks=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--rounds", type=bounded_int("rounds", 1, 10), default=3)
    parser.add_argument("--warmups", type=bounded_int("warmups", 1, 100), default=5)
    parser.add_argument("--samples", type=bounded_int("samples", 1, 200), default=30)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-home", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.worker_home or not args.worker_output:
            parser.error("--worker-home and --worker-output are required in worker mode")
        try:
            # SDK logging and raw errors never enter benchmark artifacts.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                report = worker(args.worker.resolve(), args.worker_home.resolve(), args.warmups, args.samples, args.seed)
            args.worker_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        except Exception as exc:
            frames = []
            tb = exc.__traceback__
            while tb is not None:
                frames.append({"file": Path(tb.tb_frame.f_code.co_filename).name,
                               "function": tb.tb_frame.f_code.co_name, "line": tb.tb_lineno})
                tb = tb.tb_next
            args.worker_output.write_text(json.dumps({"error_type": type(exc).__name__, "frames": frames[-8:]}))
            return 1
        return 0
    if not args.baseline_root:
        parser.error("--baseline-root is required")
    if not args.candidate_root:
        parser.error("--candidate-root is required")
    if not args.json_output:
        parser.error("--json-output is required")
    if not args.markdown_output:
        parser.error("--markdown-output is required")
    args.json_output, args.markdown_output = publication_targets(args.json_output, args.markdown_output)
    roots = {"baseline": args.baseline_root.resolve(), "candidate": args.candidate_root.resolve()}
    if roots["candidate"] == roots["baseline"]:
        parser.error("Baseline and candidate must be distinct roots")
    identities = {label: checkout_identity(root) for label, root in roots.items()}
    sources = {label: source_manifest(root) for label, root in roots.items()}
    schedule = make_schedule(list(roots), args.rounds, args.seed)
    runs = []
    for round_index, labels in enumerate(schedule):
        for label in labels:
            result = run_worker(roots[label], args.warmups, args.samples, args.seed + round_index)
            if result["identity"]["source_manifest"] != sources[label]:
                raise RuntimeError("Source changed before worker import; rerun after freeze")
            runs.append({"round": round_index, "label": label, "measurement": result})
    if identities != {label: checkout_identity(root) for label, root in roots.items()}:
        raise RuntimeError("Checkout identity changed during run; rerun after freeze")
    for label, root in roots.items():
        verify_source_manifest(root, sources[label])
    summary = aggregate(runs)
    status = "error" if any(data["status"] == "error" for data in summary.values()) else "measured"
    report = {"schema_version": 2, "status": status, "measurement_kind": "actual-root execution on disposable synthetic fixtures",
              "parameters": {"rounds": args.rounds, "warmups": args.warmups, "samples": args.samples, "seed": args.seed},
              "identities": identities, "schedule": schedule, "summary": summary, "runs": runs,
              "candidate_status": summary["candidate"]["status"]}
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    markdown = markdown_report(report)
    if len(payload.encode()) > MAX_OUTPUT_BYTES:
        raise RuntimeError("Report exceeds output bound")
    for label, root in roots.items():
        verify_source_manifest(root, sources[label])
    publish_reports(args.json_output, args.markdown_output, payload, markdown)
    print(json.dumps({"json_output": str(args.json_output), "markdown_output": str(args.markdown_output),
                      "status": status, "summary": report["summary"],
                      "candidate_status": report["candidate_status"]}))
    return 1 if status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
