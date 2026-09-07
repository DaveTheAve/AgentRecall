from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Minimal stubs when running outside Hermes source tree.
try:
    import agent.memory_provider  # noqa
except Exception:
    import types
    agent = types.ModuleType("agent")
    memory_provider = types.ModuleType("agent.memory_provider")
    class MemoryProvider:
        pass
    memory_provider.MemoryProvider = MemoryProvider
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = memory_provider

spec = importlib.util.spec_from_file_location(
    "agent_recall",
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
assert spec and spec.loader
agent_recall = importlib.util.module_from_spec(spec)
sys.modules["agent_recall"] = agent_recall
spec.loader.exec_module(agent_recall)
AgentRecallProvider = agent_recall.AgentRecallProvider


class FakeEmbedder:
    def embed(self, text: str):
        # tiny deterministic-ish embedding good enough for ACL smoke tests
        return [float((sum(map(ord, text)) + i) % 17) / 17.0 for i in range(8)]


def main():
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent-recall.db")
        p1 = AgentRecallProvider({
            "db_path": db,
            "workspace_id": "shared-ws",
            "agent_id": "hermes",
            "embedding_base_url": "",
            "embedding_model": "fake",
            "excluded_terms": ["blocked-project"],
        })
        p1.initialize("s1", hermes_home=td, agent_identity="hermes")
        p1._embedder = FakeEmbedder()
        a = json.loads(p1.handle_tool_call("agent_recall_remember", {"content": "User prefers concise terminal output", "visibility": "shared", "category": "user_pref"}))
        b = json.loads(p1.handle_tool_call("agent_recall_remember", {"content": "Hermes private implementation note", "visibility": "agent", "category": "agent"}))
        assert a["success"] and b["success"]
        blocked = json.loads(p1.handle_tool_call("agent_recall_remember", {"content": "blocked-project should not be stored"}))
        # Hermes's native tool_error uses {"error": ...}; the standalone fallback
        # also includes success=False. Both must reject the excluded content.
        assert blocked.get("success") is not True and blocked.get("error")
        p1.shutdown()

        p2 = AgentRecallProvider({
            "db_path": db,
            "workspace_id": "shared-ws",
            "agent_id": "other-agent",
            "embedding_base_url": "",
            "embedding_model": "fake",
        })
        p2.initialize("s2", hermes_home=td, agent_identity="other-agent")
        p2._embedder = FakeEmbedder()
        result = json.loads(p2.handle_tool_call("agent_recall_search", {"query": "concise output", "limit": 10}))
        contents = [r["content"] for r in result["results"]]
        assert any("concise" in c for c in contents), contents
        assert not any("private implementation" in c for c in contents), contents
        stats = json.loads(p2.handle_tool_call("agent_recall_stats", {}))
        assert stats["success"]
        p2.shutdown()
    print("AgentRecall smoke test passed")


if __name__ == "__main__":
    main()
