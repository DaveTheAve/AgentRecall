from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Test outside a full Hermes checkout by providing tiny stubs for imports.
if "agent.memory_provider" not in sys.modules:
    agent = types.ModuleType("agent")
    memory_provider = types.ModuleType("agent.memory_provider")
    class MemoryProvider:
        pass
    memory_provider.MemoryProvider = MemoryProvider
    sys.modules.setdefault("agent", agent)
    sys.modules.setdefault("agent.memory_provider", memory_provider)

if "tools.registry" not in sys.modules:
    tools = types.ModuleType("tools")
    registry = types.ModuleType("tools.registry")
    def tool_error(message: str) -> str:
        import json
        return json.dumps({"success": False, "error": message})
    registry.tool_error = tool_error
    sys.modules.setdefault("tools", tools)
    sys.modules.setdefault("tools.registry", registry)


def load_provider_module():
    spec = importlib.util.spec_from_file_location(
        "agent_recall",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_recall"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text: str):
        self.calls.append(text)
        seed = sum(ord(c) for c in text)
        return [float((seed + i * 13) % 101) / 100.0 for i in range(16)]
