import assert from "node:assert/strict";
import { access, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { createAgentRecallPlugin } from "../index.js";
import { PythonBridgeClient } from "../bridge-client.js";

function createFakeBridge() {
  const calls = [];
  return {
    calls,
    async call(operation, identity, args = {}) {
      calls.push({ operation, identity, args });
      if (operation === "prefetch_context") {
        return {
          success: true,
          context: "# AgentRecall Context\n- remembered OpenClaw preference",
          count: 1,
        };
      }
      if (operation === "search") {
        return {
          success: true,
          results: [
            {
              id: 7,
              content: "remembered OpenClaw preference",
              summary: "preference",
              session_id: "private-channel-session",
              metadata: { source_path: "/home/service/private/secret.md" },
              score: 0.91,
              score_explanation: { lexical: 0.8, vector: 0.9 },
            },
          ],
        };
      }
      if (operation === "get_memory") {
        return {
          success: true,
          memory: { id: 7, content: "remembered OpenClaw preference" },
        };
      }
      if (operation === "health") {
        return {
          success: true,
          embedding: { enabled: true },
          sqlite: { integrity: "ok" },
        };
      }
      return { success: true };
    },
    async close() {
      calls.push({ operation: "bridge_close" });
    },
  };
}

function createFakeApi(config = {}) {
  const tools = [];
  const hooks = new Map();
  let memoryCapability;
  let service;
  let cli;
  return {
    pluginConfig: config,
    config: {},
    logger: { debug() {}, info() {}, warn() {}, error() {} },
    registerTool(factory, options) {
      tools.push({ factory, options });
    },
    registerMemoryCapability(capability) {
      memoryCapability = capability;
    },
    registerService(value) {
      service = value;
    },
    registerCli(registrar, options) {
      cli = { registrar, options };
    },
    on(name, handler, options) {
      hooks.set(name, { handler, options });
    },
    get captured() {
      return { tools, hooks, memoryCapability, service, cli };
    },
  };
}

const hookContext = {
  agentId: "main",
  sessionId: "session-123",
  sessionKey: "agent:main:telegram:dm:42",
  contextTokenBudget: 64_000,
};

test("registers the native memory capability, standard tools, and lifecycle hooks", () => {
  const bridge = createFakeBridge();
  const api = createFakeApi({
    workspaceId: "shared",
    agentIdPrefix: "openclaw",
  });
  const plugin = createAgentRecallPlugin({ bridgeFactory: () => bridge });
  plugin.register(api);

  assert.equal(plugin.kind, "memory");
  assert.ok(api.captured.memoryCapability);
  assert.equal(api.captured.cli.options.descriptors[0].name, "memory");
  assert.deepEqual(
    api.captured.tools.flatMap(
      (entry) => entry.options.names ?? [entry.options.name],
    ),
    [
      "memory_search",
      "memory_get",
      "memory_store",
      "memory_forget",
      "agent_recall_update",
      "agent_recall_curate",
      "agent_recall_conclude",
      "agent_recall_profile",
      "agent_recall_review",
      "agent_recall_stats",
    ],
  );
  for (const hook of [
    "before_prompt_build",
    "agent_end",
    "before_compaction",
    "before_reset",
    "session_end",
    "gateway_stop",
  ]) {
    assert.ok(api.captured.hooks.has(hook), `missing hook ${hook}`);
  }
});

test("injects automatic recall through before_prompt_build with host identity", async () => {
  const bridge = createFakeBridge();
  const api = createFakeApi({
    workspaceId: "shared",
    agentIdPrefix: "openclaw",
    maxContextChars: 4000,
  });
  createAgentRecallPlugin({ bridgeFactory: () => bridge }).register(api);

  const result = await api.captured.hooks
    .get("before_prompt_build")
    .handler({ prompt: "What do I prefer?", messages: [] }, hookContext);

  assert.equal(
    result.prependContext,
    "# AgentRecall Context\n- remembered OpenClaw preference",
  );
  assert.deepEqual(bridge.calls[0].identity, {
    workspace_id: "shared",
    agent_id: "openclaw:main",
    session_id: "agent:main:telegram:dm:42",
  });
  assert.equal(bridge.calls[0].operation, "prefetch_context");
  assert.equal(bridge.calls[0].args.include_results, false);
});

test("captures completed turns and compaction/reset checkpoints", async () => {
  const bridge = createFakeBridge();
  const api = createFakeApi({
    workspaceId: "shared",
    autoCaptureTurns: true,
    autoCaptureCheckpoints: true,
  });
  createAgentRecallPlugin({ bridgeFactory: () => bridge }).register(api);

  const messages = [
    { role: "user", content: "Remember this decision" },
    {
      role: "assistant",
      content: [{ type: "text", text: "Decision captured" }],
    },
  ];
  await api.captured.hooks
    .get("agent_end")
    .handler({ messages, success: true }, hookContext);
  await api.captured.hooks
    .get("before_compaction")
    .handler({ messages }, hookContext);
  await api.captured.hooks
    .get("before_reset")
    .handler({ messages }, hookContext);
  await api.captured.hooks
    .get("session_end")
    .handler({ reason: "reset" }, hookContext);

  assert.equal(bridge.calls[0].operation, "capture_turn");
  assert.equal(bridge.calls[0].args.user_content, "Remember this decision");
  assert.equal(bridge.calls[0].args.assistant_content, "Decision captured");
  assert.deepEqual(
    bridge.calls.slice(1).map((call) => call.operation),
    ["save_checkpoint", "save_checkpoint", "release"],
  );
});

test("standard memory tools and native runtime delegate without exposing filesystem paths", async () => {
  const bridge = createFakeBridge();
  const api = createFakeApi({ workspaceId: "shared" });
  createAgentRecallPlugin({ bridgeFactory: () => bridge }).register(api);

  const searchRegistration = api.captured.tools.find((entry) =>
    entry.options.names?.includes("memory_search"),
  );
  const searchTool = searchRegistration.factory(hookContext);
  const result = await searchTool.execute("call-1", {
    query: "preference",
    maxResults: 3,
  });
  assert.match(result.content[0].text, /remembered OpenClaw preference/);
  const parsedSearch = JSON.parse(result.content[0].text).results[0];
  assert.equal(parsedSearch.path, "agent-recall://memory/7");
  assert.equal(parsedSearch.session_id, undefined);
  assert.equal(parsedSearch.metadata.source_path, "[redacted]");
  const getRegistration = api.captured.tools.find((entry) =>
    entry.options.names?.includes("memory_get"),
  );
  const getResult = await getRegistration
    .factory(hookContext)
    .execute("call-2", {
      path: "agent-recall://memory/7",
    });
  const parsedGet = JSON.parse(getResult.content[0].text);
  assert.match(parsedGet.text, /remembered OpenClaw preference/);
  assert.equal(parsedGet.path, "agent-recall://memory/7");
  const curateRegistration = api.captured.tools.find((entry) =>
    entry.options.names?.includes("agent_recall_curate"),
  );
  await curateRegistration
    .factory(hookContext)
    .execute("call-3", { text: "candidate fact" });
  assert.equal(
    bridge.calls.find((call) => call.operation === "curate").args.dry_run,
    true,
  );

  const runtime = api.captured.memoryCapability.runtime;
  const { manager } = await runtime.getMemorySearchManager({
    cfg: {},
    agentId: "main",
  });
  const rows = await manager.search("preference", {
    sessionKey: hookContext.sessionKey,
    maxResults: 3,
  });
  assert.equal(rows[0].path, "agent-recall://memory/7");
  assert.equal(rows[0].snippet, "preference");
  const runtimeSearch = bridge.calls
    .filter((call) => call.operation === "search")
    .at(-1);
  assert.equal(Object.hasOwn(runtimeSearch.args, "include_shared"), false);
  assert.equal(
    runtimeSearch.identity.session_id,
    "",
    "the shared OpenClaw memory manager must not authorize session-scoped rows",
  );
  const read = await manager.readFile({ relPath: "agent-recall://memory/7" });
  assert.equal(read.text, "remembered OpenClaw preference");
  const runtimeRead = bridge.calls
    .filter((call) => call.operation === "get_memory")
    .at(-1);
  assert.equal(runtimeRead.identity.session_id, "");
  assert.equal(manager.status().dbPath, undefined);
});

test("gateway shutdown closes the persistent bridge", async () => {
  const bridge = createFakeBridge();
  const api = createFakeApi();
  createAgentRecallPlugin({ bridgeFactory: () => bridge }).register(api);
  await api.captured.hooks.get("gateway_stop").handler({}, {});
  assert.equal(bridge.calls.at(-1).operation, "bridge_close");
});

test("bridge ignores inherited PYTHONPATH and executes the packaged absolute script", async () => {
  const directory = await mkdtemp(
    path.join(os.tmpdir(), "agent-recall-hijack-"),
  );
  const marker = path.join(directory, "hijacked");
  await writeFile(
    path.join(directory, "agent_recall_bridge.py"),
    `from pathlib import Path\nPath(${JSON.stringify(marker)}).write_text('hijacked')\n`,
  );
  const configPath = path.join(directory, "agent-recall.json");
  await writeFile(
    configPath,
    JSON.stringify({
      db_path: path.join(directory, "memory.db"),
      embedding_base_url: "",
    }),
  );
  const previous = process.env.PYTHONPATH;
  process.env.PYTHONPATH = directory;
  const client = new PythonBridgeClient({ configPath, requestTimeoutMs: 5000 });
  try {
    const result = await client.call(
      "capabilities",
      { workspace_id: "security", agent_id: "openclaw:main", session_id: "" },
      {},
    );
    assert.equal(result.success, true);
    await assert.rejects(() => access(marker));
  } finally {
    if (previous === undefined) delete process.env.PYTHONPATH;
    else process.env.PYTHONPATH = previous;
    await client.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test("persistent stdio bridge executes the packaged Python core", async () => {
  const directory = await mkdtemp(
    path.join(os.tmpdir(), "agent-recall-openclaw-"),
  );
  const configPath = path.join(directory, "agent-recall.json");
  await writeFile(
    configPath,
    JSON.stringify({
      db_path: path.join(directory, "shared.db"),
      embedding_base_url: "",
      raw_memories_enabled: true,
    }),
  );
  const client = new PythonBridgeClient({ configPath, requestTimeoutMs: 5000 });
  const identity = {
    workspace_id: "shared",
    agent_id: "openclaw:main",
    session_id: "native",
  };
  try {
    const stored = await client.call("remember", identity, {
      content: "OpenClaw persistent bridge proof",
      visibility: "agent",
    });
    const found = await client.call("search", identity, {
      query: "persistent bridge proof",
    });
    assert.equal(found.results[0].id, stored.id);
    assert.equal(found.results[0].content, "OpenClaw persistent bridge proof");
  } finally {
    await client.close();
    await rm(directory, { recursive: true, force: true });
  }
});
