import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const packageRoot = path.resolve(process.argv[2] || "");
if (!process.argv[2]) {
  throw new Error("usage: node scripts/smoke_packed_openclaw.mjs /path/to/extracted/package");
}

const bridgeModule = pathToFileURL(
  path.join(packageRoot, "openclaw_plugin", "bridge-client.js"),
).href;
const { PythonBridgeClient } = await import(bridgeModule);
const scratch = await mkdtemp(path.join(os.tmpdir(), "agent-recall-packed-smoke-"));
const configPath = path.join(scratch, "agent-recall.json");
await writeFile(
  configPath,
  JSON.stringify({
    db_path: path.join(scratch, "shared.db"),
    embedding_base_url: "",
    raw_memories_enabled: true,
  }),
);

const client = new PythonBridgeClient({ configPath, requestTimeoutMs: 5000 });
const identity = {
  workspace_id: "package-smoke",
  agent_id: "test-agent",
  session_id: "test-session",
};
try {
  const stored = await client.call("remember", identity, {
    content: "Packed bridge smoke fact",
    visibility: "agent",
  });
  const found = await client.call("search", identity, {
    query: "packed bridge smoke",
  });
  if (found.results?.[0]?.id !== stored.id) {
    throw new Error("packed bridge failed remember/search round trip");
  }
  console.log("packed-openclaw-bridge-smoke-ok");
} finally {
  await client.close();
  await rm(scratch, { recursive: true, force: true });
}
