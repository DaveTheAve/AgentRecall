import { spawn } from "node:child_process";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";

const DEFAULT_MODULE_ROOT = fileURLToPath(new URL("..", import.meta.url));

export class PythonBridgeClient {
  constructor(options = {}) {
    this.pythonCommand =
      options.pythonCommand || process.env.AGENT_RECALL_PYTHON || "python3";
    this.moduleRoot = options.moduleRoot || DEFAULT_MODULE_ROOT;
    this.bridgeScript = path.join(this.moduleRoot, "agent_recall_bridge.py");
    this.configPath =
      options.configPath || process.env.AGENT_RECALL_CONFIG || "";
    this.maxContexts = Math.max(
      1,
      Math.min(Number(options.maxContexts || 128), 1024),
    );
    this.requestTimeoutMs = Math.max(
      100,
      Number(options.requestTimeoutMs || 10_000),
    );
    this.logger = options.logger || console;
    this.child = undefined;
    this.pending = new Map();
    this.nextId = 1;
    this.closing = false;
  }

  _start() {
    if (this.child && !this.child.killed) return;
    const args = [
      this.bridgeScript,
      "--max-contexts",
      String(this.maxContexts),
    ];
    if (this.configPath) args.push("--config", this.configPath);
    const child = spawn(this.pythonCommand, args, {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: this.moduleRoot,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        PYTHONPATH: this.moduleRoot,
      },
      shell: false,
    });
    this.child = child;
    this.closing = false;

    const output = readline.createInterface({ input: child.stdout });
    output.on("line", (line) => this._handleLine(line));
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      const text = String(chunk).trim().slice(0, 1000);
      if (text) this.logger.warn?.(`agent-recall bridge: ${text}`);
    });
    child.on("error", (error) => this._failAll(error));
    child.on("exit", (code, signal) => {
      this.child = undefined;
      if (!this.closing) {
        this._failAll(
          new Error(
            `AgentRecall bridge exited (code=${code}, signal=${signal || "none"})`,
          ),
        );
      }
    });
  }

  _handleLine(line) {
    let response;
    try {
      response = JSON.parse(line);
    } catch {
      this.logger.warn?.("agent-recall bridge returned malformed JSON");
      return;
    }
    const pending = this.pending.get(response.id);
    if (!pending) return;
    this.pending.delete(response.id);
    clearTimeout(pending.timer);
    if (response.ok) pending.resolve(response.result);
    else
      pending.reject(
        new Error(response.error || "AgentRecall bridge request failed"),
      );
  }

  _failAll(error) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }

  async call(operation, identity, args = {}, options = {}) {
    this._start();
    const id = this.nextId++;
    const timeoutMs = Math.max(
      100,
      Number(options.timeoutMs || this.requestTimeoutMs),
    );
    const request = JSON.stringify({ id, operation, identity, args }) + "\n";
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        const error = new Error(
          `AgentRecall bridge ${operation} timed out after ${timeoutMs} ms`,
        );
        reject(error);
        if (this.child) this.child.kill("SIGKILL");
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(request, "utf8", (error) => {
        if (!error) return;
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        clearTimeout(timer);
        reject(error);
      });
    });
  }

  async close() {
    const child = this.child;
    if (!child) return;
    this.closing = true;
    try {
      await this.call(
        "shutdown",
        { workspace_id: "bridge", agent_id: "bridge", session_id: "" },
        {},
        { timeoutMs: 2000 },
      );
    } catch {
      // A wedged bridge is terminated below.
    }
    if (this.child) this.child.kill("SIGTERM");
    this.child = undefined;
    this._failAll(new Error("AgentRecall bridge closed"));
  }
}
