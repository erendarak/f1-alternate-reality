// The real backend (scripts/serve.py) as a child process the test can start, kill and restart
// on the same save file: a kill is abrupt (no clean shutdown), like a crash or a closed terminal.

import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const E2E_BACKEND_PORT = 8001;

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");

/** The Monza data the race needs (fetched data is gitignored, so it may be missing). */
export const MONZA_DATA = join(ROOT, "data", "processed", "2024_italian_gp", "track_outline.parquet");
export const hasMonzaData = () => existsSync(MONZA_DATA);

function python(): { cmd: string; args: string[] } {
  const venv = process.platform === "win32"
    ? join(ROOT, ".venv", "Scripts", "python.exe")
    : join(ROOT, ".venv", "bin", "python");
  // the venv's python directly: killing `uv run` could leave the server running on Windows
  return existsSync(venv) ? { cmd: venv, args: [] } : { cmd: "uv", args: ["run", "python"] };
}

export class Backend {
  private proc: ChildProcess | null = null;
  private log = "";

  constructor(
    readonly db: string,
    readonly port: number,
  ) {}

  get url(): string {
    return `http://127.0.0.1:${this.port}`;
  }

  async start(): Promise<void> {
    const { cmd, args } = python();
    this.log = "";
    this.proc = spawn(cmd, [...args, "scripts/serve.py", "--port", String(this.port), "--db", this.db], {
      cwd: ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
      stdio: ["ignore", "pipe", "pipe"],
    });
    this.proc.stdout?.on("data", (d: Buffer) => (this.log += d.toString()));
    this.proc.stderr?.on("data", (d: Buffer) => (this.log += d.toString()));
    const deadline = Date.now() + 60_000;
    while (Date.now() < deadline) {
      if (this.proc.exitCode !== null) throw new Error(`backend exited early:\n${this.log}`);
      try {
        const r = await fetch(`${this.url}/api/health`);
        if (r.ok) return;
      } catch {
        // not up yet
      }
      await new Promise((r) => setTimeout(r, 250));
    }
    throw new Error(`backend did not start within 60 s:\n${this.log}`);
  }

  /** Abrupt stop: the process gets no chance to pause, autosave or flush. */
  async kill(): Promise<void> {
    const proc = this.proc;
    if (!proc || proc.exitCode !== null) return;
    const exited = new Promise<void>((r) => proc.once("exit", () => r()));
    proc.kill("SIGKILL");
    await exited;
    this.proc = null;
  }

  async health(): Promise<{ live_world_id: string | null; store_schema_version: number }> {
    const r = await fetch(`${this.url}/api/health`);
    return (await r.json()) as { live_world_id: string | null; store_schema_version: number };
  }
}
