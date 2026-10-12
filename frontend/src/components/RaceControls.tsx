// Play / pause / step / speed / save. Buttons show "pending" until the backend's ack arrives;
// success is only ever shown from an accepted ack (plan §13.4, US-09). The play state shown is
// the stream's (authoritative) status, never a local guess.

import { useState } from "react";

import { api, ApiError } from "../api/client";
import type { CommandAck, WorldState } from "../api/types";

const SPEEDS = [1, 5, 10, 25];
const STEPS: { label: string; ticks: number; title: string }[] = [
  { label: "+1 tick", ticks: 1, title: "Advance 0.5 s of race time" },
  { label: "+10 s", ticks: 20, title: "Advance 10 s of race time" },
  { label: "+1 min", ticks: 120, title: "Advance 1 minute of race time" },
];

type Notice = { kind: "ok" | "error" | "pending"; text: string } | null;

interface Props {
  state: WorldState;
}

export function RaceControls({ state }: Props) {
  const [pending, setPending] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice>(null);
  const id = state.world_id;

  const run = async (name: string, send: () => Promise<CommandAck>, ok?: (a: CommandAck) => string) => {
    setPending(name);
    if (name === "save") setNotice({ kind: "pending", text: "Saving…" });
    try {
      const ack = await send();
      if (!ack.accepted) setNotice({ kind: "error", text: `${ack.type} refused: ${ack.reason ?? "?"}` });
      else if (ok) setNotice({ kind: "ok", text: ok(ack) });
      else setNotice(null);
    } catch (e) {
      const text = e instanceof ApiError ? `${e.code}: ${e.message}` : `backend unreachable (${String(e)})`;
      setNotice({ kind: "error", text });
    } finally {
      setPending(null);
    }
  };

  const playPause = () => {
    if (state.status === "CREATED") return run("play", () => api.start(id));
    if (state.status === "RUNNING") return run("play", () => api.pause(id));
    if (state.status === "PAUSED") return run("play", () => api.resume(id));
  };
  const playLabel =
    state.status === "RUNNING" ? "Pause" : state.status === "CREATED" ? "Start" : "Play";
  const paused = state.status === "PAUSED";
  const ps = state.persistence;

  return (
    <section className="controls" aria-label="Race controls">
      <button
        className="primary"
        onClick={playPause}
        disabled={pending !== null || state.status === "COMPLETED"}
        aria-busy={pending === "play"}
      >
        {pending === "play" ? "…" : playLabel}
      </button>
      <div className="group" role="group" aria-label="Step (paused only)">
        {STEPS.map((s) => (
          <button
            key={s.ticks}
            title={paused ? s.title : "Pause first to step"}
            disabled={!paused || pending !== null}
            onClick={() => run("step", () => api.step(id, s.ticks))}
          >
            {s.label}
          </button>
        ))}
      </div>
      <div className="group" role="group" aria-label="Speed">
        {SPEEDS.map((v) => (
          <button
            key={v}
            className={state.speed === v ? "active" : ""}
            aria-pressed={state.speed === v}
            disabled={pending !== null}
            onClick={() => run("speed", () => api.speed(id, v))}
          >
            x{v}
          </button>
        ))}
      </div>
      <button
        onClick={() =>
          run("save", () => api.save(id), (a) => {
            const label = a.snapshot?.label;
            const named = label && label !== `tick ${a.applied_tick}` ? ` · ${label}` : "";
            return `Saved at tick ${a.applied_tick}${named}`;
          })
        }
        disabled={pending !== null || !ps?.store}
        title={
          ps?.last_autosave
            ? `Autosaves every 5 laps and after strategy commands (last ${ps.last_autosave.slice(0, 13)}…)`
            : "Autosaves every 5 laps and after strategy commands"
        }
      >
        {pending === "save" ? "Saving…" : "Save"}
      </button>
      <div className={`notice ${notice?.kind ?? ""}`} role="status" aria-live="polite">
        {ps?.error ? `⚠ ${ps.error}` : notice?.text ?? ""}
      </div>
    </section>
  );
}
