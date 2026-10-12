// Play / pause / step / speed / save / fork / race director. Buttons show "pending" until the
// backend's ack arrives; success is only shown from an accepted ack (plan §13.4, US-09). The
// play state shown is the stream's (authoritative) status, never a local guess.
//
// Keyboard (plan §14.5): Space play/pause, S save, → step 1 tick, Shift+→ step 10 s (paused).
// Ignored while typing or while a dialog is open.

import { useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import { useCommand } from "../api/useCommand";
import type { CommandAck, WorldState } from "../api/types";
import { isTyping, Modal, NoticeLine } from "./common";

const SPEEDS = [1, 5, 10, 25];
const STEPS: { label: string; ticks: number; title: string }[] = [
  { label: "+1 tick", ticks: 1, title: "Advance 0.5 s of race time (→)" },
  { label: "+10 s", ticks: 20, title: "Advance 10 s of race time (Shift+→)" },
  { label: "+1 min", ticks: 120, title: "Advance 1 minute of race time" },
];

function savedText(a: CommandAck): string {
  const label = a.snapshot?.label;
  const named = label && label !== `tick ${a.applied_tick}` ? ` · “${label}”` : "";
  return `Saved at tick ${a.applied_tick}${named}`;
}

interface Props {
  state: WorldState;
  onFork: () => void;
}

export function RaceControls({ state, onFork }: Props) {
  const { pending, notice, run } = useCommand();
  const [saveAs, setSaveAs] = useState(false);
  const [label, setLabel] = useState("");
  const id = state.world_id;
  const paused = state.status === "PAUSED";
  const live = state.status === "RUNNING" || paused;
  const ps = state.persistence;

  const playPause = () => {
    if (state.status === "CREATED") return run("play", () => api.start(id));
    if (state.status === "RUNNING") return run("play", () => api.pause(id));
    if (state.status === "PAUSED") return run("play", () => api.resume(id));
  };
  const save = (name?: string) =>
    run("save", () => api.save(id, name), savedText, "Saving…");
  const step = (ticks: number) => run("step", () => api.step(id, ticks));

  // keyboard shortcuts read the latest handlers through a ref (no re-subscribing every frame)
  const keys = useRef({ playPause, save, step, paused, pending });
  keys.current = { playPause, save, step, paused, pending };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = keys.current;
      if (k.pending !== null) return;
      if (e.code === "Space") {
        e.preventDefault();
        void k.playPause();
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault();
        void k.save();
      } else if (e.key === "ArrowRight" && k.paused) {
        e.preventDefault();
        void k.step(e.shiftKey ? 20 : 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const playLabel =
    state.status === "RUNNING" ? "Pause" : state.status === "CREATED" ? "Start" : "Play";
  const rc = state.race_control;

  return (
    <section className="controls" aria-label="Race controls">
      <button
        className="primary"
        onClick={playPause}
        disabled={pending !== null || state.status === "COMPLETED"}
        aria-busy={pending === "play"}
        title="Space"
      >
        {pending === "play" ? "…" : playLabel}
      </button>
      <div className="group" role="group" aria-label="Step (paused only)">
        {STEPS.map((s) => (
          <button
            key={s.ticks}
            title={paused ? s.title : "Pause first to step"}
            disabled={!paused || pending !== null}
            onClick={() => step(s.ticks)}
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
      <div className="group" role="group" aria-label="Save and fork">
        <button
          onClick={() => save()}
          disabled={pending !== null || !ps?.store}
          title={`Save now (S). Autosaves every 5 laps and after decisions${ps?.last_autosave ? `; last ${ps.last_autosave.slice(0, 13)}…` : ""}`}
        >
          {pending === "save" ? "Saving…" : "Save"}
        </button>
        <button disabled={pending !== null || !ps?.store} onClick={() => setSaveAs(true)}>
          Save as…
        </button>
        <button disabled={pending !== null || state.status === "CREATED"} onClick={onFork} title="Branch the race here and try another decision">
          Fork…
        </button>
      </div>
      <div className="group" role="group" aria-label="Race director">
        <button
          disabled={pending !== null || !live || rc.phase === "SC"}
          title="Race director: send out the safety car now (4 laps)"
          onClick={() => run("sc", () => api.action(id, { action_type: "DEPLOY_SC", laps: 4 }), () => "Safety car deployed")}
        >
          Safety car
        </button>
        <button
          disabled={pending !== null || !live || rc.phase !== "GREEN"}
          title="Race director: virtual safety car now (90 s)"
          onClick={() => run("vsc", () => api.action(id, { action_type: "DEPLOY_VSC", duration_s: 90 }), () => "VSC deployed")}
        >
          VSC
        </button>
      </div>
      <NoticeLine notice={ps?.error ? { kind: "error", text: `⚠ ${ps.error}` } : notice} />
      {saveAs ? (
        <Modal title="Save as" onClose={() => setSaveAs(false)}>
          <form
            className="dialog-body"
            onSubmit={(e) => {
              e.preventDefault();
              setSaveAs(false);
              void save(label.trim() || undefined);
              setLabel("");
            }}
          >
            <label className="block">
              Name
              <input
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder={`tick ${state.tick}`}
                maxLength={120}
              />
            </label>
            <div className="dialog-actions">
              <button type="button" onClick={() => setSaveAs(false)}>
                Cancel
              </button>
              <button className="primary" type="submit">
                Save
              </button>
            </div>
          </form>
        </Modal>
      ) : null}
    </section>
  );
}
