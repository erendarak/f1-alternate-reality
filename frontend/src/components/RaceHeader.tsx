// Race, lap, clock, status, the mode indicator and the SC/VSC banner.

import type { Connection } from "../api/stream";
import type { WorldState } from "../api/types";
import { fmtClock } from "../format";

const MODE: Record<string, { label: string; title: string }> = {
  FREE_SIMULATION: {
    label: "Alternate simulation",
    title:
      "Simulated by our engine from the real grid and data-derived inputs; not the real race. " +
      "Driver personas are editorial assumptions.",
  },
  HISTORICAL_REPLAY: { label: "Historical replay", title: "The real race, as recorded" },
};

interface Props {
  state: WorldState;
  connection: Connection;
}

export function RaceHeader({ state, connection }: Props) {
  const mode = MODE[state.mode] ?? { label: state.mode, title: "" };
  const rc = state.race_control;
  return (
    <header className="race-header">
      <div className="brand">F1 ALTERNATE REALITY</div>
      <div className="race">{state.race_name}</div>
      <div className="lap" aria-label="lap">
        LAP <b>{Math.min(state.leader_lap, state.total_laps)}</b>/{state.total_laps}
      </div>
      <div className="clock" title="race time">
        {fmtClock(state.sim_clock_s)}
      </div>
      <span className={`pill status-${state.status.toLowerCase()}`}>{state.status}</span>
      <span className="pill mode" title={`${mode.title}\nengine ${state.engine_version}, seed ${state.seed}`}>
        {mode.label}
      </span>
      {state.parent_timeline_id ? (
        <span className="pill branch" title={`branched from ${state.parent_timeline_id}`}>
          Branch
        </span>
      ) : null}
      {connection !== "open" ? <span className="pill warn">{connection}…</span> : null}
      {rc.phase !== "GREEN" ? (
        <div className={`banner banner-${rc.phase.toLowerCase()}`} role="alert">
          {rc.phase === "SC"
            ? `SAFETY CAR${rc.sc_laps_left != null ? ` · ${rc.sc_laps_left} laps` : ""}`
            : `VIRTUAL SAFETY CAR${rc.vsc_left_s != null ? ` · ${Math.max(0, Math.round(rc.vsc_left_s))} s` : ""}`}
          {rc.cause ? ` · ${rc.cause}` : ""}
        </div>
      ) : null}
    </header>
  );
}
