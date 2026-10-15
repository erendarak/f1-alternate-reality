// Text for the timing tower, the event log and the lap strip (pure functions).

import type { Car, RaceEvent } from "./api/types";

export function fmtLap(seconds: number | null | undefined): string {
  if (seconds == null) return "–";
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(3).padStart(6, "0")}`;
}

export function fmtClock(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

export function gapText(car: Car): string {
  if (car.status === "RETIRED") return "OUT";
  if (car.position === 1) return car.status === "FINISHED" ? "WINNER" : "LEADER";
  if (car.laps_down != null && car.laps_down >= 1) {
    return `+${car.laps_down} LAP${car.laps_down > 1 ? "S" : ""}`;
  }
  return car.gap_to_leader_s == null ? "–" : `+${car.gap_to_leader_s.toFixed(1)}`;
}

/** Event types shown in the log (lap completions are too frequent to list). */
export const LOGGED = new Set([
  "PIT_STOP",
  "PIT_REQUESTED",
  "PIT_CANCELLED",
  "PACE_MODE_CHANGED",
  "CONTROLLER_CHANGED",
  "OVERTAKE",
  "MISTAKE",
  "CAR_RETIRED",
  "SAFETY_CAR_DEPLOYED",
  "SAFETY_CAR_IN",
  "VSC_DEPLOYED",
  "VSC_ENDED",
  "RESTART",
  "CHEQUERED_FLAG",
  "CAR_FINISHED",
  "RACE_COMPLETED",
  "TYRE_RULE_BREACHED",
  "RAIN_STARTED",
  "RAIN_STOPPED",
  "RED_FLAG",
  "TIME_PENALTY",
]);

type CodeOf = (driverId: string | null | undefined) => string;

const num = (v: number | string | undefined, digits = 1): string =>
  typeof v === "number" ? v.toFixed(digits) : String(v ?? "?");

/** One readable line per event; `code` turns driver ids into 3-letter codes. The log prefixes
 * each line with the race (leader) lap, so car-lap numbers are only kept where they matter. */
export function eventText(e: RaceEvent, code: CodeOf): string {
  const p = e.payload;
  const who = e.driver_id ? code(e.driver_id) : "";
  const director = p.actor_id === "USER" ? " (race director)" : "";
  switch (e.type) {
    case "PIT_STOP":
      if (p.under === "RED_FLAG") return `${who} new tyres under the red flag: ${p.from_compound} → ${p.to_compound}`;
      return `${who} pits: ${p.from_compound} → ${p.to_compound} after ${p.tyre_age_laps} laps, ${num(p.pit_loss_s)} s${p.under && p.under !== "GREEN" ? ` (under ${p.under})` : ""}`;
    case "PIT_REQUESTED":
      return `${who} told to box for ${p.compound} at the end of lap ${p.in_lap}`;
    case "PIT_CANCELLED":
      return `${who} stays out (lap ${p.in_lap})`;
    case "PACE_MODE_CHANGED":
      return `${who} ${p.from} → ${p.to} from lap ${p.effective_from_lap}`;
    case "CONTROLLER_CHANGED":
      return `${who} strategy: ${p.from} → ${p.to}`;
    case "OVERTAKE":
      return `${who} passes ${code(String(p.passed))}`;
    case "MISTAKE":
      return `${who} mistake on lap ${p.lap}, ${num(p.time_loss_s)} s lost`;
    case "CAR_RETIRED":
      return `${who} OUT on lap ${p.lap} (${p.reason === "CRASH" ? "crash" : "mechanical failure"})`;
    case "SAFETY_CAR_DEPLOYED":
      return `SAFETY CAR for ${p.laps} laps${director}`;
    case "SAFETY_CAR_IN":
      return "Safety car in this lap";
    case "VSC_DEPLOYED":
      return `VIRTUAL SAFETY CAR for ${Math.round(Number(p.duration_s))} s${director}`;
    case "VSC_ENDED":
      return "VSC ending: green flag";
    case "RESTART":
      return "Green flag: racing resumes";
    case "CHEQUERED_FLAG":
      return "Chequered flag";
    case "CAR_FINISHED":
      return `${who} finishes (${p.laps} laps)`;
    case "RACE_COMPLETED":
      return "Race completed";
    case "RAIN_STARTED":
      return "Rain starting";
    case "RAIN_STOPPED":
      return "Rain stopping";
    case "TYRE_RULE_BREACHED":
      return `${who} didn't use two dry compounds`;
    case "RED_FLAG":
      return `RED FLAG (${who} crash on lap ${p.lap}): free tyre change, restart behind the safety car`;
    case "TIME_PENALTY":
      return `${who} ${num(p.seconds, 0)} s penalty (${p.reason})`;
    default:
      return `${who} ${e.type}`.trim();
  }
}

/** Kind of marker on the lap strip, or null if the event isn't marked. */
export function markerKind(type: string): "pit" | "sc" | "vsc" | "out" | "pass" | null {
  switch (type) {
    case "PIT_STOP":
      return "pit";
    case "SAFETY_CAR_DEPLOYED":
      return "sc";
    case "VSC_DEPLOYED":
      return "vsc";
    case "CAR_RETIRED":
      return "out";
    case "OVERTAKE":
      return "pass";
    default:
      return null;
  }
}

/**
 * Race lap (1-based) in progress at each event, from the leader's lap completions: an event is
 * on lap N+1 if the leader had completed N laps by its tick. Works for events whose payload has
 * no lap (safety car, VSC).
 */
export function lapOfEvents(events: RaceEvent[]): Map<number, number> {
  const firstCompletion = new Map<number, number>(); // lap -> first tick any car completed it
  for (const e of events) {
    if (e.type !== "LAP_COMPLETED") continue;
    const lap = Number(e.payload.lap);
    const known = firstCompletion.get(lap);
    if (known === undefined || e.tick < known) firstCompletion.set(lap, e.tick);
  }
  const ticks = [...firstCompletion.entries()].sort((a, b) => a[0] - b[0]);
  const laps = new Map<number, number>();
  for (const e of events) {
    let done = 0;
    for (const [lap, tick] of ticks) {
      if (tick <= e.tick) done = lap;
      else break;
    }
    laps.set(e.seq, done + 1);
  }
  return laps;
}
