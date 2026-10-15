import { describe, expect, it } from "vitest";

import { initialStream, MAX_EVENTS, mergeEvents, streamReducer, type StreamState } from "./api/stream";
import type { Car, RaceEvent, StreamMessage, WorldState } from "./api/types";
import { eventText, fmtLap, gapText, lapOfEvents } from "./format";
import { fitView, toScreen, TrackPath } from "./track/geometry";
import { blendFactor, interpolateCars } from "./track/interpolate";

// 1000 m square, like the backend's test track
const SQUARE: [number, number][] = [
  [0, 0],
  [250, 0],
  [250, 250],
  [0, 250],
];

function car(over: Partial<Car> = {}): Car {
  return {
    position: 1, driver_id: "a", code: "AAA", team_color: "ffffff", x: 0, y: 0, heading_deg: 0,
    lap_fraction: 0, lap: 1, laps_completed: 0, status: "RUNNING", compound: "MEDIUM",
    tyre_age_laps: 0, pace_mode: "BALANCED", lap_pace_mode: "BALANCED", last_lap_time_s: null,
    gap_to_leader_s: 0, laps_down: 0, pit_stops: 0, box: null, in_pit: false,
    controller: "BASELINE", plan: "", last_lap_traffic_loss_s: 0, retired_reason: null,
    ...over,
  };
}

function world(cars: Car[], over: Partial<WorldState> = {}): WorldState {
  return {
    mode: "FREE_SIMULATION", event_id: "e", race_name: "R", world_id: "w", timeline_id: "w.main",
    parent_timeline_id: null, seed: 1, status: "RUNNING", tick: 0, sim_clock_s: 0, speed: 1,
    leader_lap: 1, total_laps: 53, engine_version: "v",
    race_control: { phase: "GREEN", safety_car: null, sc_laps_left: null, vsc_left_s: null,
      cause: null, sc_count: 0, vsc_count: 0 },
    cars, ...over,
  };
}

const ev = (seq: number, type = "OVERTAKE", tick = seq, payload: RaceEvent["payload"] = {}): RaceEvent => ({
  seq, tick, type, driver_id: "a", payload,
});

const frameMsg = (seq: number, gen: number, reset: boolean, events: RaceEvent[] = [], timeline = "w.main"): StreamMessage => ({
  type: "state", stream_seq: seq, generation: gen, reset,
  state: world([car()], { timeline_id: timeline, tick: seq }), events,
});

describe("TrackPath", () => {
  const path = new TrackPath(SQUARE);
  it("matches the backend's arc-length rule", () => {
    expect(path.length).toBe(1000);
    expect(path.pointAt(0)).toEqual([0, 0]);
    expect(path.pointAt(0.125)).toEqual([125, 0]);
    expect(path.pointAt(0.3)).toEqual([250, 50]);
    expect(path.pointAt(0.875)).toEqual([0, 125]); // closing segment back to the line
    expect(path.pointAt(1.25)).toEqual(path.pointAt(0.25)); // wraps
  });
  it("heading follows the direction of travel", () => {
    expect(path.headingAt(0.1)).toBeCloseTo(0);
    expect(path.headingAt(0.3)).toBeCloseTo(Math.PI / 2);
  });
  it("fits the outline into the canvas, y up", () => {
    const view = fitView(SQUARE, 0, 300, 300, 25);
    expect(toScreen(view, [0, 0])).toEqual([25, 275]);
    expect(toScreen(view, [250, 250])).toEqual([275, 25]);
  });
});

describe("interpolation", () => {
  const at = (state: WorldState, t: number) => ({ state, at: t });
  it("moves along the track from the previous frame to the newest over one interval", () => {
    const prev = at(world([car({ lap_fraction: 0.9 })]), 0);
    const next = at(world([car({ lap_fraction: 0.1, laps_completed: 1 })]), 100);
    expect(blendFactor(prev, next, 100)).toBe(0);
    const mid = interpolateCars(prev, next, 150)[0]!;
    expect(mid.fraction).toBeCloseTo(0.0); // halfway from 0.9 to 1.1 crosses the line forward
    expect(interpolateCars(prev, next, 400)[0]!.fraction).toBeCloseTo(0.1);
  });
  it("shows the exact frame when paused, for pitting cars and for jumps", () => {
    const prev = at(world([car({ lap_fraction: 0.2 })]), 0);
    const paused = at(world([car({ lap_fraction: 0.3 })], { status: "PAUSED" }), 100);
    expect(interpolateCars(prev, paused, 150)[0]!.fraction).toBe(0.3);
    const pit = at(world([car({ lap_fraction: 0.3, in_pit: true })]), 100);
    expect(interpolateCars(prev, pit, 150)[0]!.fraction).toBe(0.3);
    const jump = at(world([car({ lap_fraction: 0.1 })]), 100); // backwards: snap
    expect(interpolateCars(prev, jump, 150)[0]!.fraction).toBe(0.1);
  });
});

describe("stream reducer", () => {
  const apply = (s: StreamState, m: StreamMessage, at = 0) => streamReducer(s, { kind: "message", message: m, at });

  it("drops stale frames and appends events", () => {
    let s = apply(initialStream, frameMsg(1, 1, true, [ev(0), ev(1)]));
    s = apply(s, frameMsg(2, 1, false, [ev(2)]));
    const stale = apply(s, frameMsg(2, 1, false, [ev(9)]));
    expect(stale).toBe(s);
    expect(s.events.map((e) => e.seq)).toEqual([0, 1, 2]);
    expect(s.previous?.state.tick).toBe(1);
  });
  it("a reconnect to the same runner keeps the history; a fork replaces it", () => {
    let s = apply(initialStream, frameMsg(1, 1, true, [ev(0), ev(1), ev(2)]));
    s = streamReducer(s, { kind: "connecting" });
    expect(s.connection).toBe("reconnecting");
    s = apply(s, frameMsg(1, 1, true, [ev(2), ev(3)])); // new connection: seq restarts at 1
    expect(s.events.map((e) => e.seq)).toEqual([0, 1, 2, 3]);
    s = apply(s, frameMsg(2, 2, true, [ev(0), ev(1)], "branch"));
    expect(s.events.map((e) => e.seq)).toEqual([0, 1]);
    expect(s.previous).toBeNull(); // never blend two timelines
  });
  it("history is merged only into the timeline it was fetched for", () => {
    const s = apply(initialStream, frameMsg(1, 1, true, [ev(5)]));
    expect(streamReducer(s, { kind: "history", events: [ev(4)], timelineId: "w.main" }).events).toHaveLength(2);
    expect(streamReducer(s, { kind: "history", events: [ev(4)], timelineId: "other" })).toBe(s);
  });
  it("closed and error messages end the stream with the live world id", () => {
    const s = apply(initialStream, { type: "closed", reason: "world_unloaded", live_world_id: "w2" });
    expect([s.connection, s.liveWorldId]).toEqual(["closed", "w2"]);
  });
  it("requests the full history on resets and on detected gaps", () => {
    let s = apply(initialStream, frameMsg(1, 1, true, [ev(0)]));
    expect(s.historyRequest).toBe(1);
    s = apply(s, frameMsg(2, 1, false, [ev(1)]));
    expect(s.historyRequest).toBe(1); // contiguous: nothing to fetch
    s = apply(s, frameMsg(3, 1, false, [ev(5)]));
    expect(s.historyRequest).toBe(2); // events 2..4 never arrived
    s = apply(s, frameMsg(5, 1, false, [ev(6)]));
    expect(s.historyRequest).toBe(3); // frame 4 never arrived
    s = streamReducer(s, { kind: "history", events: [ev(2), ev(3), ev(4)], timelineId: "w.main" });
    expect(s.events.map((e) => e.seq)).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });
  it("reset forgets a closed stream (reopening the same world must not bounce)", () => {
    const closed = apply(initialStream, { type: "closed", reason: "world_unloaded", live_world_id: null });
    expect(closed.connection).toBe("closed");
    expect(streamReducer(closed, { kind: "reset" })).toEqual(initialStream);
  });
  it("caps the event log", () => {
    const many = Array.from({ length: MAX_EVENTS + 10 }, (_, i) => ev(i));
    expect(mergeEvents(many, [])).toHaveLength(MAX_EVENTS);
  });
});

describe("format", () => {
  it("lap times and gaps", () => {
    expect(fmtLap(81.234)).toBe("1:21.234");
    expect(fmtLap(null)).toBe("–");
    expect(gapText(car({ position: 2, gap_to_leader_s: 3.456 }))).toBe("+3.5");
    expect(gapText(car({ position: 9, laps_down: 2 }))).toBe("+2 LAPS");
    expect(gapText(car({ status: "RETIRED" }))).toBe("OUT");
  });
  it("event lines use driver codes", () => {
    const code = (id: string | null | undefined) => (id === "a" ? "VER" : id === "b" ? "HAM" : "");
    expect(eventText(ev(1, "OVERTAKE", 1, { passed: "b", lap: 3 }), code)).toBe("VER passes HAM");
    expect(eventText({ ...ev(2, "SAFETY_CAR_DEPLOYED", 2, { laps: 4, actor_id: "USER" }), driver_id: null }, code))
      .toBe("SAFETY CAR for 4 laps (race director)");
    // W4: penalties and red flags
    expect(eventText(ev(3, "TIME_PENALTY", 3, { seconds: 10, reason: "collision", lap: 7 }), code))
      .toBe("VER 10 s penalty (collision)");
    expect(eventText(ev(4, "RED_FLAG", 4, { lap: 1 }), code)).toContain("RED FLAG");
    expect(eventText(ev(5, "PIT_STOP", 5, { under: "RED_FLAG", from_compound: "MEDIUM", to_compound: "HARD" }), code))
      .toBe("VER new tyres under the red flag: MEDIUM → HARD");
  });
  it("laps of events from the leader's lap completions", () => {
    const events = [
      ev(0, "LAP_COMPLETED", 160, { lap: 1 }),
      ev(1, "LAP_COMPLETED", 163, { lap: 1 }),
      ev(2, "SAFETY_CAR_DEPLOYED", 200),
      ev(3, "LAP_COMPLETED", 320, { lap: 2 }),
      ev(4, "VSC_DEPLOYED", 100),
    ];
    const laps = lapOfEvents(events);
    expect([laps.get(2), laps.get(4), laps.get(3)]).toEqual([2, 1, 3]);
  });
});
