// Display smoothing between stream frames (plan §14.4). Display only: the canonical state is
// whatever the backend last sent; this never feeds back into anything.
//
// Frames arrive at up to 10 Hz. The canvas draws at 60 fps one frame interval behind the newest
// frame, moving each car along the track from its progress in the previous frame to its progress
// in the newest one. Progress = laps_completed + lap_fraction, so a car crossing the line keeps
// moving forward, and following the outline (not a straight line) keeps cars on the track.

import type { Car } from "../api/types";
import type { Frame } from "../api/stream";

/** Never stretch interpolation over more than this (a stalled stream shouldn't creep). */
const MAX_INTERVAL_MS = 500;
const MIN_INTERVAL_MS = 30;

export interface Shown {
  driverId: string;
  /** Lap fraction to draw the car at (0..1). */
  fraction: number;
}

export function progressOf(car: Pick<Car, "laps_completed" | "lap_fraction">): number {
  return car.laps_completed + car.lap_fraction;
}

/** 0..1: how far the display has moved from `previous` to `frame` at time `now`. */
export function blendFactor(previous: Frame, frame: Frame, now: number): number {
  const interval = Math.min(MAX_INTERVAL_MS, Math.max(MIN_INTERVAL_MS, frame.at - previous.at));
  return Math.min(1, Math.max(0, (now - frame.at) / interval));
}

/** Where to draw each car at `now`. Without a usable previous frame, the newest frame as is. */
export function interpolateCars(previous: Frame | null, frame: Frame, now: number): Shown[] {
  const state = frame.state;
  const exact = state.cars.map((c) => ({ driverId: c.driver_id, fraction: c.lap_fraction }));
  if (previous === null || state.status !== "RUNNING") return exact;
  const before = new Map(previous.state.cars.map((c) => [c.driver_id, c]));
  const k = blendFactor(previous, frame, now);
  return state.cars.map((car) => {
    const old = before.get(car.driver_id);
    const to = progressOf(car);
    if (old === undefined || car.status !== "RUNNING" || car.in_pit) {
      return { driverId: car.driver_id, fraction: car.lap_fraction };
    }
    const from = progressOf(old);
    // backwards or more than half a lap in one frame: a jump (e.g. restart), don't animate it
    if (to < from || to - from > 0.5) return { driverId: car.driver_id, fraction: car.lap_fraction };
    const p = from + (to - from) * k;
    return { driverId: car.driver_id, fraction: p - Math.floor(p) };
  });
}
