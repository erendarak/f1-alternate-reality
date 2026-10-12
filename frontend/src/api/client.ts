// REST client. Commands return the backend's acknowledgement: success is shown only from an
// accepted ack (plan §13.4), never optimistically.

import type {
  Action,
  CommandAck,
  DriverCard,
  ForkResult,
  Health,
  RaceEvent,
  RaceSummary,
  SnapshotMeta,
  TimelineInfo,
  Track,
  WorldState,
  WorldSummary,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const err = (data ?? {}) as { error?: string; detail?: unknown };
    const detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
    throw new ApiError(response.status, err.error ?? "http_error", detail || response.statusText);
  }
  return data as T;
}

/** POST a command; resolves with the ack whether accepted (200) or refused (409). */
async function command(path: string, body: Record<string, unknown> = {}): Promise<CommandAck> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ idempotency_key: crypto.randomUUID(), ...body }),
  });
  const data: unknown = await response.json().catch(() => null);
  if (response.status === 200 || (response.status === 409 && isAck(data))) {
    return data as CommandAck;
  }
  const err = (data ?? {}) as { error?: string; detail?: unknown };
  throw new ApiError(response.status, err.error ?? "http_error", String(err.detail ?? ""));
}

function isAck(data: unknown): boolean {
  return typeof data === "object" && data !== null && "accepted" in data;
}

const world = (id: string) => `/api/worlds/${encodeURIComponent(id)}`;

export const api = {
  races: () => request<{ races: RaceSummary[] }>("GET", "/api/races"),
  track: (raceId: string) => request<Track>("GET", `/api/races/${encodeURIComponent(raceId)}/track`),
  worlds: () =>
    request<{ live_world_id: string | null; worlds: WorldSummary[] }>("GET", "/api/worlds"),
  health: () => request<Health>("GET", "/api/health"),
  createWorld: (raceId: string, speed: number, seed?: number) =>
    request<{ world_id: string; seed: number; state: WorldState }>("POST", "/api/worlds", {
      race_id: raceId,
      speed,
      ...(seed === undefined ? {} : { seed }),
    }),
  /** Make a saved world live; no snapshot = continue from its newest valid save. */
  load: (worldId: string, snapshotId?: string) =>
    request<{ branched: boolean; loaded_snapshot: SnapshotMeta; state: WorldState }>(
      "POST",
      `${world(worldId)}/load`,
      snapshotId ? { snapshot_id: snapshotId } : {},
    ),
  timelines: (worldId: string) =>
    request<{ live_timeline_id: string | null; timelines: TimelineInfo[] }>(
      "GET",
      `${world(worldId)}/timelines`,
    ),
  driver: (worldId: string, driverId: string) =>
    request<DriverCard>("GET", `${world(worldId)}/drivers/${encodeURIComponent(driverId)}`),
  fork: (
    worldId: string,
    body: {
      name: string;
      source_snapshot_id?: string;
      seed_mode: "COMMON_EXOGENOUS_RANDOMNESS" | "INDEPENDENT";
      action_overrides: Action[];
    },
  ) =>
    request<ForkResult>("POST", `${world(worldId)}/fork`, {
      ...body,
      action_overrides: body.action_overrides.map((a) => ({
        ...a,
        idempotency_key: crypto.randomUUID(),
      })),
    }),
  events: (worldId: string, limit = 5000) =>
    request<{ timeline_id: string; events: RaceEvent[] }>(
      "GET",
      `${world(worldId)}/events?limit=${limit}`,
    ),

  start: (worldId: string) => command(`${world(worldId)}/start`),
  pause: (worldId: string) => command(`${world(worldId)}/pause`),
  resume: (worldId: string) => command(`${world(worldId)}/resume`),
  speed: (worldId: string, speed: number) => command(`${world(worldId)}/speed`, { speed }),
  step: (worldId: string, ticks: number) => command(`${world(worldId)}/step`, { ticks }),
  save: (worldId: string, label?: string) =>
    command(`${world(worldId)}/save`, label ? { label } : {}),
  action: (worldId: string, action: Action) =>
    command(`${world(worldId)}/actions`, { ...action }),
};
