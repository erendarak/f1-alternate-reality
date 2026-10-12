// WebSocket state stream (docs/API_CONTRACTS.md): a pure reducer plus a React hook.
//
// Ordering rule (plan §13.4): on one connection `stream_seq` only grows; anything not newer than
// the last applied frame is dropped. A new connection starts a new sequence. A `reset` frame
// (every new connection, fork, load) replaces the event log, unless it is the same runner and
// timeline as before (a reconnect), where it is merged so no history is lost.
//
// Self-healing: event seqs are contiguous on a timeline and frames are numbered, so a missed
// frame or an event gap is detectable; it (and every reset) requests the full stored history
// from REST (`historyRequest` increases), which is merged in.

import { useEffect, useReducer, useRef } from "react";

import type { RaceEvent, StreamMessage, WorldState } from "./types";

export const MAX_EVENTS = 5000;

export type Connection = "connecting" | "open" | "reconnecting" | "closed";

export interface Frame {
  state: WorldState;
  /** performance.now() when the frame arrived (for display interpolation). */
  at: number;
}

export interface StreamState {
  connection: Connection;
  lastSeq: number;
  generation: number;
  frame: Frame | null;
  previous: Frame | null;
  events: RaceEvent[];
  /** Set when the world stopped being live (another world was loaded) or never was. */
  closedReason: string | null;
  liveWorldId: string | null;
  /** Increases whenever the full history should be (re)fetched: resets and detected gaps. */
  historyRequest: number;
}

export type StreamAction =
  | { kind: "connecting" }
  | { kind: "message"; message: StreamMessage; at: number }
  | { kind: "history"; events: RaceEvent[]; timelineId: string }
  | { kind: "dropped" }
  /** The hook stopped watching this world: forget it, so reopening it starts clean. */
  | { kind: "reset" };

export const initialStream: StreamState = {
  connection: "connecting",
  lastSeq: 0,
  generation: -1,
  frame: null,
  previous: null,
  events: [],
  closedReason: null,
  liveWorldId: null,
  historyRequest: 0,
};

/** Merge event lists by `seq` (dedupe, keep order), capped to the newest MAX_EVENTS. */
export function mergeEvents(a: RaceEvent[], b: RaceEvent[]): RaceEvent[] {
  const bySeq = new Map<number, RaceEvent>();
  for (const e of a) bySeq.set(e.seq, e);
  for (const e of b) bySeq.set(e.seq, e);
  return [...bySeq.values()].sort((x, y) => x.seq - y.seq).slice(-MAX_EVENTS);
}

export function streamReducer(s: StreamState, action: StreamAction): StreamState {
  switch (action.kind) {
    case "reset":
      return initialStream;
    case "connecting":
      // new connection = new sequence; keep showing the last frame meanwhile
      return { ...s, connection: s.frame ? "reconnecting" : "connecting", lastSeq: 0 };
    case "dropped":
      return s.connection === "closed" ? s : { ...s, connection: "reconnecting" };
    case "history":
      // full stored history fetched after a reset; ignore it if the timeline changed since
      return s.frame?.state.timeline_id === action.timelineId
        ? { ...s, events: mergeEvents(action.events, s.events) }
        : s;
    case "message": {
      const m = action.message;
      if (m.type === "closed") {
        return { ...s, connection: "closed", closedReason: m.reason, liveWorldId: m.live_world_id };
      }
      if (m.type === "error") {
        return { ...s, connection: "closed", closedReason: m.error, liveWorldId: m.live_world_id };
      }
      if (m.stream_seq <= s.lastSeq) return s; // stale or duplicate frame
      const frame = { state: m.state, at: action.at };
      const sameRun =
        m.generation === s.generation && m.state.timeline_id === s.frame?.state.timeline_id;
      const reset = !sameRun;
      const lastEvent = s.events.length ? s.events[s.events.length - 1]!.seq : null;
      const firstNew = m.events.length ? m.events[0]!.seq : null;
      const gap =
        !reset &&
        (m.stream_seq > s.lastSeq + 1 ||
          (lastEvent !== null && firstNew !== null && firstNew > lastEvent + 1));
      return {
        ...s,
        connection: "open",
        lastSeq: m.stream_seq,
        generation: m.generation,
        // after a reset the old frame belongs to another runner/timeline: don't blend them
        previous: reset ? null : s.frame,
        frame,
        events: reset && m.reset ? m.events.slice(-MAX_EVENTS) : mergeEvents(s.events, m.events),
        closedReason: null,
        historyRequest: reset || gap || m.reset ? s.historyRequest + 1 : s.historyRequest,
      };
    }
  }
}

export interface StreamHandle {
  stream: StreamState;
  /** Latest two frames, readable from an animation loop without re-rendering. */
  frames: React.RefObject<{ frame: Frame | null; previous: Frame | null }>;
}

const RECONNECT_MS = 1000;

/** Connects to `/ws/worlds/{id}`, reconnecting after drops until the world stops being live. */
export function useWorldStream(
  worldId: string | null,
  fetchHistory: (worldId: string) => Promise<RaceEvent[]>,
): StreamHandle {
  const [stream, dispatch] = useReducer(streamReducer, initialStream);
  const frames = useRef<{ frame: Frame | null; previous: Frame | null }>({
    frame: null,
    previous: null,
  });
  frames.current = { frame: stream.frame, previous: stream.previous };

  const timelineId = stream.frame?.state.timeline_id ?? null;
  useEffect(() => {
    if (worldId === null || timelineId === null || stream.historyRequest === 0) return;
    let cancelled = false;
    fetchHistory(worldId)
      .then((events) => {
        if (!cancelled) dispatch({ kind: "history", events, timelineId });
      })
      .catch(() => undefined); // the log then shows the streamed events only
    return () => {
      cancelled = true;
    };
    // refetch only when requested; the timeline id is read at request time
  }, [stream.historyRequest, worldId, fetchHistory]);

  useEffect(() => {
    if (worldId === null) return;
    let socket: WebSocket | null = null;
    let timer: number | undefined;
    let stopped = false;

    const connect = () => {
      dispatch({ kind: "connecting" });
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${location.host}/ws/worlds/${encodeURIComponent(worldId)}`);
      let final = false;
      socket.onmessage = (ev: MessageEvent<string>) => {
        const message = JSON.parse(ev.data) as StreamMessage;
        if (message.type !== "state") final = true;
        dispatch({ kind: "message", message, at: performance.now() });
      };
      socket.onclose = () => {
        if (stopped || final) return;
        dispatch({ kind: "dropped" });
        timer = window.setTimeout(connect, RECONNECT_MS);
      };
    };
    connect();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      socket?.close();
      // Otherwise a "closed" state from this connection would still be there when the same
      // world is opened again, and the app would leave it at once (found in the browser).
      dispatch({ kind: "reset" });
    };
  }, [worldId]);

  return { stream, frames };
}
