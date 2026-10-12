// Race events, newest first, with the lap they happened on; plus the lap strip with markers.

import { useMemo } from "react";

import type { Car, RaceEvent } from "../api/types";
import { eventText, lapOfEvents, LOGGED, markerKind } from "../format";

const SHOWN = 200;

function useCodes(cars: Car[]) {
  return useMemo(() => {
    const codes = new Map(cars.map((c) => [c.driver_id, c.code]));
    return (id: string | null | undefined) => (id ? codes.get(id) ?? id : "");
  }, [cars]);
}

export function EventLog({ events, cars }: { events: RaceEvent[]; cars: Car[] }) {
  const code = useCodes(cars);
  const laps = useMemo(() => lapOfEvents(events), [events]);
  const shown = useMemo(
    () => events.filter((e) => LOGGED.has(e.type)).slice(-SHOWN).reverse(),
    [events],
  );
  return (
    <section className="event-log" aria-label="Race events">
      <h2>Race control</h2>
      {shown.length === 0 ? <div className="empty">No events yet</div> : null}
      <ol>
        {shown.map((e) => (
          <li key={e.seq} className={`ev ev-${markerKind(e.type) ?? "other"}`}>
            <span className="ev-lap">L{laps.get(e.seq) ?? "?"}</span>
            {eventText(e, code)}
          </li>
        ))}
      </ol>
    </section>
  );
}

const SYMBOL = { pit: "P", sc: "SC", vsc: "V", out: "✕", pass: "↑" } as const;

export function LapStrip({
  events,
  cars,
  leaderLap,
  totalLaps,
}: {
  events: RaceEvent[];
  cars: Car[];
  leaderLap: number;
  totalLaps: number;
}) {
  const code = useCodes(cars);
  const markers = useMemo(() => {
    const laps = lapOfEvents(events);
    const byLap = new Map<number, { kind: keyof typeof SYMBOL; texts: string[] }[]>();
    for (const e of events) {
      const kind = markerKind(e.type);
      if (kind === null) continue;
      const lap = Math.min(laps.get(e.seq) ?? 1, totalLaps);
      const list = byLap.get(lap) ?? [];
      const same = list.find((m) => m.kind === kind);
      if (same) same.texts.push(eventText(e, code));
      else list.push({ kind, texts: [eventText(e, code)] });
      byLap.set(lap, list);
    }
    return byLap;
  }, [events, code, totalLaps]);
  const done = Math.min(leaderLap - 1, totalLaps);
  return (
    <section className="lap-strip" aria-label="Race timeline by lap">
      <div className="strip-track">
        <div className="strip-progress" style={{ width: `${(100 * done) / totalLaps}%` }} />
        {[...markers.entries()].map(([lap, list]) =>
          list.map((m, i) => (
            <span
              key={`${lap}-${m.kind}`}
              className={`marker m-${m.kind}`}
              style={{ left: `${(100 * (lap - 0.5)) / totalLaps}%`, top: `${2 + i * 14}px` }}
              title={`Lap ${lap}\n${m.texts.join("\n")}`}
            >
              {SYMBOL[m.kind]}
            </span>
          )),
        )}
      </div>
      <div className="strip-scale">
        <span>Lap 1</span>
        <span>Lap {totalLaps}</span>
      </div>
    </section>
  );
}
