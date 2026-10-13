// World Library (plan §14.1): saved worlds, continue / load a save, create a new race.

import { useEffect, useState } from "react";

import { api } from "../api/client";
import { errorText, useCommand } from "../api/useCommand";
import type { Health, RaceSummary, WorldSummary } from "../api/types";
import { fmtClock } from "../format";
import { NoticeLine } from "./common";
import { Timelines } from "./Timelines";

interface Props {
  onOpen: (worldId: string) => void;
  message: string | null;
}

function created(utc: string): string {
  const d = new Date(utc);
  return Number.isNaN(d.getTime()) ? utc : d.toLocaleString();
}

export function WorldLibrary({ onOpen, message }: Props) {
  const [races, setRaces] = useState<RaceSummary[]>([]);
  const [worlds, setWorlds] = useState<WorldSummary[] | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [raceId, setRaceId] = useState("");
  const [seed, setSeed] = useState("");
  const [speed, setSpeed] = useState(5);
  const [weather, setWeather] = useState<"historical" | "random">("historical");
  const [loadError, setLoadError] = useState<string | null>(null);
  const { pending, notice, request } = useCommand();

  useEffect(() => {
    Promise.all([api.races(), api.worlds(), api.health()])
      .then(([r, w, h]) => {
        setRaces(r.races);
        setRaceId((current) => current || r.races[0]?.race_id || "");
        setWorlds(w.worlds);
        setHealth(h);
      })
      .catch((e: unknown) => setLoadError(errorText(e)));
  }, []);

  const continueWorld = async (w: WorldSummary) => {
    if (w.live) return onOpen(w.world_id); // already loaded: just watch it
    const r = await request("load", () => api.load(w.world_id), () => "Loaded", "Loading…");
    if (r) onOpen(w.world_id);
  };

  const create = async () => {
    const parsed = seed.trim() === "" ? undefined : Number(seed);
    if (parsed !== undefined && (!Number.isInteger(parsed) || parsed < 0)) return;
    const r = await request(
      "create",
      () => api.createWorld(raceId, speed, parsed, weather),
      (x) => `Created (seed ${x.seed})`,
      "Creating…",
    );
    if (r) onOpen(r.world_id);
  };

  const engine = health?.engine_version ?? null;
  const seedInvalid = seed.trim() !== "" && !/^\d+$/.test(seed.trim());

  return (
    <main className="library">
      <header className="lib-head">
        <div>
          <div className="brand">F1 ALTERNATE REALITY</div>
          <h1>World Library</h1>
          <p className="sub">
            Pausable, saveable, forkable races on real 2024 grids. Everything that happens in a race
            is simulated, not the real result.
          </p>
        </div>
        {health ? (
          <div className="meta right">
            engine {health.engine_version} · world schema v{health.world_schema_version} · store v
            {health.store_schema_version}
          </div>
        ) : null}
      </header>
      {message ? <p className="notice">{message}</p> : null}
      {loadError ? <p className="notice error">{loadError}</p> : null}

      <section className="panel new-race" aria-label="New race">
        <h2>New race</h2>
        <div className="form-row">
          <label>
            Race
            <select value={raceId} onChange={(e) => setRaceId(e.target.value)}>
              {races.map((r) => (
                <option key={r.race_id} value={r.race_id}>
                  {r.name} · {r.total_laps} laps
                </option>
              ))}
            </select>
          </label>
          <label>
            Seed
            <input
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="random"
              inputMode="numeric"
              aria-invalid={seedInvalid}
              size={10}
            />
          </label>
          <label>
            Speed
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
              {[1, 5, 10, 25].map((v) => (
                <option key={v} value={v}>
                  x{v}
                </option>
              ))}
            </select>
          </label>
          <label>
            Weather
            <select value={weather} onChange={(e) => setWeather(e.target.value as typeof weather)}>
              <option value="historical">Real race conditions</option>
              <option value="random">Random showers (what-if)</option>
            </select>
          </label>
          <button className="primary" disabled={pending !== null || !raceId || seedInvalid} onClick={create}>
            Create
          </button>
        </div>
        <p className="meta">
          Full 20-car grid with the real starting order and tyres; car pace, base lap and pit loss
          estimated from that race's data; driver personas are editorial assumptions.
        </p>
      </section>

      <section aria-label="Saved worlds">
        <h2>Saved worlds</h2>
        {worlds === null ? <div className="empty">Loading…</div> : null}
        {worlds?.length === 0 ? <div className="empty">No saved worlds yet.</div> : null}
        <ul className="world-list">
          {worlds?.map((w) => {
            const latest = w.latest;
            const incompatible = engine !== null && latest !== null && latest.engine_version !== engine;
            return (
              <li key={w.world_id} className="panel world-card">
                <div className="wc-main">
                  <div className="wc-title">
                    {w.race_name ?? w.event_id}
                    {w.live ? <span className="pill live">live</span> : null}
                  </div>
                  <div className="meta">
                    created {created(w.created_utc)} · {w.timeline_count} timeline
                    {w.timeline_count === 1 ? "" : "s"} · {w.snapshot_count} saves
                  </div>
                  {latest ? (
                    <div className="meta">
                      newest: “{latest.label}” at {fmtClock(latest.sim_clock_s)} ·{" "}
                      {latest.world_status.toLowerCase()}
                    </div>
                  ) : null}
                  {incompatible ? (
                    <div className="meta warn">
                      Saved by {latest?.engine_version}; this backend runs {engine}. Not loadable (saves
                      aren't migrated between engine versions).
                    </div>
                  ) : null}
                  {!w.race_available ? <div className="meta warn">Race data not fetched on this machine.</div> : null}
                </div>
                <div className="wc-actions">
                  <button
                    className="primary"
                    disabled={pending !== null || incompatible || !w.race_available}
                    onClick={() => continueWorld(w)}
                  >
                    {w.live ? "Open" : "Continue"}
                  </button>
                  <button
                    aria-expanded={open === w.world_id}
                    onClick={() => setOpen(open === w.world_id ? null : w.world_id)}
                  >
                    Saves &amp; timelines
                  </button>
                </div>
                {open === w.world_id ? (
                  <Timelines
                    worldId={w.world_id}
                    engineVersion={engine}
                    refreshKey={0}
                    busy={pending !== null || !w.race_available}
                    onLoad={async (s) => {
                      const r = await request(
                        "load",
                        () => api.load(w.world_id, s.snapshot_id),
                        () => "Loaded",
                        "Loading…",
                      );
                      if (r) onOpen(w.world_id);
                    }}
                  />
                ) : null}
              </li>
            );
          })}
        </ul>
      </section>
      <NoticeLine notice={notice} />
    </main>
  );
}
