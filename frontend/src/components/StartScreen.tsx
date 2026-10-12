// Minimal entry point until the World Library (M3 slice 3): continue the newest world or start
// a new race. If the backend already has a live world, the app attaches to it directly.

import { useEffect, useState } from "react";

import { api, ApiError } from "../api/client";
import type { RaceSummary, WorldSummary } from "../api/types";

interface Props {
  onOpen: (worldId: string) => void;
  message: string | null;
}

export function StartScreen({ onOpen, message }: Props) {
  const [races, setRaces] = useState<RaceSummary[]>([]);
  const [latest, setLatest] = useState<WorldSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.races(), api.worlds()])
      .then(([r, w]) => {
        setRaces(r.races);
        setLatest(w.worlds.find((x) => x.race_available) ?? null);
      })
      .catch((e: unknown) => setError(`backend unreachable: ${String(e)}`));
  }, []);

  const act = async (fn: () => Promise<string>) => {
    setBusy(true);
    setError(null);
    try {
      onOpen(await fn());
    } catch (e) {
      setError(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="start">
      <h1>F1 Alternate Reality</h1>
      <p className="sub">
        Pausable, saveable, forkable races on real 2024 grids. Simulated, not the real results.
      </p>
      {message ? <p className="notice">{message}</p> : null}
      {latest ? (
        <button
          className="primary big"
          disabled={busy}
          onClick={() => act(async () => (await api.load(latest.world_id), latest.world_id))}
        >
          Continue {latest.race_name} · {latest.latest?.label ?? "latest"} (tick {latest.latest?.tick ?? 0})
        </button>
      ) : null}
      <h2>New race</h2>
      <ul className="race-list">
        {races.map((r) => (
          <li key={r.race_id}>
            <button
              disabled={busy}
              onClick={() => act(async () => (await api.createWorld(r.race_id, 5)).world_id)}
            >
              {r.name}
            </button>
            <span className="meta">
              {r.total_laps} laps · {r.driver_count} drivers ·{" "}
              {r.has_data ? "data-derived inputs" : "synthetic inputs"}
            </span>
          </li>
        ))}
      </ul>
      {error ? <p className="notice error">{error}</p> : null}
    </main>
  );
}
