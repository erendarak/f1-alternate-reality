import { useCallback, useEffect, useState } from "react";

import { api } from "./api/client";
import { useWorldStream } from "./api/stream";
import type { RaceEvent, Track } from "./api/types";
import { EventLog, LapStrip } from "./components/EventLog";
import { RaceControls } from "./components/RaceControls";
import { RaceHeader } from "./components/RaceHeader";
import { StartScreen } from "./components/StartScreen";
import { TimingTower } from "./components/TimingTower";
import { TrackCanvas } from "./components/TrackCanvas";

export function App() {
  const [worldId, setWorldId] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [track, setTrack] = useState<{ raceId: string; track: Track } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  // attach to the backend's live world if there is one
  useEffect(() => {
    api
      .worlds()
      .then((w) => setWorldId(w.live_world_id))
      .catch(() => undefined)
      .finally(() => setChecked(true));
  }, []);

  const fetchHistory = useCallback(
    async (id: string): Promise<RaceEvent[]> => (await api.events(id)).events,
    [],
  );
  const { stream, frames } = useWorldStream(worldId, fetchHistory);

  // the world stopped being live: follow the new live world, or go back to the start screen
  useEffect(() => {
    if (stream.connection !== "closed" || worldId === null) return;
    if (stream.liveWorldId && stream.liveWorldId !== worldId) {
      setWorldId(stream.liveWorldId);
    } else {
      setMessage("That world is no longer loaded. Continue it or start a new race.");
      setWorldId(null);
    }
  }, [stream.connection, stream.liveWorldId, worldId]);

  const state = stream.frame?.state ?? null;
  const raceId = state?.event_id ?? null;
  useEffect(() => {
    if (raceId === null || track?.raceId === raceId) return;
    api
      .track(raceId)
      .then((t) => setTrack({ raceId, track: t }))
      .catch(() => setMessage(`could not load the ${raceId} track`));
  }, [raceId, track?.raceId]);

  if (!checked) return null;
  if (worldId === null) {
    return (
      <StartScreen
        message={message}
        onOpen={(id) => {
          setMessage(null);
          setWorldId(id);
        }}
      />
    );
  }
  if (state === null) {
    return <main className="start">Connecting to the race… ({stream.connection})</main>;
  }

  return (
    <div className="app">
      <RaceHeader state={state} connection={stream.connection} />
      <div className="layout">
        <div className="left">
          {track && track.raceId === state.event_id ? (
            <TrackCanvas track={track.track} frames={frames} selected={selected} onSelect={setSelected} />
          ) : (
            <div className="track-wrap">Loading track…</div>
          )}
          <RaceControls state={state} />
          <LapStrip
            events={stream.events}
            cars={state.cars}
            leaderLap={state.leader_lap}
            totalLaps={state.total_laps}
          />
        </div>
        <div className="right">
          <TimingTower
            cars={state.cars}
            totalLaps={state.total_laps}
            selected={selected}
            onSelect={(id) => setSelected(id === selected ? null : id)}
          />
          <EventLog events={stream.events} cars={state.cars} />
        </div>
      </div>
    </div>
  );
}
