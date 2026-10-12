import { useCallback, useEffect, useState } from "react";

import { api } from "./api/client";
import { useWorldStream } from "./api/stream";
import { useCommand } from "./api/useCommand";
import type { RaceEvent, SnapshotMeta, Track } from "./api/types";
import { NoticeLine } from "./components/common";
import { DriverPanel } from "./components/DriverPanel";
import { EventLog, LapStrip } from "./components/EventLog";
import { ForkDialog } from "./components/ForkDialog";
import { RaceControls } from "./components/RaceControls";
import { RaceHeader } from "./components/RaceHeader";
import { Timelines } from "./components/Timelines";
import { TimingTower } from "./components/TimingTower";
import { TrackCanvas } from "./components/TrackCanvas";
import { WorldLibrary } from "./components/WorldLibrary";

type Tab = "driver" | "events" | "timelines";

export function App() {
  const [worldId, setWorldId] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [track, setTrack] = useState<{ raceId: string; track: Track } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("events");
  const [fork, setFork] = useState<{ source: SnapshotMeta | null } | null>(null);
  const [engine, setEngine] = useState<string | null>(null);
  const timelineCmd = useCommand();

  // attach to the backend's live world if there is one
  useEffect(() => {
    api
      .health()
      .then((h) => {
        setEngine(h.engine_version);
        setWorldId(h.live_world_id);
      })
      .catch(() => undefined)
      .finally(() => setChecked(true));
  }, []);

  const fetchHistory = useCallback(
    async (id: string): Promise<RaceEvent[]> => (await api.events(id)).events,
    [],
  );
  const { stream, frames } = useWorldStream(worldId, fetchHistory);

  // the world stopped being live: follow the new live world, or go back to the library
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

  const select = (id: string | null) => {
    setSelected(id);
    if (id) setTab("driver");
  };

  if (!checked) return null;
  if (worldId === null) {
    return (
      <WorldLibrary
        message={message}
        onOpen={(id) => {
          setMessage(null);
          setSelected(null);
          setWorldId(id);
        }}
      />
    );
  }
  if (state === null) {
    return <main className="library">Connecting to the race… ({stream.connection})</main>;
  }
  const selectedCar = state.cars.find((c) => c.driver_id === selected) ?? null;

  return (
    <div className="app">
      <RaceHeader
        state={state}
        connection={stream.connection}
        onLibrary={() => {
          setMessage(null);
          setWorldId(null); // leaving closes the stream: a running race auto-pauses (D-027)
        }}
      />
      <div className="layout">
        <div className="left">
          {track && track.raceId === state.event_id ? (
            <TrackCanvas track={track.track} frames={frames} selected={selected} onSelect={select} />
          ) : (
            <div className="track-wrap">Loading track…</div>
          )}
          <RaceControls state={state} onFork={() => setFork({ source: null })} />
          <LapStrip
            events={stream.events}
            cars={state.cars}
            leaderLap={state.leader_lap}
            totalLaps={state.total_laps}
          />
        </div>
        <div className="tower-col">
          <TimingTower
            cars={state.cars}
            totalLaps={state.total_laps}
            selected={selected}
            onSelect={(id) => select(id === selected ? null : id)}
          />
        </div>
        <div className="side-col">
          <section className="side-panel">
            <div className="tabs" role="tablist">
              {(["driver", "events", "timelines"] as Tab[]).map((t) => (
                <button
                  key={t}
                  role="tab"
                  aria-selected={tab === t}
                  className={tab === t ? "tab active" : "tab"}
                  onClick={() => setTab(t)}
                >
                  {t === "driver" ? (selectedCar ? selectedCar.code : "Driver") : t === "events" ? "Race control" : "Timelines"}
                </button>
              ))}
            </div>
            <div className="tab-body" role="tabpanel">
              {tab === "driver" ? (
                selectedCar ? (
                  <DriverPanel worldId={state.world_id} car={selectedCar} timelineId={state.timeline_id} />
                ) : (
                  <div className="empty">Click a car on the map or a row in the tower.</div>
                )
              ) : tab === "events" ? (
                <EventLog events={stream.events} cars={state.cars} />
              ) : (
                <>
                  <NoticeLine notice={timelineCmd.notice} />
                  <Timelines
                    worldId={state.world_id}
                    engineVersion={engine}
                    refreshKey={`${state.timeline_id}|${state.persistence?.last_autosave}|${timelineCmd.notice?.text}`}
                    busy={timelineCmd.pending !== null}
                    onLoad={(s) =>
                      void timelineCmd.request(
                        "load",
                        () => api.load(state.world_id, s.snapshot_id),
                        (r) =>
                          r.branched
                            ? `Loaded “${s.label}” into a new branch (that timeline's later history is kept)`
                            : `Loaded “${s.label}”`,
                        "Loading…",
                      )
                    }
                    onFork={(s) => setFork({ source: s })}
                  />
                </>
              )}
            </div>
          </section>
        </div>
      </div>
      {fork ? (
        <ForkDialog
          worldId={state.world_id}
          cars={state.cars}
          driverId={selected}
          leaderLap={state.leader_lap}
          source={fork.source}
          onClose={() => setFork(null)}
          onForked={(text) => {
            setFork(null);
            timelineCmd.setNotice({ kind: "ok", text });
            setTab("timelines");
          }}
        />
      ) : null}
    </div>
  );
}
