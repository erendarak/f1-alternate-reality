// A world's timelines (main + branches, as a tree) and their saves, with Load / Fork actions.

import { useEffect, useState } from "react";

import { api } from "../api/client";
import { errorText } from "../api/useCommand";
import type { SnapshotMeta, TimelineInfo } from "../api/types";
import { fmtClock } from "../format";

/** Timelines in tree order (parents before their branches) with their depth. */
export function timelineTree(timelines: TimelineInfo[]): { t: TimelineInfo; depth: number }[] {
  const ids = new Set(timelines.map((t) => t.timeline_id));
  const children = new Map<string | null, TimelineInfo[]>();
  for (const t of timelines) {
    const parent = t.parent_timeline_id && ids.has(t.parent_timeline_id) ? t.parent_timeline_id : null;
    children.set(parent, [...(children.get(parent) ?? []), t]);
  }
  const out: { t: TimelineInfo; depth: number }[] = [];
  const walk = (parent: string | null, depth: number) => {
    for (const t of children.get(parent) ?? []) {
      out.push({ t, depth });
      walk(t.timeline_id, depth + 1);
    }
  };
  walk(null, 0);
  return out;
}

interface Props {
  worldId: string;
  engineVersion: string | null;
  refreshKey: unknown;
  busy: boolean;
  onLoad: (snapshot: SnapshotMeta) => void;
  onFork?: (snapshot: SnapshotMeta) => void;
}

export function Timelines({ worldId, engineVersion, refreshKey, busy, onLoad, onFork }: Props) {
  const [data, setData] = useState<{ live: string | null; timelines: TimelineInfo[] } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .timelines(worldId)
      .then((r) => !cancelled && setData({ live: r.live_timeline_id, timelines: r.timelines }))
      .catch((e: unknown) => !cancelled && setError(errorText(e)));
    return () => {
      cancelled = true;
    };
  }, [worldId, refreshKey]);

  if (error) return <div className="notice error">{error}</div>;
  if (!data) return <div className="empty">Loading timelines…</div>;
  return (
    <ul className="timelines">
      {timelineTree(data.timelines).map(({ t, depth }) => (
        <li key={t.timeline_id} style={{ marginLeft: depth * 16 }}>
          <div className="tl-head">
            <span className="tl-name">{t.parent_timeline_id ? `↳ ${t.label}` : "Main timeline"}</span>
            {t.timeline_id === data.live ? <span className="pill live">live</span> : null}
            <span className="meta">
              {t.snapshot_count} saves · {t.command_count} decisions
            </span>
          </div>
          <ul className="snaps">
            {t.snapshots.map((s) => {
              const incompatible = engineVersion !== null && s.engine_version !== engineVersion;
              return (
                <li key={s.snapshot_id} className={incompatible ? "incompatible" : ""}>
                  <span className={`kind kind-${s.kind.toLowerCase()}`}>{s.kind === "AUTOSAVE" ? "auto" : "save"}</span>
                  <span className="snap-label" title={s.snapshot_id}>
                    {s.label}
                  </span>
                  <span className="meta">
                    {fmtClock(s.sim_clock_s)} · {s.world_status.toLowerCase()}
                  </span>
                  {incompatible ? (
                    <span
                      className="meta warn"
                      title={`Saved by ${s.engine_version}; this backend runs ${engineVersion}. Saves aren't migrated between engine versions (D-014).`}
                    >
                      older engine
                    </span>
                  ) : (
                    <span className="snap-actions">
                      <button disabled={busy} onClick={() => onLoad(s)}>
                        Load
                      </button>
                      {onFork ? (
                        <button disabled={busy} onClick={() => onFork(s)}>
                          Fork…
                        </button>
                      ) : null}
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        </li>
      ))}
    </ul>
  );
}
