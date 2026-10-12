// Fork a timeline (plan §13.3, US-04): from now or from a save, with the same or new luck, and
// optionally one decision applied on the branch right away. The source timeline isn't changed.

import { useState } from "react";

import { api } from "../api/client";
import { useCommand } from "../api/useCommand";
import type { Action, Car, Compound, SnapshotMeta } from "../api/types";
import { fmtClock } from "../format";
import { Modal, NoticeLine } from "./common";

type Decision = "none" | "pit" | "pace" | "control";

interface Props {
  worldId: string;
  cars: Car[];
  driverId: string | null;
  leaderLap: number;
  source: SnapshotMeta | null;
  onClose: () => void;
  onForked: (message: string) => void;
}

export function ForkDialog({ worldId, cars, driverId, leaderLap, source, onClose, onForked }: Props) {
  const racing = cars.filter((c) => c.status === "RUNNING");
  const [driver, setDriver] = useState(driverId ?? racing[0]?.driver_id ?? "");
  const [decision, setDecision] = useState<Decision>("pit");
  const [compound, setCompound] = useState<Compound>("HARD");
  const [pace, setPace] = useState<"PUSH" | "BALANCED" | "CONSERVE">("PUSH");
  const [luck, setLuck] = useState<"COMMON_EXOGENOUS_RANDOMNESS" | "INDEPENDENT">(
    "COMMON_EXOGENOUS_RANDOMNESS",
  );
  const code = cars.find((c) => c.driver_id === driver)?.code ?? "?";
  const defaultName =
    decision === "pit"
      ? `What if ${code} boxes for ${compound} (lap ${leaderLap})`
      : decision === "pace"
        ? `What if ${code} switches to ${pace} (lap ${leaderLap})`
        : decision === "control"
          ? `I take over ${code}'s strategy (lap ${leaderLap})`
          : `Branch at lap ${leaderLap}`;
  const [name, setName] = useState("");
  const { pending, notice, request } = useCommand();

  const overrides = (): Action[] => {
    if (decision === "none" || !driver) return [];
    if (decision === "pit") return [{ action_type: "PIT_REQUEST", driver_id: driver, compound }];
    if (decision === "pace") return [{ action_type: "SET_PACE_MODE", driver_id: driver, pace_mode: pace }];
    return [{ action_type: "SET_CONTROLLER", driver_id: driver, controller: "USER" }];
  };

  const submit = async () => {
    const title = name.trim() || defaultName;
    const r = await request(
      "fork",
      () =>
        api.fork(worldId, {
          name: title,
          source_snapshot_id: source?.snapshot_id,
          seed_mode: luck,
          action_overrides: overrides(),
        }),
      () => "Forked",
      "Forking…",
    );
    if (!r) return;
    const refused = r.override_acks.filter((a) => !a.accepted);
    onForked(
      refused.length
        ? `Branch “${title}” created, but the decision was refused: ${refused.map((a) => a.reason).join("; ")}`
        : `Now on branch “${title}”. The original timeline is unchanged.`,
    );
  };

  return (
    <Modal title="Fork timeline" onClose={onClose}>
      <form
        className="dialog-body"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <p className="meta">
          From:{" "}
          {source
            ? `save “${source.label}” at ${fmtClock(source.sim_clock_s)}`
            : "this moment (saved first as the fork point)"}
        </p>
        <fieldset>
          <legend>Decision on the branch</legend>
          <label>
            <input type="radio" checked={decision === "pit"} onChange={() => setDecision("pit")} /> Box this lap for
            <select value={compound} onChange={(e) => setCompound(e.target.value as Compound)} disabled={decision !== "pit"}>
              <option value="SOFT">Soft</option>
              <option value="MEDIUM">Medium</option>
              <option value="HARD">Hard</option>
            </select>
          </label>
          <label>
            <input type="radio" checked={decision === "pace"} onChange={() => setDecision("pace")} /> Pace mode
            <select
              value={pace}
              onChange={(e) => setPace(e.target.value as typeof pace)}
              disabled={decision !== "pace"}
            >
              <option value="PUSH">Push</option>
              <option value="BALANCED">Balanced</option>
              <option value="CONSERVE">Conserve</option>
            </select>
          </label>
          <label>
            <input type="radio" checked={decision === "control"} onChange={() => setDecision("control")} /> Take
            over the strategy (no automatic stops)
          </label>
          <label>
            <input type="radio" checked={decision === "none"} onChange={() => setDecision("none")} /> No decision
            (just branch)
          </label>
          <label className="block">
            Driver
            <select value={driver} onChange={(e) => setDriver(e.target.value)} disabled={decision === "none"}>
              {racing.map((c) => (
                <option key={c.driver_id} value={c.driver_id}>
                  P{c.position} {c.code}
                </option>
              ))}
            </select>
          </label>
        </fieldset>
        <fieldset>
          <legend>Luck on the branch</legend>
          <label>
            <input
              type="radio"
              checked={luck === "COMMON_EXOGENOUS_RANDOMNESS"}
              onChange={() => setLuck("COMMON_EXOGENOUS_RANDOMNESS")}
            />{" "}
            Same luck: only the decision differs (fair comparison)
          </label>
          <label>
            <input type="radio" checked={luck === "INDEPENDENT"} onChange={() => setLuck("INDEPENDENT")} /> New luck
            from here
          </label>
        </fieldset>
        <label className="block">
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder={defaultName} maxLength={120} />
        </label>
        <div className="dialog-actions">
          <NoticeLine notice={notice} />
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button className="primary" type="submit" disabled={pending !== null}>
            Fork
          </button>
        </div>
      </form>
    </Modal>
  );
}
