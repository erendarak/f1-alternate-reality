// Selected driver: identity, data provenance on every value, persona (editorial), live race
// state and the driver/team decisions (plan §14.1-14.3, M3 "profiles' editorial labels").

import { useEffect, useState } from "react";

import { api } from "../api/client";
import { errorText, useCommand } from "../api/useCommand";
import type { Action, Car, Compound, DriverCard } from "../api/types";
import { NoticeLine, SourceTag } from "./common";
import { TyreBadge } from "./TimingTower";

const TRAIT_NAMES: Record<string, string> = {
  race_pace_skill: "Race pace",
  consistency: "Consistency",
  risk_appetite: "Risk appetite",
  tire_management: "Tyre management",
  wet_skill: "Wet skill",
  adaptability: "Adaptability",
  qualifying_skill: "Qualifying",
  overtake_skill: "Overtaking",
  defense_skill: "Defending",
  team_orientation: "Team orientation",
  composure: "Composure",
  ambition: "Ambition",
  sociability: "Sociability",
  negotiation_skill: "Negotiation",
};

function Trait({ name, value, used, estimated }: { name: string; value: number; used: boolean; estimated?: string }) {
  return (
    <div className={`trait ${used ? "" : "unused"}`}>
      <span className="trait-name">
        {TRAIT_NAMES[name] ?? name}
        {estimated ? (
          <>
            {" "}
            <SourceTag source="estimated_from_data" note={estimated} />
          </>
        ) : null}
      </span>
      <span className="bar" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={value} aria-label={name}>
        <span style={{ width: `${Math.round(value * 100)}%` }} />
      </span>
      <span className="trait-value">{value.toFixed(2)}</span>
    </div>
  );
}

interface Props {
  worldId: string;
  car: Car;
  timelineId: string;
}

export function DriverPanel({ worldId, car, timelineId }: Props) {
  const [card, setCard] = useState<DriverCard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const { pending, notice, run } = useCommand();

  useEffect(() => {
    let cancelled = false;
    setError(null);
    api
      .driver(worldId, car.driver_id)
      .then((c) => !cancelled && setCard(c))
      .catch((e: unknown) => !cancelled && setError(errorText(e)));
    return () => {
      cancelled = true;
    };
  }, [worldId, car.driver_id, timelineId]);

  const act = (label: string, action: Action, ok: string) =>
    run(label, () => api.action(worldId, { driver_id: car.driver_id, ...action }), () => ok);
  const racing = car.status === "RUNNING";
  const busy = pending !== null || !racing;

  if (error) return <div className="notice error">{error}</div>;
  if (!card || card.driver_id !== car.driver_id) return <div className="empty">Loading driver…</div>;
  const p = card.profile;
  const used = new Set(p.used_by_engine);
  const traits = Object.entries(p.values);

  return (
    <div className="driver-panel">
      <div className="dp-head">
        <span className="team-bar tall" style={{ background: `#${car.team_color}` }} />
        <div>
          <div className="dp-name">
            {card.display_name} <span className="meta">#{card.car_number}</span>
          </div>
          <div className="meta">
            {card.team_name} <SourceTag source={card.identity_source_type} note="Jolpica entry list" />
          </div>
        </div>
      </div>

      <h3>Race now <SourceTag source="simulated_dynamic" /></h3>
      <dl className="facts">
        <dt>Position</dt>
        <dd>P{car.position}</dd>
        <dt>Tyre</dt>
        <dd>
          <TyreBadge compound={car.compound} age={car.tyre_age_laps} /> {car.tyre_age_laps} laps · {car.pit_stops} stop
          {car.pit_stops === 1 ? "" : "s"}
          {car.box ? ` · boxing for ${car.box}` : ""}
        </dd>
        <dt>Pace</dt>
        <dd>{car.pace_mode.toLowerCase()}</dd>
        <dt>Strategy</dt>
        <dd>
          {car.controller === "USER"
            ? "you (no automatic stops)"
            : car.controller === "AGENT"
              ? "AI policy (learned)"
              : `baseline plan: ${car.plan || "–"}`}
        </dd>
      </dl>

      <div className="dp-actions" aria-label="Decisions">
        <div className="group" role="group" aria-label="Pit">
          {(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"] as Compound[]).map((c) => (
            <button
              key={c}
              disabled={busy}
              title={`Box at the end of this lap for ${c.toLowerCase()} (takes the strategy over)`}
              onClick={() => act("pit", { action_type: "PIT_REQUEST", compound: c }, `${car.code} will box for ${c}`)}
            >
              Box {c === "INTERMEDIATE" ? "I" : c === "WET" ? "W" : c[0]}
            </button>
          ))}
          <button
            disabled={busy || !car.box}
            onClick={() => act("cancel", { action_type: "PIT_CANCEL" }, `${car.code} stays out`)}
          >
            Stay out
          </button>
        </div>
        <div className="group" role="group" aria-label="Pace mode">
          {(["PUSH", "BALANCED", "CONSERVE"] as const).map((m) => (
            <button
              key={m}
              className={car.pace_mode === m ? "active" : ""}
              aria-pressed={car.pace_mode === m}
              disabled={busy}
              onClick={() =>
                act("pace", { action_type: "SET_PACE_MODE", pace_mode: m }, `${car.code}: ${m.toLowerCase()} from next lap`)
              }
            >
              {m[0] + m.slice(1).toLowerCase()}
            </button>
          ))}
        </div>
        <div className="group" role="group" aria-label="Strategy">
          {(["BASELINE", "USER"] as const).map((c) => (
            <button
              key={c}
              className={car.controller === c ? "active" : ""}
              aria-pressed={car.controller === c}
              disabled={busy}
              onClick={() =>
                act("control", { action_type: "SET_CONTROLLER", controller: c }, c === "USER" ? `You run ${car.code}'s strategy` : `Baseline plans ${car.code}'s stops`)
              }
            >
              {c === "USER" ? "My strategy" : "Baseline"}
            </button>
          ))}
        </div>
        <NoticeLine notice={notice} fallback={racing ? "" : "This car is no longer racing."} />
      </div>

      <h3>Race inputs</h3>
      <dl className="facts">
        <dt>Grid</dt>
        <dd>
          P{card.grid_position.value} <SourceTag source={card.grid_position.source_type} />
        </dd>
        <dt>Start tyre</dt>
        <dd>
          {card.start_compound.value.toLowerCase()}{" "}
          <SourceTag source={card.start_compound.source_type} note={card.start_compound.note} />
        </dd>
        <dt>Car pace</dt>
        <dd>
          +{card.car_pace_offset_s.value.toFixed(2)} s/lap vs fastest car{" "}
          <SourceTag source={card.car_pace_offset_s.source_type} note={card.car_pace_offset_s.note} />
        </dd>
      </dl>

      <h3>
        Persona <SourceTag source={p.source_type} note={p.origin} />
      </h3>
      <p className="meta">
        {p.persona_label ? `“${p.persona_label}” · ` : ""}template {p.base_template}
        {p.overridden_fields.length ? ` · adjusted: ${p.overridden_fields.join(", ")}` : " · unmodified template"} ·
        confidence {p.confidence.toFixed(2)}
      </p>
      {p.explanation ? <p className="explain">{p.explanation}</p> : null}
      <div className="traits">
        <div className="meta">Used by the race engine</div>
        {traits
          .filter(([k]) => used.has(k))
          .map(([k, v]) => (
            <Trait key={k} name={k} value={v} used estimated={p.estimated_fields?.[k]} />
          ))}
        <button className="ghost small" aria-expanded={showAll} onClick={() => setShowAll(!showAll)}>
          {showAll ? "Hide" : "Show"} traits not used yet ({traits.length - used.size})
        </button>
        {showAll
          ? traits
              .filter(([k]) => !used.has(k))
              .map(([k, v]) => (
                <Trait key={k} name={k} value={v} used={false} estimated={p.estimated_fields?.[k]} />
              ))
          : null}
      </div>
      <p className="meta">
        Profile {p.revision} · {card.policy.type} {card.policy.version}. Persona values are design
        assumptions, not claims about the real driver, except those tagged as estimated from 2024
        data.
      </p>
    </div>
  );
}
