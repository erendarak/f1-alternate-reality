// Timing tower: order, gaps, last lap, tyre (letter + colour, readable without colour), stops.

import type { Car } from "../api/types";
import { fmtLap, gapText } from "../format";

interface Props {
  cars: Car[];
  totalLaps: number;
  selected: string | null;
  onSelect: (driverId: string) => void;
}

const COMPOUND_NAME = {
  SOFT: "Soft",
  MEDIUM: "Medium",
  HARD: "Hard",
  INTERMEDIATE: "Intermediate",
  WET: "Full wet",
} as const;
const COMPOUND_LETTER = { SOFT: "S", MEDIUM: "M", HARD: "H", INTERMEDIATE: "I", WET: "W" } as const;

export function TyreBadge({ compound, age }: { compound: Car["compound"]; age?: number }) {
  const name = COMPOUND_NAME[compound];
  return (
    <span className={`tyre tyre-${compound.toLowerCase()}`} title={`${name}${age != null ? `, ${age} laps old` : ""}`}>
      {COMPOUND_LETTER[compound]}
    </span>
  );
}

function status(car: Car): { text: string; cls: string; title?: string } {
  if (car.status === "RETIRED") return { text: "OUT", cls: "st-out", title: car.retired_reason ?? undefined };
  if (car.status === "FINISHED") return { text: "FIN", cls: "st-fin" };
  if (car.in_pit) return { text: "PIT", cls: "st-pit" };
  if (car.box) return { text: "BOX", cls: "st-box", title: `boxing this lap for ${car.box}` };
  return { text: "", cls: "" };
}

export function TimingTower({ cars, totalLaps, selected, onSelect }: Props) {
  return (
    <section className="tower" aria-label="Timing tower">
      <table>
        <thead>
          <tr>
            <th>P</th>
            <th>Driver</th>
            <th className="num">Gap</th>
            <th className="num">Last lap</th>
            <th>Tyre</th>
            <th className="num" title="pit stops">Stops</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {cars.map((car) => {
            const st = status(car);
            const traffic = car.last_lap_traffic_loss_s > 0.05;
            return (
              <tr
                key={car.driver_id}
                className={`${car.driver_id === selected ? "sel" : ""} ${car.status === "RETIRED" ? "out" : ""}`}
                onClick={() => onSelect(car.driver_id)}
              >
                <td className="pos">{car.position}</td>
                <td>
                  <span className="team-bar" style={{ background: `#${car.team_color}` }} />
                  <span className="code">{car.code}</span>
                </td>
                <td className="num">{gapText(car)}</td>
                <td
                  className={`num ${traffic ? "traffic" : ""}`}
                  title={traffic ? `lost ${car.last_lap_traffic_loss_s.toFixed(2)} s in traffic` : undefined}
                >
                  {fmtLap(car.last_lap_time_s)}
                </td>
                <td>
                  <TyreBadge compound={car.compound} age={car.tyre_age_laps} />
                  <span className="age">{car.tyre_age_laps}</span>
                </td>
                <td
                  className="num"
                  title={
                    car.controller === "USER"
                      ? "strategy: user"
                      : car.controller === "AGENT"
                        ? "strategy: AI policy"
                        : `baseline plan: ${car.plan}`
                  }
                >
                  {car.pit_stops}
                  {car.controller === "USER" ? <sup>U</sup> : car.controller === "AGENT" ? <sup>A</sup> : null}
                </td>
                <td className={`st ${st.cls}`} title={st.title}>
                  {st.text}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="tower-note">
        Gaps are display estimates from track position · {totalLaps} laps
      </div>
    </section>
  );
}
