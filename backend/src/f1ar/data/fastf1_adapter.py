"""FastF1 adapter: https://docs.fastf1.dev/ (API checked against installed fastf1 3.8.3).

``load_race_session`` is the only function that touches the network (through FastF1's own
cache). Everything else is a pure ``DataFrame -> DataFrame`` normaliser into our conventions:

- snake_case columns, explicit units in names (``_s`` seconds, ``_m`` metres, ``_c`` Celsius);
- times as float seconds of *session time* (FastF1 Timedeltas converted once, here);
- positions converted from FastF1's 1/10 m to metres;
- nullable integer/boolean dtypes, so "missing" stays visibly missing instead of becoming 0/False.

All tables produced here are ``historical_verified`` observations from the official timing feed
as processed by FastF1 (which partly repairs/derives data; see its ``FastF1Generated`` and
``IsAccurate`` flags, kept as columns). ``track_outline`` is ``derived``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

TRACK_STATUS_NAMES: dict[str, str] = {
    "1": "green",
    "2": "yellow",
    "3": "unknown_3",
    "4": "safety_car",
    "5": "red_flag",
    "6": "vsc_deployed",
    "7": "vsc_ending",
}
NEUTRALISED_STATUSES = frozenset({"4", "5", "6", "7"})


@dataclass(frozen=True)
class RaceTables:
    """Normalised tables for one race session."""

    event: dict[str, Any]
    laps: pd.DataFrame
    results: pd.DataFrame
    weather: pd.DataFrame
    track_status: pd.DataFrame
    race_control: pd.DataFrame
    positions: pd.DataFrame
    corners: pd.DataFrame = field(default_factory=lambda: _empty_corners())


class DataSourceError(RuntimeError):
    """The source returned without the data we asked for (FastF1 logs and swallows failures)."""


def load_race_session(year: int, gp: str | int, cache_dir: Path, *, telemetry: bool = True) -> Any:
    """Download (or read from cache) a race session. Returns a loaded ``fastf1.core.Session``.

    FastF1 catches most internal errors, logs "Failed to load ...", and returns a partially
    loaded session. We verify the essentials and fail loudly instead.
    """
    import fastf1  # local import: keeps the pure normalisers importable without side effects
    from fastf1.exceptions import DataNotLoadedError

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    session = fastf1.get_session(year, gp, "R")
    session.load(laps=True, telemetry=telemetry, weather=True, messages=True)
    try:
        n_laps, n_results = len(session.laps), len(session.results)
    except DataNotLoadedError as exc:
        raise DataSourceError(
            f"FastF1 did not load timing data for {year} {gp}; see its log output above"
        ) from exc
    if n_laps == 0 or n_results == 0:
        raise DataSourceError(f"FastF1 returned {n_laps} laps / {n_results} results")
    return session


def tables_from_session(session: Any) -> RaceTables:
    event = {
        "event_name": str(session.event["EventName"]),
        "round": int(session.event["RoundNumber"]),
        "event_date": str(pd.Timestamp(session.event["EventDate"]).date()),
        "session_name": str(session.name),
        "session_start_utc": str(session.date),
        "country": str(session.event["Country"]),
        "location": str(session.event["Location"]),
    }
    from fastf1.exceptions import DataNotLoadedError

    try:
        positions = normalize_positions(session.pos_data)
    except DataNotLoadedError:  # loaded with telemetry=False
        positions = _empty_positions()
    return RaceTables(
        event=event,
        laps=normalize_laps(session.laps),
        results=normalize_results(session.results),
        weather=normalize_weather(session.weather_data),
        track_status=normalize_track_status(session.track_status),
        race_control=normalize_race_control(session.race_control_messages),
        positions=positions,
        corners=normalize_corners(session.get_circuit_info()),
    )


# --- pure normalisers ---------------------------------------------------------------------


def _seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series).dt.total_seconds().astype("Float64")


def _int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _str(series: pd.Series) -> pd.Series:
    out = series.astype("string")
    return out.mask(out.str.strip().isin(["", "nan", "None", "UNKNOWN"]))


def normalize_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "driver_number": _int(laps["DriverNumber"]),
            "driver_code": _str(laps["Driver"]),
            "team": _str(laps["Team"]),
            "lap_number": _int(laps["LapNumber"]),
            "lap_time_s": _seconds(laps["LapTime"]),
            "sector1_s": _seconds(laps["Sector1Time"]),
            "sector2_s": _seconds(laps["Sector2Time"]),
            "sector3_s": _seconds(laps["Sector3Time"]),
            "lap_end_time_s": _seconds(laps["Time"]),
            "lap_start_time_s": _seconds(laps["LapStartTime"]),
            "pit_in_time_s": _seconds(laps["PitInTime"]),
            "pit_out_time_s": _seconds(laps["PitOutTime"]),
            "stint": _int(laps["Stint"]),
            "compound": _str(laps["Compound"]),
            "tyre_life_laps": _int(laps["TyreLife"]),
            "fresh_tyre": laps["FreshTyre"].astype("boolean"),
            "position": _int(laps["Position"]),
            "track_status": _str(laps["TrackStatus"]),
            "speed_trap_kph": pd.to_numeric(laps["SpeedST"], errors="coerce").astype("Float64"),
            "is_accurate": laps["IsAccurate"].astype("boolean"),
            "deleted": laps["Deleted"].astype("boolean"),
            "fastf1_generated": laps["FastF1Generated"].astype("boolean"),
        }
    )
    return df.sort_values(["driver_number", "lap_number"], kind="stable").reset_index(drop=True)


def normalize_results(results: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "driver_number": _int(results["DriverNumber"]),
            "driver_code": _str(results["Abbreviation"]),
            "full_name": _str(results["FullName"]),
            "team_name": _str(results["TeamName"]),
            "grid_position": _int(results["GridPosition"]),
            "position": _int(results["Position"]),
            "classified_position": _str(results["ClassifiedPosition"]),
            "status": _str(results["Status"]),
            "laps": _int(results["Laps"]),
            "points": pd.to_numeric(results["Points"], errors="coerce").astype("Float64"),
            # Official timing colour (hex, no '#'). A colour is data; no logos or liveries are used.
            "team_color": _str(results["TeamColor"])
            if "TeamColor" in results
            else pd.Series(pd.NA, index=results.index, dtype="string"),
        }
    )
    return df.sort_values("position", kind="stable", na_position="last").reset_index(drop=True)


def normalize_weather(weather: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_time_s": _seconds(weather["Time"]),
            "air_temp_c": pd.to_numeric(weather["AirTemp"]).astype("Float64"),
            "track_temp_c": pd.to_numeric(weather["TrackTemp"]).astype("Float64"),
            "humidity_pct": pd.to_numeric(weather["Humidity"]).astype("Float64"),
            "pressure_mbar": pd.to_numeric(weather["Pressure"]).astype("Float64"),
            "rainfall": weather["Rainfall"].astype("boolean"),
            "wind_speed_ms": pd.to_numeric(weather["WindSpeed"]).astype("Float64"),
            "wind_direction_deg": _int(weather["WindDirection"]),
        }
    )


def normalize_track_status(track_status: pd.DataFrame) -> pd.DataFrame:
    status = _str(track_status["Status"])
    return pd.DataFrame(
        {
            "session_time_s": _seconds(track_status["Time"]),
            "status": status,
            "status_name": status.map(TRACK_STATUS_NAMES).astype("string"),
            "message": _str(track_status["Message"]),
        }
    )


def normalize_race_control(messages: pd.DataFrame) -> pd.DataFrame:
    lap = messages["Lap"] if "Lap" in messages else pd.Series(pd.NA, index=messages.index)
    return pd.DataFrame(
        {
            "utc_time": pd.to_datetime(messages["Time"]).astype("string"),
            "category": _str(messages["Category"]),
            "flag": _str(messages["Flag"]),
            "scope": _str(messages["Scope"]),
            "lap": _int(lap),
            "message": _str(messages["Message"]),
        }
    )


def _empty_corners() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number": pd.Series(dtype="Int64"),
            "letter": pd.Series(dtype="string"),
            "x_m": pd.Series(dtype="Float64"),
            "y_m": pd.Series(dtype="Float64"),
            "angle_deg": pd.Series(dtype="Float64"),
            "display_rotation_deg": pd.Series(dtype="Float64"),
        }
    )


def normalize_corners(circuit_info: Any) -> pd.DataFrame:
    """Corner markers from ``Session.get_circuit_info()`` (same 1/10 m frame as positions).

    ``display_rotation_deg`` is FastF1's suggested rotation so the map looks like the usual
    broadcast orientation; it is repeated on every row to keep this a single table.
    """
    if circuit_info is None:
        return _empty_corners()
    c = circuit_info.corners
    return pd.DataFrame(
        {
            "number": _int(c["Number"]),
            "letter": _str(c["Letter"]),
            "x_m": (pd.to_numeric(c["X"]) / 10.0).astype("Float64"),
            "y_m": (pd.to_numeric(c["Y"]) / 10.0).astype("Float64"),
            "angle_deg": pd.to_numeric(c["Angle"]).astype("Float64"),
            "display_rotation_deg": pd.Series(
                float(circuit_info.rotation), index=c.index, dtype="Float64"
            ),
        }
    )


def _empty_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_number": pd.Series(dtype="Int64"),
            "session_time_s": pd.Series(dtype="Float64"),
            "x_m": pd.Series(dtype="Float64"),
            "y_m": pd.Series(dtype="Float64"),
            "z_m": pd.Series(dtype="Float64"),
            "status": pd.Series(dtype="string"),
        }
    )


def normalize_positions(pos_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-driver position samples -> one long table in metres (FastF1 gives 1/10 m)."""
    frames = []
    for number, df in pos_data.items():
        frames.append(
            pd.DataFrame(
                {
                    "driver_number": pd.Series(int(number), index=df.index, dtype="Int64"),
                    "session_time_s": _seconds(df["SessionTime"]),
                    "x_m": (pd.to_numeric(df["X"]) / 10.0).astype("Float64"),
                    "y_m": (pd.to_numeric(df["Y"]) / 10.0).astype("Float64"),
                    "z_m": (pd.to_numeric(df["Z"]) / 10.0).astype("Float64"),
                    "status": _str(df["Status"]),
                }
            )
        )
    if not frames:
        return _empty_positions()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["driver_number", "session_time_s"], kind="stable").reset_index(
        drop=True
    )


def track_outline(
    positions: pd.DataFrame, laps: pd.DataFrame, *, driver_number: int, lap_number: int
) -> pd.DataFrame:
    """X/Y samples (metres) of one car over one lap: a ``derived`` outline for the 2D view."""
    lap = laps[(laps["driver_number"] == driver_number) & (laps["lap_number"] == lap_number)]
    if len(lap) != 1:
        raise ValueError(f"no unique lap {lap_number} for car {driver_number}")
    start, end = lap["lap_start_time_s"].iloc[0], lap["lap_end_time_s"].iloc[0]
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"lap {lap_number} of car {driver_number} has no start/end time")
    car = positions[positions["driver_number"] == driver_number]
    window = car[(car["session_time_s"] >= start) & (car["session_time_s"] <= end)]
    window = window[window["status"] == "OnTrack"]
    outline: pd.DataFrame = window[["session_time_s", "x_m", "y_m", "z_m"]]
    return outline.reset_index(drop=True)
