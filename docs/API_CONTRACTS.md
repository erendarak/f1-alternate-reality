# API contracts (v1, M3)

The backend is authoritative (plan §1.3, §13.4). The UI only shows backend state and sends
commands; a command counts as done only when its acknowledgement says so. Interactive docs:
`uv run python scripts/serve.py` → <http://127.0.0.1:8000/docs>. Code: `backend/src/f1ar/api/`.
Decisions: D-027.

## Model

- **One live world per backend** (local single-user app). All other worlds exist only in the
  save file (`saves/worlds.sqlite` by default). Creating, loading or forking makes a world live;
  the previous live world is first paused, autosaved ("autosave: shutdown") and its logs are
  written. Live-only requests for another world return `409 world_not_live`.
- **World** = one race universe (one `world_id`, many timelines). **Timeline** = one history
  (`<world_id>.main`, or a branch). **Snapshot** = a saved, verified canonical state.
- Ticks are 0.5 s of race time. Speeds: 1, 2, 5, 10, 25 (wall-clock pacing only; the results
  don't depend on speed).

## Errors

JSON `{"error": <code>, "detail": <text>, ...}`. Body validation errors use FastAPI's 422 shape.

| error | status | when |
|---|---|---|
| `unknown_race` | 404 | race not in the catalog (not simulated or not fetched) |
| `unknown_world` | 404 | world not in the save file |
| `not_found` | 404 | unknown snapshot, timeline or driver; snapshot of another world |
| `world_not_live` | 409 | live-only request for a stored world (`live_world_id` included) |
| `snapshot_unusable` | 409 | checksum / schema / engine / policy mismatch (D-012, D-025) |
| `command_rejected` | 409 | fork: the fork-point save was refused (`ack` included) |
| `command_timeout` | 504 | runner didn't acknowledge within 5 s |
| `invalid_request` | 422 | a value the domain refuses (e.g. unknown driver code) |

## Command acknowledgements

`start`, `pause`, `resume`, `speed`, `step`, `save` and `actions` return the runner's
`CommandAck`, with **200 if accepted, 409 if the world refused it** (the body is the ack in both
cases):

```json
{"idempotency_key": "web-…", "type": "PAUSE", "accepted": true, "reason": null,
 "applied_tick": 615, "world_status": "PAUSED", "speed": 5, "duplicate": false,
 "snapshot_id": null}
```

- `applied_tick`: the canonical tick boundary where it took effect (or was refused).
- `idempotency_key` (optional in every request body; generated if missing): repeating a key
  returns the original ack with `duplicate: true` and never applies the command twice, also
  after a backend restart (D-025).
- `save` is acknowledged only after the database commit; a failed save is `accepted: false`
  with the reason (US-09). Its response adds `snapshot` (metadata below).

## REST

| Verb | Path | Body | Result |
|---|---|---|---|
| GET | `/api/health` | | versions (engine, world/store schema, policies), races, `live_world_id` |
| GET | `/api/races` | | `races[]` (id, name, season, round, laps, length, `has_data`), `unavailable[]` |
| GET | `/api/races/{race_id}` | | summary + `drivers[]` (verified entry list, `historical_verified`) |
| GET | `/api/races/{race_id}/track` | | outline, corners, rotation, bounds, start/finish, `team_colors` |
| GET | `/api/worlds` | | `live_world_id`, `worlds[]` (event, timeline/snapshot counts, `latest` snapshot, `live`) |
| POST | `/api/worlds` | `{race_id, seed?, drivers?, synthetic?, speed?}` | 201 `{world_id, seed, state}`; new world, live, saved once ("created") |
| GET | `/api/worlds/{id}` | | `{live, state}` (+ `latest_snapshot` for a stored world, read from its newest valid save) |
| GET | `/api/worlds/{id}/timelines` | | `live_timeline_id`, `timelines[]` with lineage, counts and their `snapshots[]` |
| GET | `/api/worlds/{id}/events` | `?timeline_id&since_seq&limit` | a timeline's full history (ancestor prefix + own + live unsaved) |
| GET | `/api/worlds/{id}/drivers/{driver_id}` | | driver card, every value with its `source_type` (below) |
| POST | `/api/worlds/{id}/start` · `/pause` · `/resume` | `{idempotency_key?}` (optional) | ack |
| POST | `/api/worlds/{id}/speed` | `{speed}` | ack |
| POST | `/api/worlds/{id}/step` | `{ticks: 1..10000}` | ack; only while PAUSED |
| POST | `/api/worlds/{id}/save` | `{label?}` | ack + `snapshot` |
| POST | `/api/worlds/{id}/actions` | action (below) | ack |
| POST | `/api/worlds/{id}/load` | `{snapshot_id?, speed?}` | `{loaded_snapshot, timeline_id, branched, state}` |
| POST | `/api/worlds/{id}/fork` | fork (below) | 201 `{timeline_id, parent_timeline_id, source_snapshot_id, branch_point, override_acks, overrides_rejected, state}` |

### Actions (plan §13.3)

```json
{"action_type": "PIT_REQUEST", "driver_id": "lando_norris_2024", "compound": "HARD",
 "actor_id": "team_mclaren_2024", "idempotency_key": "uuid-from-client"}
```

| action_type | fields | effect |
|---|---|---|
| `PIT_REQUEST` | `driver_id`, `compound` | box at the end of the current lap; the team takes the car's strategy over |
| `PIT_CANCEL` | `driver_id` | stay out this lap |
| `SET_PACE_MODE` | `driver_id`, `pace_mode` (PUSH/BALANCED/CONSERVE) | from the next lap |
| `SET_CONTROLLER` | `driver_id`, `controller` (BASELINE/USER) | BASELINE replans the stops |
| `DEPLOY_SC` | `laps` (1-20, default 4) | race director: safety car now |
| `DEPLOY_VSC` | `duration_s` (≤600, default 90) | race director: VSC now |

`actor_id` defaults to `USER` and is logged with the command (US-08). `SYSTEM` is used by the
backend itself (auto-pause, pause-on-load).

### Load

- No `snapshot_id` = **Continue** from the world's newest valid snapshot (taken after the live
  runner's shutdown autosave, so continuing the live world doesn't branch).
- `SnapshotStore.resume` decides the timeline: the snapshot's own timeline continues if nothing
  state-changing happened after it; otherwise a **new branch** is created (`branched: true`,
  label "resumed from: …"). History is never rewritten.
- A loaded world is always **PAUSED**. A save taken while running (e.g. the last autosave
  before a crash) gets a logged `SYSTEM` PAUSE command.

### Fork (plan §13.3)

```json
{"name": "Monza 2024 — Norris pits on lap 18", "source_snapshot_id": "snap_…",
 "seed_mode": "COMMON_EXOGENOUS_RANDOMNESS",
 "action_overrides": [{"action_type": "PIT_REQUEST", "driver_id": "lando_norris_2024",
                       "compound": "HARD"}]}
```

- No `source_snapshot_id`: for the live world, the current tick boundary is saved first
  ("fork point: <name>") and forked from; for a stored world, its newest valid snapshot.
- `seed_mode`: `COMMON_EXOGENOUS_RANDOMNESS` (default: same luck, only the decision differs) or
  `INDEPENDENT` (fresh luck from the branch point), see `simulation/timeline.py`.
- The branch becomes the live world (PAUSED, or CREATED if forked before the start); the
  source timeline is untouched. `action_overrides` are applied in order as logged commands on
  the branch; one that the world refuses is reported in `override_acks` (the fork still exists).

### Driver card (`source_type` labels, plan §3.4)

`identity_source_type` and `grid_position` are `historical_verified`; `start_compound` and
`car_pace_offset_s` carry the source named by the world's input notes (`historical_verified` /
`estimated_from_data`, or `editorial_assumption` for synthetic defaults); `profile.source_type`
is the persona's (`editorial_assumption` today) with `confidence`, `persona_label`,
`base_template`, `overridden_fields`, `explanation`; `race` (live tyre, stint, plan) is
`simulated_dynamic`. Values come from the profile revision pinned in the world, not from the
current config files.

## State view (`state` in responses and stream frames)

`world_id, timeline_id, parent_timeline_id, mode, event_id, race_name, seed, status, tick,
sim_clock_s, speed, leader_lap, total_laps, engine_version, race_control{phase, safety_car{x,y,
heading_deg}, sc_laps_left, vsc_left_s, cause, sc_count, vsc_count}, cars[]` (+ `persistence`
for the live world: last autosave, unsaved log records, last error).

Each car: `position, driver_id, code, team_color, x, y, heading_deg` (track coordinates in
metres, same frame as `/track`), `lap, laps_completed, status, compound, tyre_age_laps,
pace_mode, lap_pace_mode, last_lap_time_s, gap_to_leader_s` (display estimate),
`laps_down, pit_stops, box, in_pit, controller, plan, last_lap_traffic_loss_s,
retired_reason`. This is a spectator view and may show what a driver can't know.

## WebSocket `/ws/worlds/{world_id}`

The server pushes; client messages are ignored (commands go through REST).

| message | when |
|---|---|
| `{"type":"state","stream_seq","generation","reset","state","events"}` | on any change (tick, status, speed, timeline, save status), at most every 100 ms, and at least every 1 s |
| `{"type":"closed","reason":"world_unloaded","live_world_id"}` | another world became live; socket closed with code 4000 |
| `{"type":"error","error":"world_not_live"\|"unknown_world","live_world_id"}` | on connect; closed with 4409 / 4404 |

- `stream_seq` increases by 1 per message on a connection: drop anything older than the last
  one you applied (plan §13.4).
- `events`: race events since the previous frame. When `reset` is true (first frame, and after a
  fork or load replaced the runner: `generation` changed), `events` is the timeline's most
  recent history (up to 50) and the client should replace its log.
- **Auto-pause:** when the last client of a running world disconnects, the backend pauses it
  with a logged `SYSTEM` command (MVP policy, plan §13.4). A browser reload therefore pauses
  the race; press Play again.
