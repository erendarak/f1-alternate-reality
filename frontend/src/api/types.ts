// Shapes of the backend API v1 (docs/API_CONTRACTS.md). The backend is authoritative: these
// are display data, never a source of truth.

export type WorldStatus = "CREATED" | "RUNNING" | "PAUSED" | "COMPLETED";
export type Compound = "SOFT" | "MEDIUM" | "HARD" | "INTERMEDIATE" | "WET";
export type CarStatus = "RUNNING" | "FINISHED" | "RETIRED";
export type RacePhase = "GREEN" | "VSC" | "SC";

export interface Car {
  position: number;
  driver_id: string;
  code: string;
  team_color: string;
  x: number;
  y: number;
  heading_deg: number;
  lap_fraction: number;
  lap: number;
  laps_completed: number;
  status: CarStatus;
  compound: Compound;
  tyre_age_laps: number;
  pace_mode: string;
  lap_pace_mode: string;
  last_lap_time_s: number | null;
  gap_to_leader_s: number | null;
  laps_down: number | null;
  pit_stops: number;
  box: Compound | null;
  in_pit: boolean;
  controller: "BASELINE" | "USER" | "AGENT";
  plan: string;
  last_lap_traffic_loss_s: number;
  retired_reason: string | null;
}

export interface RaceControl {
  phase: RacePhase;
  safety_car: { x: number; y: number; heading_deg: number; lap_fraction: number } | null;
  sc_laps_left: number | null;
  vsc_left_s: number | null;
  cause: string | null;
  sc_count: number;
  vsc_count: number;
}

export interface Persistence {
  store: boolean;
  last_autosave: string | null;
  unsaved_events: number;
  unsaved_commands: number;
  error: string | null;
}

export interface Weather {
  mode: "HISTORICAL" | "RANDOM";
  raining: boolean;
  wetness: number;
  best_tyre: Compound;
  crossovers: [number, number];
  forecast_rain_10min: number;
  source_type: SourceType;
}

export interface WorldState {
  mode: string;
  event_id: string;
  race_name: string;
  world_id: string;
  timeline_id: string;
  parent_timeline_id: string | null;
  seed: number;
  status: WorldStatus;
  tick: number;
  sim_clock_s: number;
  speed: number;
  leader_lap: number;
  total_laps: number;
  engine_version: string;
  race_control: RaceControl;
  weather?: Weather;
  cars: Car[];
  persistence?: Persistence;
}

export interface RaceEvent {
  seq: number;
  tick: number;
  type: string;
  driver_id: string | null;
  payload: Record<string, number | string>;
}

export interface Corner {
  number: number;
  letter: string | null;
  x_m: number;
  y_m: number;
  angle_deg: number;
}

export interface Track {
  outline: [number, number][];
  start_finish: [number, number];
  corners: Corner[];
  rotation_deg: number;
  outline_length_m: number;
  engine_length_m: number | null;
  bounds: number[];
  source: string;
  team_colors: Record<string, string>;
}

export interface RaceSummary {
  race_id: string;
  name: string;
  season: number;
  round: number;
  total_laps: number | null;
  length_m: number | null;
  driver_count: number;
  has_data: boolean;
}

export interface SnapshotMeta {
  snapshot_id: string;
  world_id: string;
  timeline_id: string;
  tick: number;
  sim_clock_s: number;
  world_status: WorldStatus;
  label: string;
  kind: "MANUAL" | "AUTOSAVE";
  created_utc: string;
  engine_version: string;
}

export interface WorldSummary {
  world_id: string;
  event_id: string;
  created_utc: string;
  timeline_count: number;
  snapshot_count: number;
  active_timeline_id: string | null;
  /** Where "Continue" resumes: newest save of the timeline last live. */
  latest: SnapshotMeta | null;
  race_name: string | null;
  race_available: boolean;
  live: boolean;
}

export interface CommandAck {
  idempotency_key: string;
  type: string;
  accepted: boolean;
  reason: string | null;
  applied_tick: number;
  world_status: WorldStatus;
  speed: number;
  duplicate: boolean;
  snapshot_id: string | null;
  snapshot?: SnapshotMeta | null;
}

export type StreamMessage =
  | {
      type: "state";
      stream_seq: number;
      generation: number;
      reset: boolean;
      state: WorldState;
      events: RaceEvent[];
    }
  | { type: "closed"; reason: string; live_world_id: string | null }
  | { type: "error"; error: "world_not_live" | "unknown_world"; live_world_id: string | null };

export type SourceType =
  | "historical_verified"
  | "estimated_from_data"
  | "editorial_assumption"
  | "simulated_dynamic"
  | "learned_parameter";

export interface Sourced<T> {
  value: T;
  source_type: SourceType;
  note?: string;
}

export interface DriverCard {
  driver_id: string;
  code: string;
  display_name: string;
  team_id: string;
  team_name: string;
  car_number: number;
  team_color: string | null;
  identity_source_type: SourceType;
  grid_position: Sourced<number>;
  start_compound: Sourced<Compound>;
  car_pace_offset_s: Sourced<number>;
  profile: {
    revision: string;
    content_hash: string;
    source_type: SourceType;
    estimated: boolean;
    confidence: number;
    persona_label: string | null;
    base_template: string;
    overridden_fields: string[];
    explanation: string;
    origin: string;
    values: Record<string, number>;
    stage1_fields: string[];
    used_by_engine: string[];
  };
  policy: { type: string; version: string };
  race: {
    status: CarStatus;
    compound: Compound;
    tyre_age_laps: number;
    pit_stops: number;
    pace_mode: string;
    controller: "BASELINE" | "USER" | "AGENT";
    plan: string;
    source_type: SourceType;
  };
}

export interface TimelineInfo {
  timeline_id: string;
  world_id: string;
  parent_timeline_id: string | null;
  label: string;
  fork_snapshot_id: string | null;
  created_utc: string;
  snapshot_count: number;
  fork_event_seq: number | null;
  event_count: number;
  command_count: number;
  snapshots: SnapshotMeta[];
}

export interface Health {
  status: string;
  engine_version: string;
  world_schema_version: number;
  store_schema_version: number;
  allowed_speeds: number[];
  races: string[];
  live_world_id: string | null;
}

export type ActionType =
  | "PIT_REQUEST"
  | "PIT_CANCEL"
  | "SET_PACE_MODE"
  | "SET_CONTROLLER"
  | "DEPLOY_SC"
  | "DEPLOY_VSC";

export interface Action {
  action_type: ActionType;
  driver_id?: string;
  compound?: Compound;
  pace_mode?: "PUSH" | "BALANCED" | "CONSERVE";
  controller?: "BASELINE" | "USER";
  laps?: number;
  duration_s?: number;
}

export interface ForkResult {
  world_id: string;
  timeline_id: string;
  parent_timeline_id: string;
  source_snapshot_id: string;
  override_acks: CommandAck[];
  overrides_rejected: number;
  state: WorldState;
}
