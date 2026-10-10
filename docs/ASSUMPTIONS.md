# Assumptions

Everything here is a modelling or design assumption, **not** a verified fact.

## Driver personas (2024 Italian GP seed)

Source files: `configs/drivers/archetypes.yaml`, `configs/drivers/seed_2024_monza_drivers.yaml`.

- **All 14 persona values for all 20 drivers are `editorial_assumption`.** They are illustrative
  simulation priors in [0, 1]. They are not psychometrics, telemetry or official ratings, and say nothing
  about what any real person is like.
- Persona values are resolved as **template + per-driver overrides**:
  | Group | Drivers | Where the values come from | confidence |
  |---|---|---|---|
  | Full plan persona | Verstappen | master plan §3.5, all 14 fields | 0.30 |
  | Plan Stage 1 persona | Hamilton, Leclerc, Norris, Piastri, Alonso, Sainz, Russell | 6 Stage 1 fields from plan §3.3; other 8 from template | 0.30 |
  | Template only | Pérez, Hülkenberg, Magnussen, Bottas, Ricciardo (`generic_veteran`); Albon, Gasly, Ocon, Tsunoda, Stroll, Zhou (`generic_balanced`); Colapinto (`generic_rookie`, F1 debut at this race) | template unchanged | 0.15 |
- Template choice (veteran / balanced / rookie) is a rough editorial call based on experience level.
  Template-only drivers sharing a template have **identical** personas, so in the sim they differ only
  through car, grid and later dynamic state. Edit the seed to differentiate them.
- `confidence` is a self-reported editorial confidence, not a statistical quantity.
- Car performance is **not** part of the driver profile (plan §5.2) and will get its own model.
- Changing any value must bump `profile_revision`. Once worlds exist, applying a changed profile to a
  saved world requires a fork (plan §3.4).
