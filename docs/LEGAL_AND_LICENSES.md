# Legal and licenses

This is a **non-commercial, single-user research/learning project**. Re-review everything here before any
commercial use or public hosting (plan §6.1).

Code licence for this repository: not chosen yet (all rights reserved until a LICENSE file is added).

## Data sources in use

Terms below were read from each source's own pages on **2026-10-12**. They can change; re-check them
before heavier or different use.

| Source | Used for | Stored in repo? | Terms (as read) |
|---|---|---|---|
| **Jolpica-F1 API** (`api.jolpi.ca`, Ergast successor) | Entry list + grid (`configs/rosters/*.json`); independent results cross-check in the quality report | Yes: small derived facts (names, numbers, teams, grid). Results are fetched, not stored | Free for **non-commercial use**; data licensed **CC BY-NC-SA 4.0** (TERMS.md). Code Apache-2.0. Unauthenticated limits: **4 req/s burst, 500 req/hour**, "will decrease in the future". A **custom User-Agent is required** (we send `f1ar/<version> (non-commercial research project)`). No uptime or correctness guarantee. |
| **FastF1** 3.8.3 (Python library) | Laps, results, weather, track status, race control, car positions for 2024 Italian GP | **No.** Raw cache in `.cache/fastf1/`, processed Parquet in `data/processed/` (both gitignored). Only the manifest (hashes, row counts) and the generated quality report are committed | Library is MIT-licensed. The **data comes from the F1 live-timing feed** (plus Ergast/Jolpica for some fields) and is **not** covered by FastF1's MIT licence. Treat as `research_only`; don't redistribute raw or processed tables. |

Attribution: data from Jolpica-F1 (CC BY-NC-SA 4.0) and from F1 live timing via the FastF1 library.

## Planned (not used yet)

| Source | Notes |
|---|---|
| OpenF1 (`openf1.org`) | FAQ: "intended for educational purposes, personal learning projects, research, and non-commercial fan engagement"; site CC BY-NC-SA 4.0. Free plan: historical data from 2023, all endpoints, no auth, **3 req/s and 30 req/min**. Credit appreciated, not required. |
| TUM race-simulation / racetrack-database | LGPL-3.0 code. Keep any use isolated and attributed (step 8 decides). |
| TUM f1-timing-database | CC BY-NC-SA 3.0, non-commercial. |

## Names, marks, persona content

- Real driver and team names are used as identity facts only. No logos or official imagery are included.
- Persona numbers are labelled editorial assumptions (see `ASSUMPTIONS.md`). They are not claims about
  real people.
- F1, FORMULA 1 and team names/logos are trademarks of their owners. This project is not affiliated
  with them.
