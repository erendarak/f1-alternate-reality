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

The debug viewer's track outline, corner markers, display rotation and team colours are all derived
from that FastF1/F1-timing data and follow the same rules (generated locally, not committed).

## Software dependencies (frontend, M3)

Direct dependencies in `frontend/package.json` (checked 2026-10-12): React + React DOM 19.3 (MIT,
shipped in the bundle); Vite 8.3, @vitejs/plugin-react 6.1, Vitest 5.0, @types/* (MIT, dev only);
TypeScript 5.9 (Apache-2.0, dev only). The full dev tree (97 packages) is MIT / Apache-2.0 /
ISC / BSD-3-Clause, plus 12 MPL-2.0 build-tool packages that aren't part of the shipped bundle.

## Visual assets

| Asset | Status | Terms |
|---|---|---|
| Track drawing | Generated from data (above) | n/a |
| Cars in the debug viewer | One pixel sprite (below) recoloured per team + 3-letter codes; dots if the sprite is missing; no logos | n/a |
| Cars in the M3 app | Plain team-coloured circles + 3-letter codes, safety car as an amber "SC" box (drawn in code; no sprites, no logos) | n/a |
| [palm-tree, "Formula 1 inspired 2D pixel art car pack"](https://palm-tree.itch.io/formula-1-inspired-2d-pixel-art-car-pack) | **In use** (debug viewer): files in `assets/third_party/palm-tree-f1-pack/` (see its `SOURCE.md`) | Read 2026-10-12. No formal licence on the page. The author's replies: commercial use *"Yes, you absolutely can"*; *"feel free to use it however you want. there's no charge."* Pay-what-you-want; the author asks for a link to games using it. 4 cars "inspired by" 2021 Red Bull/Mercedes/McLaren/Ferrari. Only `Ferrari/Ferrari.png` is used, as a neutral base recoloured per team in the browser (white → transparent, `#D30400` → team colour). Credit "palm-tree" is shown in the viewer footer. |

## Planned (not used yet)

| Source | Notes |
|---|---|
| OpenF1 (`openf1.org`) | FAQ: "intended for educational purposes, personal learning projects, research, and non-commercial fan engagement"; site CC BY-NC-SA 4.0. Free plan: historical data from 2023, all endpoints, no auth, **3 req/s and 30 req/min**. Credit appreciated, not required. |
| TUM race-simulation | **Not used** (D-016). LGPL-3.0. It was run only in an external sandbox for the feasibility probe; no TUM code is in this repo. Modelling ideas may be re-implemented with credit (ideas aren't covered by copyright; copying code would bring LGPL obligations). |
| TUM time-discrete race simulator | Not used (D-016). Apache-2.0/MIT. |
| TUM racetrack-database | Not used yet. LGPL-3.0 data; FastF1-derived outline is used instead. |
| TUM f1-timing-database | CC BY-NC-SA 3.0, non-commercial. |

## Names, marks, persona content

- Real driver and team names are used as identity facts only. No logos or official imagery are included.
- Persona numbers are labelled editorial assumptions (see `ASSUMPTIONS.md`). They are not claims about
  real people.
- F1, FORMULA 1 and team names/logos are trademarks of their owners. This project is not affiliated
  with them.
