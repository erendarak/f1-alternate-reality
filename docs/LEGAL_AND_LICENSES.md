# Legal and licenses

This is a **non-commercial, single-user research/learning project**. Re-review everything here before any
commercial use or public hosting (plan §6.1).

Code licence for this repository: not chosen yet (all rights reserved until a LICENSE file is added).

## Data sources in use

| Source | Used for | Stored in repo? | Status |
|---|---|---|---|
| Jolpica-F1 API (`api.jolpi.ca`), Ergast successor | Race entry list + starting grid (`configs/rosters/*.json`) | Yes: small derived facts (names, numbers, teams, grid) | `research_only`. Check the current terms and rate limits at https://github.com/jolpica/jolpica-f1 before heavier use. |

## Planned (not used yet)

| Source | Notes |
|---|---|
| FastF1 | MIT-licensed *code*. The underlying timing data is not covered by that licence. Cache only, not committed. |
| OpenF1 | Personal/educational/non-commercial usage terms. |
| TUM race-simulation / racetrack-database | LGPL-3.0 code. Keep any use isolated and attributed. |
| TUM f1-timing-database | CC BY-NC-SA 3.0, non-commercial. |

## Names, marks, persona content

- Real driver and team names are used as identity facts only. No logos or official imagery are included.
- Persona numbers are labelled editorial assumptions (see `ASSUMPTIONS.md`). They are not claims about
  real people.
- F1, FORMULA 1 and team names/logos are trademarks of their owners. This project is not affiliated
  with them.
