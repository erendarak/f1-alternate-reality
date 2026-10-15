# What makes the difference in real F1 — research and how our world compares

Written 2026-10-15 after M5 slice 3 (EXP-009) showed that our world gives strategy almost nothing
to win. Owner's request: research what creates the differences in real F1 (cars, drivers,
mistakes, fuel, tyres per track and weather, track position, attacking, defending) and adapt the
world to it. This document is the evidence base and the plan; nothing in the engine has changed
yet.

**Sources.** Four web research passes, cross-checked against each other. The main sources are:
- Pirelli, FIA and F1 data.
- McLaren's 2024 strategy review.
- The TUM race-simulation parameter files (Heilmeier et al., 2014-19 cars, github.com/TUMFTM/race-simulation).
- Car-vs-driver studies: van Kesteren & Bergkamp 2023; Bell et al. 2016.
- Dartmouth's lap-1 incident study; f1chronicle pit-loss and undercut data; FastF1-based analyses.

They are combined with **our own measurements on the 2024 lap dataset** (`laps_2024_v1`, 21 dry races, method notes per table).

**Reliability tags:**
- **[M]** measured or official.
- **[A]** analysis with a stated method.
- **[C]** commentary.
- **[ours]** measured by us from 2024 timing.
- **[E]** estimate or assumption.

Web numbers from older car eras (TUM 2014-19) are used as scales and priors, not as 2024 truth.

---

## 1. The short answer

| What decides a real F1 result | Share / size | Evidence |
|---|---|---|
| **The car** | 64-88 % of the variance in results; 2024 front-to-back ~1.5-1.9 % (1.4-1.6 s qualifying, **2.2 s/lap race pace [ours]**) | van Kesteren 2023 [M], Bell 2016 [M], F1Technical/PlanetF1 [A] |
| **Track position** (qualifying, overtaking difficulty) | grid-finish Spearman 0.75 (2010-23) [M], **0.83 in 2024 dry [ours]**; pole wins ~40-50 % | arXiv 2507.10966, ours |
| **Tyre strategy** (stops, compounds, undercut, SC timing) | **2024 dry finishers averaged 1.74 stops: 43 % one-stop, 42 % two-stop, 14 % three+ [ours]**; an extra stop must be worth 6-7 s on paper (McLaren) | ours, McLaren [A] |
| **Random events** (SC/VSC, failures, crashes, pit-stop errors, red flags) | ~45 % of races without SC, 41 % one SC; 36 % of SCs on lap 1; red flags ~3/season; pit stops > 5 s a few % | TUM [M], lightsoutblog 2024 [A] |
| **The driver** | 12-36 % of variance; teammate race-pace gap **0.32 s/lap median [ours]** | as above |
| **Weather** | decided all three 2024 wet races (timing of the change, red-flag tyre changes) | Pirelli, race reports [M] |

Most results come from car pace. Track position comes next, and strategy plus random events
create the differences inside that order. Drivers matter mostly through race pace, mistakes,
starts and racecraft; psychology is almost unmeasurable.

---

## 2. Factor by factor: real F1 vs our engine (v0.9.0)

The "Our engine" column describes engine v0.9.0 as it is now. Gap: **✗ missing**, **≈ roughly right**,
**↑ / ↓ too strong / too weak**.

### 2.1 Tyres (the biggest gap)

| Factor | Real F1 | Our engine | Gap |
|---|---|---|---|
| Fresh-tyre pace between compounds | **0.3-0.8 s per step, ~0.5 typical** (Pirelli Imola 2024; 2022 test 0.8-1.0) [M/A]; bigger on long high-energy tracks; the softest can be useless (C4≈C5 Bahrain) | soft −0.09 to −0.23 s, hard −0.19 to +0.21 s vs medium (hard *faster* than medium at Japan, Monaco, China) | **↓ 3-5× too small**: the compounds are nearly interchangeable |
| Degradation, fuel-corrected | soft 0.05-0.15, medium 0.03-0.10, hard 0.01-0.07 s/lap per lap; soft/hard ratio 2-4; per-track multiplier ~0.2 (Monaco) to ~2 (Bahrain) [M/A] | soft 0.05-0.14, medium 0.01-0.12, hard 0.02-0.10 (mean over a typical stint); soft/hard ratio ~1.5 | ≈ level, **↓ compound contrast too small** |
| Degradation measured on 2024 race laps [ours] | slope after removing 0.048 s/lap fuel: H 0.017, M 0.030, S 0.049 | engine 2-3× higher | caveat: depends on the fuel coefficient (0.03 s/kg × 1.7-2 kg/lap ≈ 0.05-0.06 s/lap); *observed* laps include drivers managing to a target pace |
| Cold tyres / out-lap | +0.8-1.5 s on the first lap of a new set (TUM 1.0) [M]; then a fresh-tyre peak: **the first flying lap is 0.24 s faster than laps 3-6 [ours]** | not modelled | **✗** (this drives the undercut) |
| Undercut | median +0.5 to +2.5 s; Bahrain +2.4, Las Vegas +4.2, Melbourne/Qatar negative (766 attempts, f1chronicle) [A] | only from wear differences: small | **↓** |
| Cliff / thermal degradation / graining | real but not published as numbers [C] | flattening wear curve (no cliff) | ✗ (editorial if added) |
| Tyre allocation | 13 dry sets (2H/3M/8S), used vs new sets after qualifying constrain strategy [M] | unlimited new sets | ✗ |
| Track evolution | ~0.01-0.02 s/lap rubbering-in during a race [A] | folded into fuel | ≈ |
| Track temperature | moves degradation; graining when cold, blistering when hot [C] | no | ✗ (no numbers: editorial) |
| **Outcome: stops per finisher** | **1.74; per race 1.06 (Baku, Jeddah) to 2.1 (Bahrain, Spain, Hungary) to 3.0-3.4 (Japan, Qatar) [ours]** | **1.00 everywhere** | **✗: the clearest symptom** |

**Diagnosis.** Our M4 fit (EXP-003) found compound offsets close to zero. The reason is confounding: soft tyres are
mostly run early on heavy fuel or late in short stints, and hard tyres in long managed stints. The EXP-003
report already warned that race-trim data can't pin the offsets down. With soft barely faster than hard
and a small degradation contrast, the "fast but wears" vs "slow but lasts" trade-off disappears.
One stop is always optimal and the compound choice barely matters.

### 2.2 Cars

| Factor | Real F1 | Our engine | Gap |
|---|---|---|---|
| Pace spread | race pace 2.2 s/lap front to back [ours]; the top 4 within 0.3-0.6 % | 1.74 s per race from data (D-021) | ≈ |
| Track-type dependence | car strengths differ by corner type and drag (McLaren medium/high-speed, Red Bull DRS efficiency +0.2-0.3 s at Jeddah) [A/C] | per-race offsets from that race's data | ≈ for replays; ✗ for new seasons |
| In-season development | steps at upgrades (McLaren from Miami, Ferrari Monza) [A] | none (per-race offsets only) | ✗ (matters for M9 seasons) |
| Car-specific tyre wear | slope factor ~1.5-2 between cars (TUM 2019) [M]; Red Bull 2023 [A] | no team wear factor | ✗ |
| DRS | 0.2-1.0 s/lap gain by track, 1.0 s window, from lap 3, off for 2 laps after SC (TUM) [M]; per-car efficiency | no explicit DRS (inside the fitted overtaking threshold) | ✗ (explicit trains, DRS defence) |
| Dirty air | downforce loss 2022 ~5 % at 20 m → 2025 ~20 % (FIA CFD) [M]; **lap time +0.19 s within 0.5 s of the car ahead, ≈0 at 0.5-1.0 s (DRS offsets it) [ours]**; tyres overheat when following [C] | flat 0.15 s within 1.2 s (editorial) | ↓ structure (no DRS offset, no tyre overheating) |
| Reliability | 2024: 2.2 retirements per race all causes; McLaren 0, Williams 12 [ours via results]; failures per team ~0.02-0.08 per race [E] | one failure rate for all cars (EXP-006) | ✗ per team |
| Pit crews | stationary median 2.46 s (Red Bull) to >3.0 s (Sauber/Haas/Williams); 10 % of stops > 3.5-4.9 s; disasters 10-52 s [A] | one standstill distribution for all (log-logistic) | ✗ per team |
| Power unit / ERS | engines frozen; small differences [M] | none | fine (fold into car pace) |

### 2.3 Drivers

| Factor | Real F1 | Our engine | Gap |
|---|---|---|---|
| Race pace vs one-lap pace | separate skills (Hamilton lost qualifying 5-19 but races 8-15; Bottas 20-3 vs 14-10) [M]; teammate race gap 0.32 s/lap median [ours] | `race_pace_skill` and `qualifying_skill` editorial personas; the grid is real | ≈ (personas editorial; can be fitted) |
| Consistency (lap-to-lap noise) | **0.49 s median clean-air residual [ours]**; best COL/VER/BEA 0.37-0.39, worst SAR/BOT 0.68 [ours]; TUM 0.46-0.85 [M] | ~0.08 + 0.6 × (1 − consistency) ≈ **0.14 s** | **↓ ~3× too consistent** (gaps too stable, passes too predictable) |
| Mistakes | **1.35 % of green laps ≥ 2.5 s off trend, ~0.7 per driver per race [ours]**; worst SAR/RIC ~2-2.7 %, best COL/VER/HAM 0.3-0.8 % [ours] | 1.2 % per lap, 0.8-4 s, scaled by persona | ≈ rate; per-driver spread editorial (can be fitted) |
| Crashes / DNFs | accident probability 4.5-10 % per driver per race (TUM) [M]; Pérez/Sargeant crash-prone [A] | fitted field rate (EXP-006), persona-scaled | ≈ |
| Starts | reaction 0.2-0.3 s; **lap 1: 1.29 positions changed on average; front third −0.31, back third +0.86 [ours]**; per-driver start skill SD 0.1-0.25 s (TUM) | launch spread fitted (EXP-006); no per-driver start skill | ≈ field, ✗ driver |
| Lap-1 incidents | 2.7 car-involvements per race, 63 % in T1-T2, mid-pack rows most (Dartmouth) [A] | lap-1 crash multiplier ×37 (fitted), not by grid position | ↓ structure |
| Attacking / defending | 33 passes per race in 2024 (Monaco 4, Bahrain 36, Suzuka 48) [M/A]; a deliberate defence can cost the attacker up to 3-4 s/lap [A] | fitted pass model per track (EXP-006 passes 2-5× too few at test tracks); `overtake_skill` / `defense_skill` personas | ↓ passes; skills editorial |
| Tyre management | slope factor ~0.85-1.15 between drivers [E]; drivers hold back 1-2 s/lap at times [C] | `tire_management` persona scales wear 1.3 → 0.7 | ↑ range probably too wide; editorial |
| Wet skill | real but statistically weak (17 wet races could not identify it) [M] | `wet_skill` persona (D-034) | ≈ (keep editorial, small) |
| Aggression / pressure / psychology | penalties (Magnussen 12 points), commentary only [C] | `risk_appetite`, `composure` personas, editorial | keep editorial, labelled |
| Penalties | ~2.8 time penalties per race (5-10 s), track limits (Austria 2023: 83 lap times deleted) [A] | none (only the AI's tyre-rule DSQ) | ✗ |
| Fatigue / heat | real only in extremes (Qatar 2023) [M/C] | none | leave out |

### 2.4 Race dynamics and race control

| Factor | Real F1 | Our engine | Gap |
|---|---|---|---|
| Safety cars | 2024: 10 of 24 races with SC, ~14 with SC or VSC [A]; P(SC count 0/1/2/3) = 0.46/0.41/0.10/0.03; 36 % on lap 1; SC 2-7 laps (mode 3-4); VSC 1-2 laps; per-track history (Canada 88 %) [M] | SC/VSC only after a retirement (0.24/0.26 per retirement, fitted), SC 4-6 laps | ↓ no debris SCs, ✗ per track |
| Cheap stops under SC/VSC | pit loss ~40-60 % lower [M] | yes (neutralised window) | ≈ |
| Red flags | ~3 per season, free tyre change [M] | none | ✗ (decided Brazil 2024) |
| Pit loss per track | 18.4 (Spa) to 28 s (Imola, Singapore, Qatar), median 22.5 s [A] | fitted per race (D-033) | ≈ |
| Track position / passes | see 2.3 | passes 2-5× too few at test tracks (EXP-004/006) | ↓ |
| Team orders | a few per season [C] | none | ✗ (Stage 2) |
| Blue flags / lapped traffic | 0.1-1.0 s per encounter [E] | yes, editorial | ≈ |

### 2.5 Fuel

| Factor | Real F1 | Our engine | Gap |
|---|---|---|---|
| Fuel effect | 0.025-0.035 s/kg, 100-110 kg, 1.5-2.0 kg/lap → 0.05-0.06 s/lap, ~3 s per race [M] | 0.048 s/lap remaining (fitted) | ≈ |
| Fuel saving / lift and coast | ~0.2 s/lap [C] | inside CONSERVE (editorial) | ≈ editorial |
| Burn under SC/VSC | ~0.5× (TUM) | same as green? | small |

---

## 3. Why EXP-009 failed, in this light

The AI could only change tyres and pace, and in our world both are nearly free choices:
- **The compounds are too similar** (3-5× too small an offset), so tyre choice is nearly free.
- **There is no out-lap / fresh-tyre effect**, so there is no undercut to fight for.
- **There are no debris safety cars**, so few cheap-stop chances.
- **Laps are too consistent and passes too rare**, so track position is locked.

Real teams split between 1, 2 and 3 stops because these factors make the choice matter. A
world that reproduces them gives an AI strategist something real to learn, and gives the rule
planner real decisions to get wrong.

---

## 4. Adaptation plan (proposal, to decide together)

Each item is an engine change: it bumps the engine version, so old saves are refused. Each is
fitted on 2024 where data allows (train/test split as in M4) and measured against real targets.
Editorial pieces stay labelled. Targets come from section 2.

| # | Slice | What changes | Fitted from | Real targets to hit |
|---|---|---|---|---|
| **W1** ✅ D-041 | **Tyres v2** (done; see EXP-010: life + management instead of pace gaps) | Compound offsets identified properly: same-lap comparisons between drivers on different compounds, with fuel at 0.03 s/kg, team/driver effects and priors from Pirelli's per-step gaps. Per-track degradation with a soft/hard contrast. Out-lap cold-tyre loss + fresh-tyre peak. Optional cliff (editorial). Tyre allocation: 13 sets, used sets after qualifying. Car and driver wear factors. | 2024 laps (stint starts, fresh vs used sets, pairs of cars), Pirelli nominations | stops per finisher per race (1.06-3.4, mean 1.74), compound sequences, undercut gains (+0.5-2.5 s), stint lengths |
| **W2** ✅ D-043 | **Race pace realism** (done; EXP-012; development deferred) | Lap noise to the real 0.4-0.7 s per driver (fitted per driver); driver race-pace offsets fitted vs teammate (shrunk); per-team in-season development and track-type offsets (for seasons); track evolution | 2024 clean-air residuals, teammate comparisons | lap-time spread per driver, teammate gaps, race-pace order |
| **W3** ✅ D-044 | **Racecraft** (done; EXP-013; no explicit DRS) | Explicit DRS (1 s window, from lap 3, off after SC); dirty air by gap + tyre overheating when following; attack/defend skills fitted from net passes vs pace delta; duel costs; per-driver start skill; lap-1 incidents by grid zone | 2024 gap traces and position changes | passes per race per track (Monaco 4 … Suzuka 48), lap-1 changes (1.29 positions; front −0.3, back +0.9), grid-finish correlation 0.83 |
| **W4** ✅ D-042 | **Race control and events** (done; EXP-011) | Debris/incident safety cars per track (not only after retirements), lap-1 SC share, real SC/VSC lengths; red flags with free tyre change; per-team pit crew distributions with rare disasters; per-team reliability; 5/10 s penalties (track limits, collisions; aggression-linked, editorial) | 2024 track status, pit timing, results/status | SC races 10/24, SC+VSC ~14/24, red flags ~3/season, stationary-time medians per team, DNF rates |
| **W5** | **Drivers' character, measured where possible** | Personas re-fitted from data where measurable (consistency, mistake rate, start skill, race pace, tyre management, overtaking/defending); PUSH/MANAGE pace modes fitted from stints that were pushed vs managed; aggression ↔ penalties/incidents and wet skill stay editorial | 2024 data | per-driver spreads; every remaining editorial trait labelled |
| **W6** | **Weather extensions** | Track temperature → degradation, graining/blistering (editorial ranges), visibility red flags, SC probability in rain | literature + 3 wet races | small set of wet-race checks (EXP-005 style) |

Then back to **M5** with:
- an observation that includes what now matters (`rl_obs_v2`: compound offsets and wear per
  track, tyre wear in seconds, sets left, DRS/gap state);
- training on randomised tyre parameters, so the AI learns rules rather than tracks;
- a re-run of EXP-008/009. The bar may move: the rule planner must also learn about cold tyres and sets.

**Recommended order: W1 → W4 → W2 → W3 → W5 → W6.** W1 alone should produce 2-stops, real
compound choices and undercuts, which is the biggest lever for both realism and AI learning. W4
adds the random events that make strategy decisions under uncertainty interesting. W2/W3 make
track position and racecraft right. W5/W6 refine characters and weather.

**What stays out or editorial:** psychology and pressure, home race, contract year, fatigue
(no reliable evidence), the cliff shape, and track-temperature sensitivity numbers. Team orders
belong to Stage 2 (M8).

---

## 5. Our own measurement notes

- Data: `data/processed/datasets/laps_2024_v1.parquet`, dry groups (21 races).
- "Green" means track status 1, no pit in/out laps, lap ≥ 2.
- Fuel correction is 0.048 s/lap (the engine's fitted value).
- Residuals are taken against each driver-stint's linear trend.
- Clean air means a gap ahead > 2.5 s and FastF1 `representative` laps.

| Measurement | Method | Result |
|---|---|---|
| Degradation | stint slope after fuel correction, stints ≥ 8 laps, stint lap ≥ 2 | H 0.017, M 0.030, S 0.049 s/lap per lap (median over races); sensitive to the fuel coefficient |
| Consistency | residual SD, clean air, \|res\| < 2 s | median 0.49 s |
| Gap ahead vs lap time | residual by gap bucket | < 0.5 s: +0.19 s; 0.5-1.0 s: −0.03 s; 1-1.5 s: −0.08 s; 2.5-5 s: −0.12 s |
| Mistake proxy | green laps ≥ 2.5 s slower than trend | 1.35 % of laps |
| Fresh-tyre peak | first flying lap after a stop vs laps 3-6 of that stint | −0.24 s (n = 541) |
| Stops per finisher | finishers within one lap of the full distance | 1.74 (43 % / 42 % / 14 %) |
| Lap 1 | grid − position after lap 1 | 1.29 positions changed on average |
| Car vs driver | race-pace medians per team / teammates | team front-to-back 2.16 s; teammate gap 0.32 s median; SD across teams 0.68 vs within 0.23 |
| Grid vs finish | Spearman per dry race | median 0.83 (0.48-0.98) |

---

## 6. Update after W1 (2026-10-15)

- **Same-lap finding:** on race laps the compounds are about equally fast (median gap ~0 per
  race; pooled soft -0.24 s, hard +0.10 s). Pirelli's 0.3-0.8 s per step is one-lap potential.
  In a race the softer tyre's advantage shows up as shorter life and management. W1 therefore
  models **tyre life + management + cliff** (D-041), not bigger pace gaps.
- **EXP-010:** stops per finisher error halved (test 0.56 -> 0.28, correlation 0.75).
  Still open: Las Vegas and Qatar, variety inside a race, same-lap compound pace, retirements,
  tyre allocation.
- **After W4 (EXP-011):** red flags, penalties, VSCs, SCs and retirements at 2024's level;
  finishing order vs real 0.62 -> 0.68. Open: SC clustering, lap-1 pile-ups, red-flag stoppage.
- **After W2 (EXP-012):** lap noise ~0.45 s per lap with the real drivers' consistency ranks,
  teammate gaps of the real size (with race form), passes kept at 2024's level.
- **After W3 (EXP-013):** passes from a principled refit; attack/defend and launch skills add
  a little (car pace dominates); lap-1 incidents concentrated in the midfield.
