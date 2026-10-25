# Plan: an F1 world where nobody knows the answers (hidden race-day truth, 2000-2025)

Owner's request (2026-10-16): *"in the real world, teams don't know the answers. In the alternate
F1, the AI should not know them either. It should learn through the laps of that race, and after
that race there are completely new conditions because of a new track. Cover 2000 to 2025, all
races, because with the same 24 tracks and one year's statistics it would be learned quickly.
Learning should be slow in general and differ from race to race, within a year and over the
whole period."*

Status: **agreed 2026-10-16** (owner: H0 for all of 2000-2025, then 2010-2025 first). H0-H2
done (D-052..D-057); **H3 done (D-058): gate EXP-027 (numbered after the plan; the table's
EXP-024..027 became EXP-027 onward) +0.019-0.021 score per car-race overall, +0.04 in 2018-2025,
below 0.02 before 2018.** Next: the owner's call (recommended: H4, 2018-2025 first).

---

## 1. Why the AI couldn't learn so far

Measured across EXP-017, EXP-018 and EXP-019 (D-047..D-050):

- **The rule planner knows the truth.** It reads the true tyre life, wear and cliff of this race
  from the setup (the same numbers the simulation runs on) and optimises 0/1/2-stop plans
  exactly. No learner can beat a planner with perfect knowledge of the physics by much.
- **What's left is luck.** With exact counterfactuals on identical luck (slice 4b), a perfect
  hindsight oracle could gain +0.002 (racecraft) to +0.011 (pit timing) per decision, but the part
  predictable from what a driver can see is about 0: it was the rules' already.
- **It isn't the undercut** (EXP-020): our undercut matches 2024 (median +1.7 s vs +1.5 s, 50 %
  vs 46 % of attempts end ahead). The clear gap is that real teams *try* undercuts 3x as often
  (4.6 vs 1.6 per race): they react to what they see, because they don't know the answer.
- **And one season is too narrow:** 24 tracks of 2024 with fitted parameters would be learned
  (memorised) quickly; real strategists face a new mix of track, tyres, cars and rules each race
  and each era.

## 2. What the research says

**Teams don't know degradation before the race; they estimate it and update it.**
- Pirelli starts from compound performance differences from pre-event simulations combined with
  historical degradation data; teams compare those projections with their own data
  ([F1technical, F1MATHS](https://www.f1technical.net/news/28001)).
- Predictions are regularly off: e.g. Pirelli expected Hungary to open up into a two-stop after
  Friday showed higher-than-expected degradation
  ([F1technical](https://www.f1technical.net/news/29017),
  [strategy preview](https://www.f1technical.net/news/29030)); softer-than-usual compound
  choices have produced higher-than-expected degradation in Barcelona
  ([Autosport](https://www.autosport.com/f1/news/at-least-a-two-stop-george-russell-and-lewis-hamilton-expect-barcelona-gp-to-be-hard-on-tyres/10830018/)).
  Practice data is thin: a single long run on a compound leaves its curve very uncertain
  ([R-bloggers FP2 analysis](https://www.r-bloggers.com/2022/04/australian-gp-fp2/)).
- In-race, degradation can be estimated as a **hidden state from lap times**, with fuel burn
  modelled and pit stops resetting the state: a Bayesian state-space model on public FastF1 data
  gives probabilistic, interpretable estimates and beats an ARIMA baseline
  ([arXiv 2512.00640](https://arxiv.org/abs/2512.00640v1)).

**Learning race strategy with AI: memory matters because the race is partially observed.**
- TUM's Virtual Strategy Engineer: neural networks trained on 2014-2019 timing data decide pit
  stops and compounds inside a race simulation
  ([Heilmeier et al. 2020](https://portal.fis.tum.de/en/publications/virtual-strategy-engineer-using-artificial-neural-networks-for-ma/)).
- Mercedes-AMG with Imperial/King's: RSRL uses a **recurrent** Q-network (DRQN) *because the race
  environment is partially observable*; it is reported to beat fixed strategies by 8.6 s on
  average in simulation
  ([review](https://www.themoonlight.io/en/review/explainable-reinforcement-learning-for-formula-one-race-strategy),
  [KCL news](https://www.kcl.ac.uk/news/ai-driven-race-strategy-could-give-formula-one-teams-competitive-advantage)).
- Multi-agent RL agents adapting pit timing, tyres and energy to opponents
  ([Fieni et al. 2026](https://arxiv.org/pdf/2602.23056)).

The common thread: the edge comes from **acting well under uncertainty** (inferring the race as it
unfolds), not from out-optimising a known model.

## 3. Data for 2000-2025 (verified against the APIs)

| Source | Coverage | What we get |
|---|---|---|
| Jolpica-F1 (Ergast successor) | results 1950-2025; **lap times from 1996** (e.g. 2000 Australia: 863 lap timings); **pit stops from 2011** (2005 returns 0, 2011 returns 45) | every car's lap time and position per lap, grid, results, status, qualifying, pit stop laps and durations (2011+) |
| FastF1 (F1 live timing) | **2018 onwards** ([docs](https://docs.fastf1.dev/data_reference/index.html)) | compounds and tyre age per stint, track status (SC/VSC/red), weather, race control, positions; what our 2024 pipeline already uses |

Sources: [lap times 1996+](https://www.mintlify.com/TracingInsights-Archive/Stats/data-collection/lap-times),
[pit stops 2011+](https://www.mintlify.com/TracingInsights-Archive/Stats/data-collection/pitstops),
[Jolpica analysis](https://cran.nics.utk.edu/cran/web/packages/f1dataR/vignettes/jolpica-data-analysis.html).
Rate limits: about 4 requests/s burst and 500/hour sustained, so ~500 races x ~15 requests is
about 16 hours of background fetching (cached, resumable).

**Gaps and how to fill them (labelled):**

| Missing | Years | Plan |
|---|---|---|
| pit stops | 2000-2010 | infer from lap times (in-lap + out-lap ~20+ s slower, the position drop): `estimated_from_data` |
| tyre compounds per stint | 2000-2017 | unknown; model "tyre age" degradation without compound labels (2007-2017: two compounds per race and the mandatory-use rule are known from regulations); `estimated_from_data` / `editorial_assumption` |
| SC / VSC periods | 2000-2017 | infer from field-wide slow laps (every car +20-40 %), cross-check with retirements: `estimated_from_data` |
| weather | 2000-2017 | infer wet races from lap times and compounds where possible; otherwise mark the race "unknown weather" and keep it out of dry fits |

## 4. Eras: the rules change, so strategy changes

To be verified against the FIA sporting regulations in H1 (the items below are well documented;
[refuelling ban 2010](https://www.formulaonehistory.com/seasons/2010-f1-world-championship/),
[tyre suppliers 1999-2011](https://the-race.com/formula-1/what-bridgestone-f1-past-tells-us-about-its-bid-to-depose-pirelli/)):

| Years | What changes the strategy game |
|---|---|
| 2000 | refuelling; Bridgestone the de facto sole supplier |
| 2001-2006 | refuelling; **tyre war** (Bridgestone vs Michelin: car-specific tyre behaviour) |
| 2005 | **no tyre changes**: one set for the whole race; stops are fuel-only |
| 2007-2010 | Bridgestone control tyre; two compounds per race, both must be used; 2010 **no refuelling** |
| 2011-2013 | Pirelli high degradation by design; DRS; pit stop data begins |
| 2014-2021 | hybrid era; VSC (2015); wider cars (2017); 2021 sprints |
| 2022-2025 | ground-effect cars; current tyre rules; FastF1 data like ours |

Points systems also change (2000-2002, 2003-2009, 2010-, fastest-lap point 2019-2024): the
objectives (D-050 title probability) must use each season's system.

**Refuelling (2000-2009) is a different strategy problem:** a stop is fuel + tyres; the first
stint length sets the fuel load (lighter = faster but stop earlier). The engine needs fuel load
as a strategic variable for that era (H1).

## 5. Design: truth, knowledge and learning on three timescales

**Two layers in every race:**

- **Truth (hidden):** this race's real tyre life, wear and cliff per compound, the track's
  evolution through the race, and each car's true race pace. Drawn per race (keyed seed) around
  what history predicts, with the size of the surprise **measured from data**: how far each
  track's fitted degradation in year Y is from what years < Y at that track (and the era) would
  have predicted. Nobody in the world reads it, not the rules and not the AI.
- **Knowledge (what a team has):**
  - before the race: the prior (previous years at this track, the era's tyre supplier,
    Pirelli-style compound gaps where data exists), optionally a noisy practice long-run
    estimate;
  - during the race: every car's lap times, stint ages and compounds (visible on the timing
    screens), its own lap times; the weather forecast as today.

**Who decides with what:**

| Strategist | Knows | Role |
|---|---|---|
| rule teams (B1 v2) | prior + a simple in-race estimate (fit of their own lap times), replan each lap | the realistic average team; the field the AI races against |
| oracle rules | the truth | reference only (how much information is worth); never in a world |
| AI (`rl_obs_v4`) | prior + lap-time history (and a Bayesian degradation estimate as a team tool, labelled) | learns to infer and to decide under uncertainty: recurrent policy (memory, as RSRL) or belief features |

**Three learning timescales (slow by design):**

1. **Within a race (fast):** the AI's belief about this race's tyres sharpens lap by lap; a new
   race starts from the prior again: "completely new conditions at a new track".
2. **Within a season (medium):** team memory: what a team learned at earlier races of the season
   (tyre supplier behaviour, its car) feeds the next prior. Labelled; never the future.
3. **Across seasons (slow):** the strategy policy itself is trained offline across eras and,
   in a live world, retrained between seasons (M9 offseason learning, pinned model per season).
   Regulation changes (refuelling ban, Pirelli, 2022 cars) force it to adapt, like real teams.

**Generalisation, not memorisation:** training samples (track, year, hidden truth) from ~500
races; evaluation on **held-out seasons** from each era (e.g. 2003, 2008, 2013, 2019, 2023) and
**held-out tracks**; never the same track-year in training and test.

## 6. Plan (slices, one commit each; gates decide whether to continue)

| Slice | What | Experiment / gate | Estimate |
|---|---|---|---|
| **H0 Data 2000-2025** | Jolpica adapter for laps, pit stops, qualifying, status; FastF1 2018-2025; inferred pit stops / SC laps pre-2011/2018; quality reports and dataset manifests per season | EXP-021: coverage and inference checks (inferred pit stops vs real on 2011-2025) | 1 session of code + ~1 day of background fetching |
| **H1 Eras and regulations** | `RegulationSet` per season (points, refuelling, tyre-change ban, compound rule, DRS, VSC); engine: fuel load as a strategy variable (refuelling era) | EXP-022: era sanity (stops per race, stint lengths vs real by era) | 1-2 sessions; engine bump |
| **H2 Calibration across eras** | per-race car pace, lap noise, stint degradation (compound-aware 2018+, age-only before), pit loss, overtakes per era (no DRS before 2011), incidents/SC rates | EXP-023: fit on training seasons, test on held-out seasons per era | 2 sessions + fits |
| **H3 Hidden race-day truth** | priors from history (no future), truth distribution with data-measured surprise size, keyed per race | **gate EXP-024: value of information** = oracle rules vs rule teams with the prior; if < ~0.02 score per race, information doesn't matter enough and we stop here to rethink | 1 session |
| **H4 Teams that learn in-race** | `rl_obs_v4` without hidden truth (lap-time history, rivals' timing, prior); in-race estimator for rule teams; Bayesian degradation filter as a labelled team tool | EXP-025: estimator accuracy vs truth lap by lap; undercut attempts per race vs 2024 (4.6) | 1-2 sessions |
| **H5 The AI learns under uncertainty** | recurrent PPO or belief+MLP; counterfactual targets on identical luck (D-049) where useful; curriculum by era; held-out seasons | **EXP-026: beats the rule teams held out (paired CI > 0), gap to oracle reported**; behaviour differs by goal/character (careful/daring, leader/chaser) | several sessions; training hours |
| **H6 Living seasons** | seasons in sequence, team memory, offseason retraining (M9 link) | EXP-027: learning curve across seasons (slow, era shocks visible) | later |

## 7. Honesty labels

- Real: lap times, positions, results, pit stops (2011+), compounds/status/weather (2018+).
- Estimated from data: inferred pit stops and SC laps, per-race fits, surprise sizes.
- Editorial: compound behaviour where no compound data exists (2000-2017), personas for
  2000-2025 drivers until measured, the practice-estimate noise.
- Learned: only the AI's decisions, judged against rule teams with the same knowledge.

## 8. Risks

- Pre-2018 data has no compounds, status or weather: the older the era, the more is inferred.
- Refuelling-era strategy needs a real engine extension (fuel load), not a parameter tweak.
- ~500 races x 20 drivers: personas and car strengths for every season (data-backed where W5
  methods work; editorial otherwise).
- Fetch time and rate limits (resumable, cached).
- H3's gate could fail (information worth too little): then the honest conclusion is that strategy
  in our world is decided by luck and car pace, and the next lever is elsewhere.

## 9. Decisions for the owner

1. Order: H0 -> H1 -> H2 -> H3 (gate) -> H4 -> H5, or skip the refuelling era at first (start
   with 2010-2025, add 2000-2009 in H1b)? Recommendation: **data for all of 2000-2025 now (H0),
   then build 2010-2025 first** (no refuelling, compounds known 2018+), then add the refuelling
   era, so the hidden-information design is proven before the biggest engine change.
2. Keep the AI's tools honest: should the AI get the Bayesian degradation estimate as a feature
   (like a team's software) or must it infer purely from raw lap times (memory)? Recommendation:
   both, compared (EXP-026).
