# M5 design — the first trained AI: a field of AI strategists

Agreed with the owner on 2026-10-13 (D-037). Where it differs from plan §9/§16 (Turkish), this
document wins; the reasons are in D-037. Nothing here is built yet.

## 1. Goal

The project exists to train AI: a **live AI world** in F1, where drivers, strategists and teams act
with their own character, compete, make mistakes and, over seasons, write their own alternative
history. M5 is the first step that actually trains something. It is judged by two questions:

1. Did an AI really learn to make better race decisions than our rules (honestly measured)?
2. Does the race feel more alive: 20 cars with different, explainable decisions instead of 20
   identical one-stoppers?

M5 stays **Stage 1** (plan §2): each car optimises its own goal; it does not model rivals'
intentions (Stage 2). Other cars are part of the environment.

## 2. Roles and models

| Role | Decides | Model in M5 | Later |
|---|---|---|---|
| Strategist + driver (one decision maker per car, plan §2 "driver-specific strategy policy") | pit / tyre at decision points; pace mode each lap | **one shared RL policy** (PyTorch), conditioned on driver profile, car, track, team objective | Stage 2: attack/defend, team orders; split driver vs strategist |
| Team principal | the team's objective for this race | **rules** from standings and car strength (editorial, labelled) | learned (Stage 2/3) |
| FIA / race director | rules, legal actions, bans | deterministic, versioned rule set | per-season regulation changes |
| Media, contracts, negotiation | — | — | Stage 3: local LLM (Qwen/Llama via Ollama), proposals only, checked by `SocialActionValidator` |

**Why one shared policy for all cars (not 20 networks):** each car's network input includes its
identity, profile and objective, so one network plays 20 characters. It learns from 20x the
experience, works for a new driver at once, and characters can later get small per-driver
adapters (fine-tuned heads) without retraining the core.

**Why not an LLM for race decisions:** RL needs millions of fast, exactly reproducible decisions;
LLMs give a few per second, aren't reproducible enough for forks, and are weak at numeric
trade-offs. LLMs are reserved for Stage 3 language and negotiation.

**Model size and hardware:** MLP actor-critic, ~0.1-1 M parameters (a small GRU/transformer over
recent laps is an option if memory helps; measured, not assumed). PyTorch on the RTX 4070 (CUDA).
The simulation runs on CPU worker processes and is the expected bottleneck (~0.7 s per full race
per core, EXP-006): ~17k races = ~1 M decisions in ~15-20 min on 16 cores.

## 3. Objectives: what "winning" means (owner decision 1)

It depends on the championship and is a **team principal decision**. Each race, the team
principal sets the team's objective for each car from the standings and the car's strength:

- **Championship contender** (competitive car, in the title fight): maximise the championship
  outcome; avoid big risks when leading, take them when behind.
- **Points hunter** (midfield / back-marker): maximise expected points; take risks when a points
  chance appears (safety car, rain, retirements ahead).
- **Result / development** (no points realistic): best finishing position, lowest risk.

The objective is an **input to the policy** (goal-conditioned RL) and defines the reward, so the
same network plays all three. M5 races are single races; the championship context (standings,
races left, title rivals) is **sampled as part of the scenario** during training and taken from
the world's season when the world runs seasons (M9). The M5 team principal is a rule set
(editorial, labelled); a learned principal comes later.

Reward (terminal, per car), with C0 using plain race time:

```text
points objective:        u(points scored)
championship objective:  u(points) + lambda * (points vs title rivals), lambda from the gap and races left
result objective:        u(-finishing position)
u = the driver's risk utility (section 4); invalid/illegal actions are masked, never penalised twice
```

## 4. Character (owner decision 3: characters change)

- **M5: character is fixed per world** (pinned profile revision, D-007). It enters twice:
  1. the **dynamics** (already: wear factor, skill, mistake/crash risk; editorial personas),
  2. the **preferences**: a risk utility `u` per driver (risk-seeking drivers value upside,
     careful drivers avoid bad outcomes). Different drivers then *want* different things.
- **Honesty:** personas and utilities are editorial; the decisions that follow from them are
  learned. Both labels are shown.
- **Later (M9, "offseason learning"):** characters change with experience between seasons. Each
  season gets its own pinned model version and profile revision, with a lineage per driver. M5
  adds the registry fields for this lineage (parent model, season, profile revision) but does
  not learn online inside a live world (that would break reproducibility).

## 5. Regulations, exploits and the FIA (owner decision 5)

- **Bug exploit -> fix the engine, never reward it.** An **exploit audit** compares learned
  behaviour with 2024 statistics and with the MPC baseline; a strategy that relies on an
  engine error or an editorial knob (e.g. an unrealistic PUSH multiplier) is a bug: fix and
  retrain.
- **Real but novel strategy -> allowed.** If it is physically and legally plausible, the AI may
  use it; that is the alternative history.
- **The FIA can ban it.** Regulations are a versioned rule set (like the 2024 two-dry-compounds
  rule). A ban adds a rule; banned actions become illegal through the action mask. A world pins
  its regulation version; changes apply between seasons (like the banned steering idea of
  Hamilton's era). Every ban is logged with its reason.

## 6. Human control (owner decision 6)

The user can make any decision for anyone at any time (already: `StrategyController.USER`, pit
and pace commands, take-over). When the user releases a car, its AI continues from the current
state. User decisions are logged; the AI never learns from them inside the live world (only
through an explicit, versioned retraining later).

## 7. Environment

- **Gymnasium env over the existing `RaceEngine` port.** Decision points: lap end, safety car /
  VSC start, rain start/stop, forecast change (plan §9.3); pace mode per lap.
- **Actions:** pit `[STAY_OUT, PIT_S, PIT_M, PIT_H, PIT_I, PIT_W]` x pace
  `[CONSERVE, BALANCED, PUSH]`, masked by the engine's legal actions and the regulation set.
- **Observation:** observation schema v5 (no future: tested) + driver profile vector + car
  strength + track features + team objective + championship context.
- **Multi-agent:** event-driven; whichever car reaches a decision point asks the shared policy.
  Training against a **pool of frozen earlier policies** for the other cars keeps learning stable
  (plan §2 Stage 2 recommends this for non-stationarity; we use it early).
- **Determinism:** same seed + same model version + same commands = same race (tested).

## 8. Baselines and evaluation (honesty rules)

- **B0** the real 2024 strategy (a reference, not a fair opponent), **B1** today's rule planner,
  **B2** offline enumerator (all legal 0/1/2/3-stop plans), **B3** fork-based Monte Carlo / MPC:
  forks the world, tries each action on identical futures (common random numbers). B3 is both the
  bar to beat and a teacher (imitation warm-start).
- Paired seeds, **held-out tracks** (the 5 dry_test races) and **held-out weather**; bootstrap
  confidence intervals; regret vs MPC; beat-rate vs rules.
- **Profile effects:** swap profiles on the same seed and measure the behaviour change; risk
  Pareto (mean vs worst outcome) per profile; "high risk is not always better" (plan §9.9).
- **Realism of the AI field:** reuse EXP-004/006 metrics: stops per finisher, compound sequences,
  pass causes, vs 2024.
- A failed training is reported with a diagnosis, never as "training finished".

## 9. What you will see (UI)

- **AI thinking panel** per car: top-3 options with their value and outcome spread, and a short
  reason ("rain in ~8 min", "tyre cliff", "SC: cheap stop").
- **Behaviour cards**: measured behaviour per driver (average stop lap, push share, risk taken),
  labelled learned vs editorial; the team principal's objective for this race.
- **Fork and try model**: a new model version runs only in a fork, side by side with the old one on
  the same seed; a world reopens with its pinned model version.
- **User override** and hand-back to the AI (section 6).

## 10. Slices (one commit each)

| # | Slice | Proves |
|---|---|---|
| 1 | Gymnasium env (single- and multi-agent, decision points, masks, regulation set v1), objectives and reward variants, scenario sampler with championship context, env speed benchmark, determinism tests | env correct, fast enough, no future leakage |
| 2 | Baselines B1, B2, fork-based B3 MPC; EXP-008 baseline report | how much better a smart strategy can be in our engine |
| 3 | Training stack (PyTorch + MaskablePPO on CUDA, CPU sim workers), first policy: one learning car vs the rule field (optional MPC warm-start), **policy registry + pinning**; EXP-009 on held-out tracks/seeds | an AI really learned (or an honest diagnosis) |
| 4 | Character and goals: profile + objective conditioning, risk utilities, rule-based team principal; plan experiments E1/E2 + objective tests; EXP-010 | characters and objectives change decisions, for learned reasons |
| 5 | The AI field: shared policy for all 20 cars, frozen-opponent pool; realism vs 2024; exploit audit; FIA ban mechanism; EXP-011 | the world is AI-run, varied and plausible |
| 6 | Weather generalisation, UI (thinking panel, behaviour cards, fork-and-try, pinned versions, override hand-back), M5 done-check | — |

**Cut from plan §16:** DQN (MaskablePPO only), the full C0-C5 curriculum (keep: dry single
track -> many tracks -> weather), the polished outcome-distribution chart (top-3 list first).

## 11. M5 done-check

- The learned policy beats B1 on held-out tracks and seeds (paired, with confidence intervals);
  its gap to B3 is reported.
- Profile and objective effects are measured (swap tests, risk Pareto).
- A 20-car AI field shows strategy variety closer to 2024 than the rule field (stops, compounds,
  pass causes), with the exploit audit clean or its findings fixed.
- Reopening a race uses the same model version; a new version only runs in a fork.
- If any of these fail: the report says so and why.

## 12. Open, decided later

- Network memory (GRU/transformer over laps) only if MLP + lap features fall short.
- Learned team principal, team orders, opponent modelling: Stage 2 (M8).
- Offseason learning and character drift: M9.
- LLM for media/contracts/negotiation: Stage 3, local via Ollama, validator-gated.
