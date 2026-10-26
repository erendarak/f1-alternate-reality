# Plan: an AI-training project first (A1-A5)

Owner's direction (2026-10-25): *"the most important thing in this project should be the AI.
I will train an AI agent so that it reflects the importance of being well trained."* The vision:
trained AIs run 10 teams and 20 drivers through a whole F1 season, and **how well an AI was trained
shows in the championship**.

Status: **agreed 2026-10-25** (D-060). **A1 done (D-061): headroom gate EXP-030 +0.007 to
+0.010 score per car-race overall, +0.010 to +0.013 in 2018-2025: at the 0.01 bar.** **A1b done (D-062, EXP-031): a realistic field (per-team practice + private data,
real undercut rate) closes the information prize (n.s.); pit timing is worth ~0.03 in
hindsight, learnable share unknown.** **A2 pre-gate FAILED (D-063, EXP-032): the timing value is not predictable;
proposal: train the AI as the whole strategist from scratch, the rule teams as the expert.** Replaces the remaining slices of
`PLAN_HIDDEN_INFO_ERAS.md`: H4 becomes A1, H5 becomes A2-A3, H6 becomes A5.

---

## 1. Why now, and not before

Until H3 the AI had nothing to learn: the rule planner read the race's true tyre model, so the
best decision was already known (M5 slices 3-4b, per-decision value ~0.001). The world work since
(W1-W6, H0-H3b) made the problem *learnable*:

- every race 2000-2025 runs on its own season's fits (H0-H2), with its era's rules (H1, H1b);
- each race has a **hidden truth** (tyre life and wear) and teams only know a **prior** from
  history (H3, D-058); track evolution varies race to race and isn't predictable (H3b, D-059);
- **knowing the truth is worth +0.043 score per car-race in 2018-2025** (EXP-027): about +1 point
  per car per race, **~45 constructor points per team over a 22-race season**, roughly one place
  in the constructors' table. That is the prize a well-trained AI competes for.

## 2. What makes this an AI-training project

The claim "better training wins championships" needs four things, each with a number:

1. **A task where decisions matter**: the hidden race-day truth (done, EXP-027).
2. **Fair baselines**: rule teams with the *same* knowledge (prior + an in-race estimator, a
   reasonable average team's tool, neither handicapped nor perfect) and an **oracle** (knows the
   truth) as the ceiling.
3. **Learning curves on unseen races**: more and better training, more points on held-out
   seasons and tracks (never the same track-year in training and test).
4. **A visible payoff**: a season in which better-trained teams finish higher, with the reasons
   shown (decisions, what the agent believed, what was true).

## 3. Rules for every slice

- Each slice ends with an **AI number**: a training run, a learning curve, a held-out paired
  comparison (CI), or a gate that decides whether to continue.
- World realism only when an AI experiment needs it; the rest goes to KNOWN_LIMITATIONS.
- Learned vs rule vs editorial stays labelled; negative results are reported with a diagnosis.
- One 14-worker torch training job at a time (paging file).

## 4. Slices

| Slice | What | AI result it must show |
|---|---|---|
| **A1 Environment under uncertainty** (was H4) | hidden per-race track trend (D-059); an **in-race estimator** (the shared timing screen: every car's lap times, so the wear and trend estimates are Bayesian, pure and in the world state); rule teams **re-plan on the estimate**; `rl_obs_v4` without the truth (estimate + uncertainty + lap-time history) | **Gate EXP-030 (headroom)**: oracle vs estimator rule teams. Under ~0.01 score per car-race: little left for an AI; then rethink before training. Also EXP-029: estimator error vs truth lap by lap; stops vs real |
| **A2 A learner that learns** | policy with memory (recurrent PPO) or belief features + MLP, trained on 2018-2025 with drawn truths; counterfactual targets on identical luck where useful (D-049) | beats estimator rule teams on **held-out** races (paired CI > 0); gap to the oracle reported |
| **A3 Training matters** | the same agent at 5-6 training budgets (and one era vs all eras, with vs without memory) | **budget -> held-out score -> season points**: the project's core plot |
| **A4 The AI league season** | 10 teams, each with its own trained agent (different budgets, methods, characters), a full season with hidden truths, standings and replays in the app | better-trained teams finish higher; a rating per team; explainable decisions |
| **A5 Agents that adapt** (was H6) | population / self-play training, retraining between seasons, team memory within a season | strategies change when rivals change; a learning curve across seasons |

Then M6/M7 of the master plan (alternate seasons, the living world) run on these agents.

## 4b. Revised after A1b and the A2 pre-gate (2026-10-26, D-063): the AI learns the whole job

EXP-031 and EXP-032 showed that realistic rule teams (practice, private data, in-race learning,
the real undercut rate) leave no measurable edge for an AI: perfect tyre knowledge is worth ~0
over them and pit timing isn't predictable from what a car sees. So the AI's task changes from
"beat a strong strategist" to **"become a strategist"**: it starts from nothing (no rule planner,
no cloning of the rules, no planner advice in its inputs) and must learn everything the rule
teams encode. The rule teams are the **expert reference**; training budget is the variable.

| Slice | What | AI result it must show |
|---|---|---|
| **A2 The strategist from scratch** | (a) training scenarios in the A1b world: 2018-2025 dry_train races, hidden truths **drawn** per episode, per-team knowledge, rival rules; several AI cars per race among rule teams; (b) `rl_obs_v4`: what the pit wall sees, no truth and no planner advice: race and car state, the team's own tyre estimate and its uncertainty (the team's software, labelled), the prior's tyre lives, own recent lap residuals, neighbours' gaps / tyres / stops, rejoin traffic, SC/VSC, weather; (c) 18 actions (pit x pace; racecraft NORMAL); (d) MaskablePPO from a random network, terminal POINTS reward paired with the expert's score for the same car (control variate), an editorial "stay out" exploration prior; checkpoints along the way | a **pilot** (~100k decisions) shows the reward rising from the untrained level; then the full run |
| **A3 Training matters** (the core plot) | checkpoints of the full run (0 / 25k / 50k / 100k / 200k / 400k / 800k / 1.6M decisions, 2-3 training seeds), each evaluated on **held-out** races (dry_test 2018-2025), paired with the expert on identical luck | **score and points per season vs training budget**, with CIs: how far untrained is below the expert, how fast training closes the gap, where it plateaus (reported as it is: below, at or above the expert) |
| **A4 The AI league season** | 10 teams, each with a strategist of a different budget (plus the expert as one team), a full season with hidden drawn truths, standings and replays in the app | **better-trained teams finish higher** (rank correlation budget vs points, CIs over seasons); decisions shown with what the team believed and what was true |
| **A5 Agents that adapt** | self-play among AI teams, retraining between seasons | as before |

Honesty rules for this plan: the expert is never weakened to make the AI look good; "reaches the
expert" is a valid result and "beats the expert" is only claimed when measured; what the AI's
inputs contain is listed in its schema (the team estimate is a labelled team tool, not the truth).

## 5. Risks

- **The estimator may close the gap** (EXP-030): if rule teams with an in-race estimate get near
  the oracle, the AI's room is small again. Answer: keep the rule estimator a realistic average
  team (documented choices, not tuned down), and look for value where rules are weak (traffic,
  reacting to rivals, the first stop before estimates settle).
- **Before 2018 the information is worth little** (EXP-027: compounds aren't in the data): train
  and judge on 2018-2025 first, older eras as a curriculum later.
- **Season-scale effects are noisy**: a 22-race season has sd ~10.9 points per race between title
  rivals (D-050); A3/A4 need many simulated seasons per budget.
