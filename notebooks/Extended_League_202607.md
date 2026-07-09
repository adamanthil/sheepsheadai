# Extended League Run — pre-registration (July 2026)

Pre-registered BEFORE the run starts, per repo practice. Orchestrated by
`analysis/run_extended_league.py`; decision math in `analysis/league_stopping.py`
(unit-tested on synthetic curves in `tests/test_league_stopping.py`); endpoint
machinery in `analysis/league_progress_eval.py`.

## Design

One long league run (`train_league_ppo.py`, one subprocess per generation,
resume-chained on the absolute-episode boundaries) folding in the July-2026
learnings:

- **Oracle critic** (`--critic-mode oracle`): privileged CTDE GAE baseline —
  attacks the documented PPO-can't-learn-small-early-gaps variance mechanism.
- **League seeded** from the ablation winner's selfplay checkpoint ladder
  (`--seed-checkpoints`), resumed from its final checkpoint. Arch parameterized;
  filled in when the ablation verdict lands.
- **Generation 1 KL-anchored** (bidding-head anchor to the ORIGINAL resume
  checkpoint) to guard the shaped→terminal reward transition — the ExIt
  PASS-collapse mechanism. Generations ≥ 2 unanchored.
- **Generation length 1M episodes**; exploiter phase per trainer defaults
  (50k episodes, 3000-deal gate).

## Anchor-coefficient calibration (pre-registered criterion)

~15k-episode probes at coefficients {0.3, 1.0, 3.0} (in-process
`run_main_phase`, throwaway league, no exploiter). Baseline =
`greedy_health_probe(resume, 400 games, seed 20260709)`.

Choose the **smallest** coefficient with ALL of:
1. mean `anchor_kl` over the final 3 PPO updates ≤ 0.05 nats AND max ≤ 0.10;
2. zero greedy-health gate violations (PFSPHyperparams thresholds);
3. final greedy pick rate within ±10 percentage points of baseline.

None qualifies → largest candidate + `needs_review` halt (operator must pass
`--allow-calibration-fallback`). Smallest-sufficient because the anchor caps
bidding improvement (trainer docstring).

## Stopping rule (pre-registered constants)

Tightens the rule at Architecture_Ablation_202607.md §"Continuing league
generations": the repro league learned ~0.015 score/hand per 1M episodes, so
the 1000-deal / 0.07-MDE version would false-stop on real-but-slow learning.

**Endpoint** (per generation g): interleaved 3-checkpoint composite —
boundary−100k / −50k / boundary, deal i → checkpoint (i mod 3)+1 — on the
frozen PANEL-A field, **4000 deals** (2000/mode, called first), seed **42**,
field seed 20260619. Deal-noise SE of a single 4000-deal panel (MDE ≈ 0.035),
snapshot noise ÷√3, per-deal CRN pairing preserved across generations.
Generation 0 = the resume checkpoint (×3).

**Flat verdict**: generation g is flat when NONE of:
- **A** gain vs previous best: mean(panel_g − panel_best) ≥ **0.035** with
  bootstrap 95% lo > 0;
- **B** head-to-head vs g−1 (`paired_edge`, 2000 deals, seed **20260708**,
  previous gen's field): edge ≥ **0.05** and ≥ 2·se;
- **C** paired 3-endpoint slope mean((panel_g − panel_{g−2})/2) ≥ **0.0175**
  with lo > 0. A statistically positive slope below 0.0175 stops by design but
  is flagged `slope_small_but_significant` in the report.

**Stop candidate**: two consecutive flat generations (onehot-ff false-plateau
lesson), floor **min_generations = 4**, cap **max_generations = 12** (cap
forces confirmation regardless).

**Confirmation** (fresh deals, seed **20260706**, same design): heroes = gen g,
gen g−2, and the argmax-panel generation (deploy candidate). Contradiction —
paired fresh-deal gain of g over g−2 ≥ 0.035 with lo > 0 — resets the flat
streak and resumes training; otherwise STOP. The confirmation estimate is the
deploy candidate's reported number; the seed-42 series stays the longitudinal
curve. Guards seed-42 stopping-rule selection bias.

**Exploiter gates are recorded but are NOT stop-rule inputs** (exploitability
robustness ⊥ strength growth). Stopping while the last gate passed is reported
as "stopped while still exploitable".

## Telemetry

- `runs/<run>/orchestrator/`: `state.json` (crash-resume), `generations.csv`,
  `report.md`, `generations_curve.png`, per-gen `panel_gen<g>.npz` /
  `h2h_gen<g>.json`, calibration probe CSVs.
- Trump-lead leak trend (trick-0/1 defender trump-lead rate + trump prob mass,
  **only at nodes with a legal non-trump lead**; forced all-trump leads tallied
  separately) is harvested passively from the panel games via
  `rigorous_eval.DecisionProbe` — zero extra deals, snapshot/restore around the
  probs call (the 2026-06-10 double-encode lesson). References: 30M baseline
  t0 ≈ 4.8%, scripted = 0.
- Trainer CSVs accumulate continuously across per-generation invocations; the
  300-deal anchored_eval stays motivation-only (never slope claims).

## Standing caveats

- Trainer resets its OpenSkill training rating each invocation (re-converges
  before the first 50k snapshot inherits it).
- Health halts (orchestrator-enforced, trainer only warns): ≥3 consecutive
  greedy-gate violations of the same gate, or leaster rate > 0.30 and rising
  > +0.10 within a generation → `needs_review`.
