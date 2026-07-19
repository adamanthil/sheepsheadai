# Extended League Run — pre-registration (July 2026)

Pre-registered BEFORE the run starts, per repo practice. Orchestrated by
`sheepshead/training/run_extended_league.py`; decision math in `sheepshead/training/league_stopping.py`
(moved from `sheepshead/analysis/` 2026-07-18 — path amendment only, rule unchanged;
bootstrap primitives now shared via `sheepshead/analysis/bootstrap.py`; unit-tested
on synthetic curves in `tests/test_league_stopping.py`); endpoint
machinery in `sheepshead/analysis/league_progress_eval.py`.

## Design

One long league run (`sheepshead/training/train_league_ppo.py`, one subprocess per generation,
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
2. zero greedy-health gate violations (PFSPHyperparams thresholds; the ALONE
   gate is baseline-relative, see amendment below);
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

## Amendment 2026-07-19 (mid-run, pre-gen-3-verdict): criterion B instrument

Measurement-power fix; thresholds unchanged. Motivation, documented
before any gen-3 evidence exists: the pre-registered h2h instrument
(`training_utils.paired_edge`, 2000 deals) realized se ≈ 0.055 in the
perceiver-shared-v2 continuation — criterion B (edge ≥ 0.05 AND ≥ 2·se)
therefore cannot fire below +0.11, making B inert at its intended
threshold (h2h_min 0.05 implicitly assumed se ≈ 0.025). Observed gen-2
edge +0.104 failed B by 0.007 despite being a decisive gain on the
higher-powered instrument.

1. **Instrument replaced** by `league_progress_eval.h2h_duplicate`: the
   candidate seated in all 5 seats per CRN deal vs a field of the
   previous checkpoint (rigorous_eval gauntlet, one-member panel),
   2000 deals per mode, seed 42, deterministic play; edge = both-modes
   mean (anchor's self-field score is 0 by symmetry). Realized se
   ≈ 0.015 — B now fires at its designed threshold. `--h2h-deals` is
   per-mode.
2. **Gens 1–2 values adopted from the already-collected 2026-07-19
   hypothesis battery** (identical pipeline and seed): gen 1
   −0.086 ± 0.013, gen 2 +0.081 ± 0.015. The superseded paired_edge
   measurements (−0.089 ± 0.050 / +0.104 ± 0.055 — consistent point
   estimates) are preserved as `h2h_gen{1,2}.paired_edge.json.bak`.
   Consequence: gen 2's verdict becomes IMPROVING via B; flat streak
   at gen-3 entry is 0, not 2.
3. Orchestrator restarted to load the change (crash-resume path;
   gen-3 trainer resumed from its latest 50k checkpoint).

## Amendment 2026-07-18 (pre-launch): portability + calibration retirement

Recorded before the real run starts; the stopping rule is unchanged.

1. **Calibration phase removed.** Stage 1 of the architecture ablation ran
   its gen-1 anchor at coefficient 1.0 and validated it directly (gen 1
   trained through the shaped→terminal transition without collapse), so the
   probe-based calibration machinery (§"Anchor-coefficient calibration"
   above, `pick_anchor_coeff`) is retired unexecuted. `--anchor-coeff` is
   now a plain flag, **default 1.0**, 0 disables. The calibration criterion
   section above is retained for the record but is no longer executable.
2. **Arch derived from the resume checkpoint** (`--arch` flag removed);
   reproduction runs cannot mismatch it.
3. **`--panel` flag added.** The frozen PANEL-A remains the default;
   reproductions without the research-run checkpoints supply their own
   fixed >=4-checkpoint panel. Panel scores are field-relative, so the
   stopping rule (within-run comparisons only) is unaffected; longitudinal
   comparability to PANEL-A numbers is only claimed for PANEL-A runs.
4. **Endpoint deal count 4000 → 3996.** The composite design interleaves
   deals over 2 modes × 3 checkpoints and requires divisibility by 6;
   4000 was never launchable (the loader rejects it) and no recorded
   number used it. MDE ≈ 0.035 is unchanged.
5. **Health halts narrowed to the leaster trend; verdicts one-shot.**
   The ≥3-consecutive greedy-gate streaks (pick/ALONE/trump-lead/
   play-spread) are demoted to recorded warnings: they are 200-game
   diagnostic probes that the endpoint instruments routinely contradict
   (stage-1 gen 2 trips the trump-lead streak at 1.6–1.7M on ~80-lead
   denominators while its 2M trump-lead probe is clean), and the
   PASS-collapse attractor is now guarded in-loop by the leaster
   watchdog. Only the leaster-trend check still halts (`needs_review`).
   Each generation's health verdict is recorded once in state.json and
   never re-litigated, so relaunching after a halt continues past it
   (previously the replay loop re-tripped the same historical streak on
   every launch, and the only escape — `--ignore-health-halt` — also
   disabled checking for future generations; the flag now merely
   suppresses the halt while still recording the verdict).

## Amendment 2026-07-09 (pre-launch): baseline-relative ALONE gate

Recorded before the real run starts. The smoke run showed a selfplay-lineage
resume legitimately sits at ~18-27% greedy ALONE (weak defender-field
collaboration — the thing league training itself repairs), which tripped both
the calibration criterion and the per-generation halt against the absolute 15%
gate, forcing manual overrides. Changes:

1. `PFSPHyperparams.greedy_gate_max_alone` **15 → 20** (20% of partner
   decisions can still be clean play; trainer warning included).
2. The orchestrator's ALONE checks (calibration criterion 2 and the
   per-generation halt) use an **effective limit =
   max(20, resume-baseline alone rate + 5 points)**, with the baseline
   measured once by `greedy_health_probe(resume, seed 20260709)`. A high-alone
   warm start no longer trips the halt; a *regression* past baseline+5 still
   does. Other gates (pick, trump-lead, play-spread) stay absolute.
