# Training guide

The operator's guide to the training program: what to run, what lands on
disk, and what decides each phase. The design, thresholds and reasoning live
in the notebooks listed under [Methodology](#methodology); this page does not
repeat them.

Timings are from a 10-core M1 Max with 8 game workers. Every artifact of a
run lives under `runs/<run-name>/` (gitignored).

## Setup

```bash
uv sync --extra dev                                   # torch, the package, test tools
uv run pytest sheepshead/tests -m "not slow"          # ~15 s sanity check
uv run python -m sheepshead.devtools.check_goldens    # bit-exact vs upstream
uv run python -m sheepshead.eval.fetch_models         # reference checkpoints -> models/
uv run training-program --smoke --run-name _smoke     # every phase on a toy config, ~5 min
```

The smoke run must end with `PROGRAM FINISHED`. It proves the plumbing, not
the learning.

Reference checkpoints are optional: a comparison whose file is missing is
skipped with a log line. To compare a fresh run against the 30M production
model, point `final.references` at `models/pfsp-swish-30m.pt`.

## The orchestrator

`uv run training-program` runs the whole program from one configuration tree
(`sheepshead/training/program_config.py`). The defaults are the
pre-registered run.

```bash
uv run training-program --run-name rc_202609             # the pre-registered run
uv run training-program --config runs/x/config.json      # a modified program
uv run training-program --dry-run --run-name rc_202609   # print the config and first command
```

`--config` takes JSON shaped like `ProgramConfig`; omitted fields keep their
defaults.

### Run record

Written under `runs/<run-name>/program/`:

| File | Contents |
|---|---|
| `config.json` | the configuration actually run |
| `state.json` | resume state: every recorded number of every phase |
| `program.log` | one timestamped line per event |
| `<stage>.log` | stdout of each stage (`bootstrap.log`, `oracle.log`, `league_gen<g>.log`, `pi_iter<k>.log`, `final.log`) |
| `generations.csv` | one row per league generation |
| `report.md` | summary tables, regenerated at every phase boundary |

Markers in `program.log`: `▶` stage start (command beneath), `✔` completion,
`★` decision, `↷` stage skipped on resume, `✖` needs review.

### Resuming and exit codes

- Re-run the same command to resume. A stage whose output exists is skipped,
  and trainers continue from their last checkpoint.
- Exit code 0 means `PROGRAM FINISHED`.
- Exit code 2 means a gate failed or a stage crashed. `program.log` has a
  `NEEDS REVIEW: ...` line. Fix the cause and re-run the same command.

## Phases

| Phase | What it does | Time | Output | Decided by |
|---|---|---|---|---|
| 0 Bootstrap | shaped-reward self-play, 400k episodes | 6–8 h | `bootstrap/final.pt` | last greedy probe's leaster rate below 50% |
| 1 Oracle | pretrains the full-information critic | ~3 h | `oracle/oracle_init.pt` | none |
| 2 League | terminal-reward PPO against a snapshot population, 1M episodes per generation | ~2 days per generation, 3–10 generations | `league/checkpoints/checkpoint_<g>000000.pt` | handoff rule, aux-head readiness, convention bounds |
| 3 Policy iteration | search corpus, distillation, certification, bidding fine-tune | ~3 days per iteration, up to 5 | `pi/iter<k>/bidding/final.pt` | adoption cert per iteration; stops when play-only gain stalls |
| 4 Final | head-to-head reads against references, convention battery, exploitability audit | | `final/release.pt` | recorded for operator review |

The commands below are the ones the orchestrator issues, for a run named
`rc_202609`. They must be kept in sync with `run_training_program.py` by
hand; `--dry-run` and `program.log` show the real ones.

### Phase 0: bootstrap

```bash
uv run train-ppo --phase bootstrap --arch perceiver-recall \
    --run-name rc_202609/bootstrap --until 400000 \
    --save-interval 50000 --greedy-eval-interval 50000 --greedy-eval-games 200 \
    --num-workers 8 --seed 42
```

Under `bootstrap/checkpoints/`:

- `checkpoint_<episode>.pt`
- `training_progress.csv`: one row per PPO update
- `greedy_health.csv`: one row per greedy probe

### Phase 1: oracle pretraining

```bash
uv run python -m sheepshead.training.pretrain_oracle generate \
    --ckpt runs/rc_202609/bootstrap/final.pt --episodes 40000 --workers 8 \
    --gamma 1.0 --seed 42 --out runs/rc_202609/oracle/dataset.pt
uv run python -m sheepshead.training.pretrain_oracle pretrain \
    --dataset runs/rc_202609/oracle/dataset.pt --max-epochs 25 --patience 3 \
    --seed 42 --out runs/rc_202609/oracle/oracle_init.pt
```

`oracle.log` records per-epoch validation MSE and per-stratum explained
variance.

### Phase 2: league

Generation `g` trains to the absolute episode `g × 1,000,000`.

```bash
uv run train-ppo --phase league --resume <previous boundary checkpoint> \
    --run-name rc_202609/league --league-dir runs/rc_202609/league/league \
    --until 1000000 --save-interval 50000 --snapshot-interval 50000 \
    --greedy-eval-interval 50000 --greedy-eval-games 200 \
    --entropy-play-floor 0.28 --num-workers 8 --seed 42 \
    --worker-device mps --worker-compile default \
    --seed-checkpoints 'runs/rc_202609/seeds/*.pt' \
    --oracle-init runs/rc_202609/oracle/oracle_init.pt --no-entropy-controller
```

- Generation 1 seeds the population with four copies of the bootstrap final
  (`seeds/`), loads the pretrained oracle and uses fixed entropy
  coefficients. From generation 2 the entropy controller takes over
  (`checkpoints/entropy_controller.json`).
- League workers run inference on MPS with a compiled encoder. Other phases
  use CPU workers.
- Each generation ends at a boundary checkpoint, which also joins the
  population as a hall-of-fame anchor.

After each generation the orchestrator records, in `state.json` and
`generations.csv`:

- duplicate-deal head-to-head against the previous boundary;
- the PANEL-A and PANEL-B gauntlets;
- the convention battery and the aux-head reads.

It then logs an `aux readiness gen g: READY` / `NOT READY: ...` line and a
decision line:

```
gen 3: h2h +0.0312±0.0118 improving=True panel +0.0840 aux_ready=True -> continue (...)
```

The rules behind the decision (`stop_rules.py`) are in the Training Program
notebook, §5.

### Phase 3: policy iteration

One iteration `k` starts from θ_k and writes everything under
`pi/iter<k>/`.

```bash
# 1. corpus: 8,000 committee-acted games, every play node searched (~60 h)
uv run python -m sheepshead.training.distill_corpus --ckpt <theta_k> \
    --out-dir runs/rc_202609/pi/iter1/corpus --games 8000 --workers 8 --seed 20260903 \
    --p-base 1.0 --boost-lead 1.0 --boost-cs 1.5 --p-min 0.05 --p-max 1.0 \
    --committee-act-frac 1.0 --iters 256 --iters-schedule t0-lead:1024,t1-lead:512 \
    --replicates 3 --node-telemetry runs/rc_202609/pi/iter1/corpus/nodes.jsonl \
    --routed-encoder mps
# 2. advantage fit, tilted targets, supervised projection (~2 h)
uv run python -m sheepshead.training.policy_iteration all \
    --corpus-dir runs/rc_202609/pi/iter1/corpus --ckpt <theta_k> \
    --out-dir runs/rc_202609/pi/iter1 --trunk-epochs 6 --lr 3e-5 --lambda-ret 10.0 --head-epochs 4
# 3. certification (~1.2 h)
uv run python -m sheepshead.training.policy_iteration cert --ckpt <theta_k> \
    --out-dir runs/rc_202609/pi/iter1 --cert-games 1000 --cert-seeds 4 --h2h-deals 8000
# 4. bidding-only PG phase from the certified candidate (~8 h) and its cert (~0.4 h)
uv run train-ppo --phase bidding --resume runs/rc_202609/pi/iter1/distill_epoch<N>.pt \
    --run-name rc_202609/pi/iter1/bidding --league-dir runs/rc_202609/league/league \
    --until 200000 --save-interval 200000 --snapshot-interval 0 --greedy-eval-interval 0 \
    --num-workers 8 --seed 42
uv run python -m sheepshead.training.policy_iteration cert --ckpt runs/rc_202609/pi/iter1/distill_epoch<N>.pt \
    --out-dir runs/rc_202609/pi/iter1/bidding_cert --candidate runs/rc_202609/pi/iter1/bidding/final.pt \
    --cert-games 1000 --cert-seeds 4 --h2h-deals 8000 --no-routed-reads
```

| Step | What it does | Leaves behind |
|---|---|---|
| Corpus | θ_k plays 8,000 games; an ISMCTS committee scores the legal cards at every play node | `corpus/corpus_<shard>.pt`, `manifest.json`, `nodes.jsonl` |
| Fit | fits an advantage model to the committee's Q values | `advantage_model.pt`, `oof_predictions.pt`, `fit_report.json`, `row_table.pt` |
| Target | tilts θ_k's policy toward the estimated advantages | `targeted/`, `target_report.json` |
| Distill | trains the network toward the targets; the last epoch is the candidate | `distill_epoch<e>.pt`, `distill_log.jsonl`, `distill_best.json` |
| Cert | convention probes, head-to-head and head-routed reads against θ_k | `cert.json` |
| Bidding | PPO with the play network frozen, then a cert against the candidate | `bidding/final.pt`, `bidding_cert/cert.json` |

- An interrupted corpus resumes with `--start-game <n>`.
- A failed cert raises `NEEDS REVIEW`; what to do next is the operator's
  call.
- θ_{k+1} is the adopted bidding-phase checkpoint. It is also scored on
  PANEL-A and PANEL-B (`program/panel_iter<k>.npz`, `panelB_iter<k>.npz`),
  for the record only.
- Each iteration generates a fresh corpus. Earlier corpora are never reused.

The iteration summary and stop lines in `program.log`:

```
iter 1: compounding gain (play_only) +0.0036±0.0023; full h2h +0.0029±0.0025; theta_1 = runs/.../bidding/final.pt
policy iteration STOP: <reason>
```

The recipe and its adoption bars are in the CE Teacher notebook, §21.2.

### Phase 4: final certification

Under `final/`:

- `release.pt`: the adopted checkpoint
- `h2h_vs_<name>.json`: one per entry of `final.references` found on disk
- `exploit/gate_result.json`: the exploitability audit

Exporting the model is a manual step. To serve it, point the API's
`SHEEPSHEAD_MODEL_PATH` at `release.pt`.

## Starting from an existing checkpoint

`start_phase: "policy_iteration"` runs phases 3 and 4 from an external
checkpoint, drawing the bidding phase's opponents from an existing league
population. The pipeline was validated this way with this config:

```json
{
  "run_name": "rc_validate_v2",
  "arch": "perceiver-shared-v2",
  "start_phase": "policy_iteration",
  "policy_iteration": {
    "theta_0": "runs/policy_iteration_202609/iter29_d8k_t6/distill_epoch10.pt",
    "league_dir": "runs/league_retention_pg/league",
    "bidding_first": true,
    "max_iterations": 1,
    "corpus_seed_base": 20260913
  },
  "final": {"references": {}, "exploit_episodes": 0},
  "gates": {"gen2_panel_min": -10.0}
}
```

With `bidding_first`, θ₀ gets a bidding phase as iteration 0, and iteration
1's corpus comes from the adopted result. The measured results are in the CE
Teacher notebook, §21.

## Running the instruments by hand

```bash
# the adoption battery on any checkpoint pair
uv run python -m sheepshead.training.policy_iteration cert --ckpt <reference.pt> \
    --out-dir <dir> --candidate <candidate.pt> --h2h-deals 8000
# head-routed read: bidding from one checkpoint, play from another
uv run python -m sheepshead.eval.head_routed_h2h --bid-ckpt <a.pt> --play-ckpt <b.pt> --deals-per-mode 8000
# duplicate-deal head-to-head of two league boundaries
uv run python -m sheepshead.eval.league_progress_eval --h2h <gen.pt> <prev.pt>
# the anchored PANEL-A gauntlet / rigorous paired comparison
uv run python -m sheepshead.eval.rigorous_eval --help
# re-run the greedy probes on saved checkpoints
uv run python -m sheepshead.training.reprobe_checkpoints <run>/checkpoints/checkpoint_*.pt \
    --out <run>/checkpoints/greedy_health_recall.csv
```

## Reading the trainer log

Every `train-ppo` phase prints one line per PPO update:

```
Ep 41,594 | picker_avg +1.54 | pick 19% | leaster 6.9% | advσ all/pick/play 0.123/0.154/0.108 | 6.8 eps/s  ev O/L 0.67/0.54 | Hn 0.04/0.11/0.15/0.44
```

| Field | Meaning |
|---|---|
| `picker_avg` | the training agent's mean score over its last 3,000 picked hands |
| `pick`, `leaster` | rates in the training games; not comparable to the greedy probes |
| `advσ` | advantage standard deviations: all, pick, play |
| `ev O/L` | explained variance of the oracle and limited critics |
| `Hn` | normalized entropy of the pick, partner, bury and play heads |

The same numbers land in `checkpoints/training_progress.csv`, along with the
unweighted aux losses (`aux_loss_*`).

Greedy probes go to `checkpoints/greedy_health.csv`. Their aux-head fields:

| Field | Meaning |
|---|---|
| `seen-trump acc` | accuracy over all 14 trumps |
| `seen-trump recall` | accuracy on trumps the seat knows only from memory, overall and per trick `t0-5` |
| `false-seen` | rate of "seen" on trumps the seat has not seen, overall and per trick |
| `secret` | the secret-partner self label |
| `points exact` | known points per seat, exact after rounding, with the MAE |
| `unseen-higher` | accuracy of the unseen-trump-higher head |

Rising recall with flat false-seen is memory; rising false-seen is a head
that says "seen" for everything. These reads gate the league handoff and are
informational elsewhere.

## Methodology

In `notebooks/`:

- [Training_Program_Redesign_202609.md](../notebooks/Training_Program_Redesign_202609.md):
  the program. Phases in §4, stop rules and gates in §5, decision log in §0.
- [CE_Teacher_Design_202608.md](../notebooks/CE_Teacher_Design_202608.md):
  policy iteration. Pinned recipe in §20.14, close-out and final recipe in
  §21.
- [Learning_System_Redesign_202607.md](../notebooks/Learning_System_Redesign_202607.md):
  the league trainer's terminal-reward and oracle-critic design.
- [Architecture_Ablation_202607.md](../notebooks/Architecture_Ablation_202607.md)
  and [Blind_Bury_Ablation_202608.md](../notebooks/Blind_Bury_Ablation_202608.md):
  how the architecture was chosen.
- [Evaluation_Harnesses_202607.md](../notebooks/Evaluation_Harnesses_202607.md):
  the duplicate-deal instrument, PANEL-A and the probes.
