# Oracle Critic: Does It Need History, and Does Recurrence Deliver It? (October 2026)

**Status (2026-10-06): decision run COMPLETE on 100k episodes from the 7.7M league policy — offline outcome ≈ O3 (candidate ≥ from-scratch recurrent, ~6× faster), but all from-scratch arms trail the production oracle by ΔEV ≈ 0.08; adoption needs a transfer step + online check.**
Probe: `sheepshead/analysis/oracle_history_probe.py`. Related:
`sheepshead/agent/oracle.py` (production oracle),
`sheepshead/training/pretrain_oracle.py` (dataset + recipe this probe reuses),
`visualizations/ppo_architecture_3d.html` (oracle "memory" stage text, which
this verdict will correct).

## Question

The production oracle critic sees every hand, the true blind/bury, the under
card and the secret partner. It still runs a GRU over the hero's event
stream. The rationale in the code and the visualization is Baisero & Amato
(2022): under partial observability a critic conditioned on the state alone
is biased, and the sound target is the history-state value U(h, s).

1. Does that apply here, and how much does it matter?
2. If it matters, is a learned GRU the right way to supply h, or should the
   public history be an explicit input (feed-forward, no recurrence)?
3. What would each design cost in training throughput?

## Framing

**Why history can matter even with full information.** A Sheepshead
snapshot s is Markov for the *rules*, but not for the *players*. Each actor's
policy conditions on its own observation history: who led what, who showed
void, who schmeared, the order cards fell in. Two histories that reach the
same s can lead to different future play, so V^π(s) ≠ U^π(h, s) in general.

**What s is actually missing.** Less than the general theory suggests:

| Information | In s? |
|---|---|
| Which cards have been played | Yes, by complement (hands + blind + bury) |
| Points taken per seat | Yes (`points_taken_rel`) |
| Bidding sequence | Yes, implied: every seat before the picker passed; call/alone/under are context scalars |
| Current trick, by seat | Yes |
| **Who played which earlier card, in which trick, in what order** | **No** |

So the gap is exactly the card-play attribution and order. That is the
signal the conventions run on (C1/C2, partner trump-lead), so the bias is
plausibly non-zero, but it is a residual. The hidden-card variance, which is
the oracle's main payoff, is already gone.

**Why the oracle has a GRU at all.** The oracle was never given the public
history, so it has to learn to recall it, the same as the actor. Recurrence
here stands in for missing input, not a theoretical necessity. With every
hand plus the explicit public history, the oracle can reconstruct every
actor's h exactly, so a feed-forward network can represent U(h, s) with no
recall error.

## Design

Three arms, trained from scratch on one frozen-policy dataset with
`pretrain_oracle`'s split (episode index mod 10: 0 = test, 1 = val, rest
train), recipe (Adam 3e-4, clip 0.5, aux heads at 0.1/0.2, batch = 48
episodes for every arm) and early stopping (val value-MSE, patience 3):

| Arm | Input | Estimates |
|---|---|---|
| `recurrent` | production `OracleValueNetwork` over the hero stream | U(h, s) via learned recall (today) |
| `stateless` | same network, each decision row alone (T=1, zero memory) | V(s) |
| `history` | **candidate production layout**: feed-forward, one token per deck card + context = 33 tokens (vs 51) | U(h, s), explicit |

`--reference-weights` also scores the run's shipped `oracle_init.pt` on the
same test rows, as a reproduction check on the `recurrent` arm.

**The `history` layout** (`CardLocationEncoder`). Each of the 32 cards
appears exactly once, as MLP(card, location, seat, role, trick, order,
under):

| Location | Seat | Trick / order |
|---|---|---|
| hand | holder | — |
| blind | — | — |
| bury | picker | — |
| held under (declared, set aside face-down, not yet played) | picker | — |
| current trick | player | trick, position from the leader |
| played in an earlier trick | player | trick, position from the leader |

Role is that seat's picker / secret-partner bits, as on the production
opponent-hand tokens. The under flag marks the called-under card wherever it
is. The context token is the production oracle's, unchanged. The card
embedding (informed init), seat/role embeddings, transformer, readout, trunk
and aux heads are all reused; there is no memory token, no GRU, and no
per-bag MLPs.

The location table is rebuilt from data already in the dataset. Each
completed trick arrives in the hero stream as an observation event, and each
trick's leader is the `leader_rel` the hero saw while acting in it. A
face-down under in a trick resolves to the true `under_card_id`. The rebuild
asserts 5 cards per completed trick and all 32 cards located on every row;
both hold on all 40k episodes / 336k rows of the run's dataset.

**Confound to keep in mind.** `history` differs from `stateless` in *two*
ways: it has the history, and it uses a different layout (per-card vs.
per-bag). So history − stateless mixes "history is worth something" with
"the layout is better or worse". The head-to-head that decides adoption is
**history vs recurrent**: candidate vs production, which is exactly the
question.

**Metric.** G (the realized return; γ = 1, terminal reward) is unbiased for
U(h, s) under the data policy, so for any arm

  MSE = E[(v − U)²] + E[Var(G | h, s)].

The paired per-row difference ΔMSE(A, B) is therefore a pure **bias²
difference**: the irreducible return noise cancels exactly. It is reported
overall and per stratum (`pretrain_oracle.STRATA_ORDER`) with an
episode-clustered SE. ΔEV = −ΔMSE / Var(G).

**Cost.** Each arm records train-step seconds and decision rows/sec. The
recurrent arm's rate includes its true overhead: observation events,
padding, and T sequential encoder calls per batch.

## Pre-registered decision rules

An effect counts when it clears all three bars:

- |z| ≥ 3 (episode-clustered),
- |ΔEV| ≥ 0.005 (materiality; Var(G) ≈ 0.128 on the run's dataset, so
  |ΔMSE| ≥ ≈ 6e-4),
- larger than the seed-to-seed spread of the same arm's test MSE. The
  production val curve oscillates by ~0.01 epoch to epoch, so the decision
  run uses **≥ 2 seeds**.

| Outcome | Reading | Action |
|---|---|---|
| O1: stateless ≈ recurrent ≈ history | History carries no value info at this policy | Candidate is no worse and cheaper: adopt it on cost grounds; rewrite the viz text |
| O2: history ≈ recurrent < stateless | History matters; the GRU recovers it | Candidate matches production at lower cost: adopt; online A/B next |
| O3: history < recurrent | The GRU loses information (or the layout helps) | Adopt the candidate; online A/B next |
| O4: recurrent < history | The candidate misses something (bidding order? the per-bag layout?) or doesn't optimize | Investigate before any change |

Strata to watch: `play_lead_t02_defender` and `play_lead_t02_secret_partner`
(convention nodes), and `play_t3plus` (where history is longest).

**Offline is a proxy.** The real payoff is advantage variance and bias in
GAE. Any architecture change still goes through an online paired comparison
before adoption.

## Data caveat

The run's dataset (`runs/202609_recall_rc/oracle/dataset.pt`, 40k episodes,
~336k decision rows) was generated by the **bootstrap** policy. League play
sharpened the conventions to 84–98%, which may *raise* the history
dependence. If O1 shows up on the bootstrap data, regenerate from a current
league checkpoint before concluding:

```
uv run python -m sheepshead.training.pretrain_oracle generate \
    --ckpt runs/202609_recall_rc/league/checkpoints/<latest>.pt \
    --episodes 40000 --workers 8 --gamma 1.0 \
    --out runs/oracle_history_probe/dataset_league.pt
```

## Run

Decision run (agreed plan: 100k episodes from the 7.7M league policy;
recurrent one seed; the production oracle from the same checkpoint as the
reference arm):

```
mkdir -p runs/oracle_history_probe
set -o pipefail
PYTHONUNBUFFERED=1 uv run python -m sheepshead.training.pretrain_oracle generate \
    --ckpt runs/202609_recall_rc/league/checkpoints/checkpoint_7700000.pt \
    --episodes 100000 --workers 8 --gamma 1.0 \
    --out runs/oracle_history_probe/dataset_league_7700k.pt \
    2>&1 | tee runs/oracle_history_probe/generate_7700k.log \
&& PYTHONUNBUFFERED=1 uv run python -m sheepshead.analysis.oracle_history_probe \
    --dataset runs/oracle_history_probe/dataset_league_7700k.pt \
    --reference-weights runs/202609_recall_rc/league/checkpoints/checkpoint_7700000.pt \
    --seeds 42,43 --recurrent-seeds 42 \
    --out runs/oracle_history_probe/league_7700k.json \
    2>&1 | tee runs/oracle_history_probe/probe_7700k.log
```

`pipefail` makes the `&&` gate on the Python exit status rather than
`tee`'s, so a failed generation doesn't start the probe.

`--reference-weights` accepts a full checkpoint (reads `oracle_state_dict`)
or a bare state_dict. Arms that share seeds are compared seed-for-seed;
otherwise every seed is crossed with every seed of the other arm.

**Reading the reference arm.** The production oracle has seen ~7.7M
episodes of this policy (online, on λ-returns, lagging a moving policy),
while the from-scratch arms see ~80k. So `history − reference` is tilted
toward production; a candidate that matches or beats it is strong
evidence. `recurrent − reference` measures how data-starved the
from-scratch recurrent arm is, which tells us how to read
`history − recurrent`.

Estimated cost on CPU, from measured full-dataset epochs (history 78 s,
stateless 126 s) and the earlier smoke ratio for recurrent (~3× stateless):
at ~20 epochs, about 2 h per seed for `recurrent`, 45 min for `stateless`,
25 min for `history`. Roughly 6–7 h for two seeds, more if it overlaps the
live league run.

## Log

- **2026-10-05.** Probe written. Smoke test: 300 episodes, 1 epoch, CPU, all
  three arms plus the reference. The pipeline works end to end and the
  history-rebuild assertion holds. Smoke numbers are not evidence; one epoch
  is init noise. CPU train throughput at smoke scale (decision rows/s):
  recurrent 527, stateless 1650, history 1089. That is the first hint that
  dropping recurrence pays for 25 extra tokens about twice over.
  Reference `oracle_init.pt` on those 237 test rows: EV 0.58 (its report
  says 0.535 on the full test set), consistent.

- **2026-10-05 (later).** Per operator: the `history` arm now uses the
  candidate production layout (one token per card plus location, 33
  tokens), replacing the 76-token append version, so the experiment runs
  under the conditions the oracle would actually ship with. First rebuild
  failure: a called under leaves the picker's hand when declared, so it was
  unlocated until played. Added the held-under location; the rebuild then
  passed on the full dataset. One full-data epoch each on CPU: history
  78 s / val MSE 0.0682, stateless 126 s / 0.0809 (production recurrent's
  epoch-1 val was 0.0825). One epoch is not evidence, but the throughput
  side already looks clear: the candidate is ~1.6× stateless and ~5×
  recurrent per epoch.
- **2026-10-05 (device check).** One epoch on 8k bootstrap episodes, with
  the league trainer running. Seconds per epoch: MPS recurrent 111 /
  stateless 49 / history 49; CPU 72 / 19 / 14. Decision rows/s: CPU 628 /
  2345 / 3360. **CPU wins every arm**, so stay on the CPU default. The
  candidate is ~5.4× recurrent per decision row and has 357k params vs
  624k. Plan agreed with the operator: regenerate **100k episodes** from
  the 7.65M league checkpoint; recurrent ×1 seed, stateless and history ×2
  seeds; production oracle as a reference arm. Extrapolated (×12.5, ~20
  epochs): recurrent ~5 h, stateless ~1.3 h/seed, history ~1 h/seed,
  ≈ 9–10 h total, less if early stopping comes sooner.
- **2026-10-05 (probe changes).** Added `--recurrent-seeds` and full-checkpoint
  `--reference-weights` (plus reference pairs in the comparisons). Smoke
  test (300 bootstrap episodes, 1 epoch): seed pairing is correct, and both
  reference formats load. The 7.65M production oracle scored EV 0.65 on
  those 237 bootstrap test rows vs 0.58 for `oracle_init.pt`; that's a
  small n, noted only as a sanity check.
- **2026-10-05 (checkpoint).** The run passed 7.7M, so the decision run
  uses `checkpoint_7700000.pt` for both data generation and the production
  reference (confirmed oracle mode, aux heads present). Earlier log entries
  that say 7.65M describe the plan at the time.

- **2026-10-06 (decision run).** 100k episodes from `checkpoint_7700000.pt`
  (train 80k / val 10k / test 10k episodes, 75,320 test rows; Var(G) ≈ 0.072,
  so the ΔEV 0.005 materiality bar is ΔMSE ≈ 3.6e-4). Output:
  `runs/oracle_history_probe/league_7700k.json`, log `probe_7700k.log`.

  | Arm | Test EV | Test MSE | Rows/s (train) | Epochs (best) | s/epoch |
  |---|---|---|---|---|---|
  | reference (production, 7.7M online) | **0.711** | **0.0208** | — | — | — |
  | recurrent s42 | 0.612 | 0.0279 | 620 | 18 (14) | ~950 |
  | stateless s42 / s43 | 0.602 / 0.617 | 0.0286 / 0.0275 | ~2,600 | — / 23 (19) | ~225 |
  | history s42 / s43 | 0.630 / 0.622 | 0.0266 / 0.0272 | ~3,820 | 25 cap (23) / 18 (14) | ~157 |

  Paired ΔMSE (negative favors the first arm; z episode-clustered):
  - history − recurrent (s42): **−0.0013, z −5.0** (ΔEV +0.018). s43 vs
    recurrent is not paired by the probe (no shared seed); from the MSEs it
    is −0.0007. Seed spread of history = 0.0006.
  - stateless − recurrent (s42): +0.0007, z +1.8; stateless s43 has a
    *lower* MSE than recurrent. The from-scratch GRU extracts ~nothing
    overall beyond V(s), though it does help at `play_t3plus` (stateless
    worse there by +0.0027, z +9.8) and in leasters.
  - history − stateless: s42 −0.0020 (z −6.1), s43 −0.0004 (z −1.3). The
    sign is consistent; the size isn't. Both seeds are better at `pick`
    (−0.003 to −0.004, z ≈ −4.5). Pick rows have *no* history, so that gain
    comes from the per-card layout, not from history (the confound noted in
    Design). Both seeds are also better at `play_t3plus` and at defender
    leads (small, −3e-4).
  - Every from-scratch arm − reference: +0.006 to +0.008, z 12–19, in
    every stratum. Data and training age dominate architecture: the gap to
    production (~0.007) is ~5× the gap between the architectures (~0.001).

  Against the pre-registered bars: history beats recurrent on z and
  materiality, and the effect (~0.001) exceeds history's seed spread
  (0.0006). Recurrent ran one seed only, though, and stateless's spread is
  0.0011, so the architecture margin is of the same order as seed noise
  for the 624k-param nets. **Reading: O3, weakly** — the candidate is at
  least as good as a from-scratch recurrent oracle, plausibly slightly
  better (partly from the layout), at ~6× the training throughput and 57%
  of the params. Leaster rows are the one stratum where the candidate is
  worse than recurrent (+0.002, z +3.5).

## Verdict

**Offline (2026-10-06): candidate ≥ recurrent from scratch, ~6× faster;
NOT yet ≥ production.** Switching the oracle architecture outright would
throw away a 7.7M-episode critic (ΔEV ≈ 0.08 better than any fresh fit) and
degrade the GAE baseline at the switch. Candidate next steps:

1. **Distill production into the candidate.** Regress the candidate on the
   production oracle's values (optionally mixed with G) over freshly
   generated league data. That inherits the 7.7M episodes of experience,
   and the targets are far less noisy than G. Gate: candidate test MSE vs
   G within the materiality bar of production.
2. **Online check.** Resume the league with the distilled candidate as the
   oracle; compare `ev O/L`, advantage σ and eps/s against the current
   run's trajectory.
3. Optional: a second recurrent seed, to firm up the architecture margin.
