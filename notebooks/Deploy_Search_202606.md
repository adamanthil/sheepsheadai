# Deploy-Time Search — Design Notes (June 2026)

The June 2026 consolidation (see `Exit_Arms_202606.md`, `Exploiter_League_Plan_202606.md`)
ended with one demonstrated strength mechanism: **decision-time ISMCTS at play
time**. Search-as-teacher (ExIt) was retired — stable in its grounded form but
zero greedy value-add, with one demonstrated PASS-collapse mode — and the
training stack consolidated onto `train_league_ppo.py`. This notebook tracks the
remaining open program: turning the measured search value-add into a deployable
playing agent that meets a human-table latency bar (~1s typical, a few seconds
worst case).

## Units convention

All EV numbers below are **real sheepshead game score** (the `get_score()`
scale, e.g. −2 per defender on a 61–90 picker win), NOT the training-normalized
`final_score / RETURN_SCALE` scale. The search engine's `root_q` is internally
normalized (/12); every probe that reports a Q-derived quantity multiplies by
`RETURN_SCALE = 12` first. Cross-check: the value-add probe's realized paired
delta is computed from raw `get_score()` with no scaling and lands on the same
magnitudes as the Q-derived numbers.

## Established results (value-add probe, `validation/teacher_value_add_probe.py`)

Paired duplicate deals, greedy self-play field, probe seat plays argmax(π′) at
every PLAY decision (bidding raw in both arms):

| iters (play) | paired delta (pts/deal) | deviation rate | conditional gain | latency/decision |
|---|---|---|---|---|
| 96  | +0.042 ± 0.020 | 1.3% | +0.54 ± 0.25 | ~1.5–2 s |
| 384 | +0.103 ± 0.050 | 2.4% | +0.74 ± 0.34 | ~4–10 s |
| 768 | +0.107 ± 0.042 | 3.3% | +0.58 (falling) | ~2× 384 |

Value-add **saturates at ~384 iters**; the 384 operating point is the deploy
target. Flat 384 everywhere fails the latency bar.

## Selective-search trigger diagnostic (2026-06-12) — VERDICT: confidence gating FAILS

`validation/search_trigger_diagnostic.py`, 150 deals / 900 PLAY decisions,
384 iters, baseline 30M model both probe and field. The probe seat plays
policy-greedy (the no-search deployment trajectory); at every PLAY node it logs
cheap policy-only features (top-1 margin, entropy, trick, #legal) plus search's
self-assessed value of deviating, `q_gap = 12 × (root_q[search_argmax] −
root_q[policy_argmax])` (0 when search agrees). Offline sweep: trigger rate
(= fraction of PLAY decisions searched = latency cost) vs fraction of total
q_gap EV captured. Raw rows: `/tmp/search_trigger_384.csv` (re-analyze with
`--analyze-only`).

**Cross-validation:** total self-assessed EV = **+0.122 pts/deal**, matching
the realized +0.103 ± 0.050 from the value-add probe. The per-node q_gap signal
is trustworthy in aggregate, so the curves below mean what they say.

1. **Forced-move skip is free and large: 36.3%** of PLAY decisions have one
   legal card. Zero search, zero EV lost. **Trick 5 is also worthless**
   (q_gap ≈ 0; almost all forced/trivial).
2. **Policy confidence does NOT predict where search helps.** Margin trigger:
   searching the lowest-margin 10% of nodes keeps only **28%** of the EV; 30%
   trigger keeps 35%; reaching 80% requires searching ~**59%** of decisions
   (margin < 1.0, i.e. nearly every unforced node). Entropy trigger is
   equivalent (10% → 32%, 30% → 35%). The profitable deviations frequently sit
   at nodes where the policy is *confident but wrong* — precisely the errors a
   self-play policy cannot flag about itself. (Consistent with the trump-lead
   finding in `Exit_Arms_202606.md`: leaks live where the policy is confident.)
3. **EV by trick** (pts/deal of the +0.122): t0 +0.012, t1 +0.005, t2 +0.008,
   **t3 +0.067**, t4 +0.030, t5 0.000. Spread across tricks 0–4 with the bulk
   mid-game — not concentrated early where rollouts are deepest.
4. Deviation rate 4.2% of searched nodes (24/573), in line with the value-add
   probe's 2.4% (different node mix: that probe's trajectory includes
   search-altered states).

**Deploy implication — the question is closed:** node *selection* cannot rescue
latency beyond the free skips. Skip forced moves + trick 5 (≈100% EV kept,
~55–60% of nodes still searched → ~3.5–4 s average at current speed); beyond
that, any confidence gate trades EV near-linearly for latency. Hitting the
latency bar must come from **making each search cheaper**, not searching fewer
nodes:

- **Time-budgeted / anytime search**: cap wall-clock per decision (~1–1.5 s ≈
  96–128 iters keeps ≥ +0.042 of the +0.103); early-stop when the visit leader
  is uncatchable within budget.
- **World-pool reuse**: amortize determinization + recurrent-memory replay
  across iterations (per-iteration re-derivation is a dominant cost).
- **Batched / GPU leaf inference**: current numbers are single-process CPU with
  batch-32 leaf evals; server-class or GPU inference plausibly 3–5×.
- **Iteration reallocation** (untested): trick-3/4 nodes carry most EV per the
  table above; if a fixed total budget is reallocated from t0–t2 (deep rollouts,
  expensive, little EV) toward t3–t4, the same latency may buy more EV. Worth a
  one-off probe variant before engineering.

If those land 3–5×, flat-384-minus-free-skips comes in around ~1 s average,
p95 of a few seconds on early-trick nodes — the human pattern, with the full
+0.10 retained.

## Status

- Trigger question: **answered (negative) and closed**, 2026-06-12.
- Next: deploy-engineering direction per the menu above; pick after the
  from-scratch league reproduction (`runs/repro_league`) is underway and the
  selective-search engineering can be tested against its checkpoints too.
