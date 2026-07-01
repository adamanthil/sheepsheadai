# League Repro Run Review & Fix Plan (July 2026)

Review of the first long league run (`train_league_ppo.py`, log
`runs/repro_league_train.log`, run dir `runs/repro_league/`), assessed at
episode ~13.68M (mid generation 14). Companion to
`Exploiter_League_Plan_202606.md` (the design this run executes) and
`Exit_Arms_202606.md` (the evidence base). Outcome: the design is sound, but
the run exposed two implementation gaps that mean the league mechanism has
barely been exercised yet — this run is best treated as the **PFSP-only
control arm**, with a fresh run after the fixes below.

## 1. Run configuration (as executed)

- Resumed from `runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt`;
  league seeded with the 20 selfplay snapshots as past_mains.
- 1M-episode main phases; 50k-episode exploiter phases; gate = paired
  duplicate-deal edge ≥ 0.10 score/deal AND ≥ 2·SE over 1000 deals.
- Terminal reward only, no anchor, 8 workers, ~12.2 eps/s (~23h per main phase).
- Mid-run code changes landed while it was running — notably
  `67e2821` (2026-06-25): exploiter seat share driven off the frozen gate edge
  instead of the live EMA. Several crashes/restarts (duplicate GENERATION 8/10
  headers; `runs/repro_league_corrupted` is a casualty of one of them).

## 2. Design verdict

Well-motivated and correct in its core:

- The main/exploiter generation loop is a faithful, appropriately simplified
  AlphaStar-style league; the gated best-response edge as the empirical
  exploitability headline is the right grounded metric.
- The paired duplicate-deal gate (CRN, same seat, greedy everywhere,
  challenger−incumbent delta) is good variance reduction: SE ≈ 0.065
  score/deal at 1000 deals (~3x better than unpaired).
- The frozen-main throwaway league trick in `exploiter.py` (4 copies, shares
  zeroed) cleanly reuses `run_main_phase` — no second training loop.
- Frozen-gate-edge seat share (`league.py exploiter_share`) fixes a real
  failure — the log confirms the old EMA share decayed 0.03 → 0.00 within one
  update of the gen-1 insertion, while the gen-12 exploiter has held a steady
  0.154 share for 1.6M+ episodes since the fix.
- Terminal-only reward + greedy argmax health probes as collapse guard match
  the run-2 / ExIt-collapse lessons.

## 3. Findings (ranked)

### F1. The anti-forgetting layer does not exist in practice
`ROLE_HOF_ANCHOR` is only assigned in `League.migrate_legacy`; a
`--seed-checkpoints` run never creates one, so `hof_floor_prob` is dead code.
Worse, skill pruning is structurally newest-wins: new snapshots enter at the
OpenSkill prior μ=25 while the rated population's μ scale has drifted to
roughly −6..0 (training μ prints −5..+1), so `_manage_size` "highest skill"
retention systematically keeps the least-rated newest members.
**Observed roster at 13.68M: 30 past_mains spanning episodes 9.85M–13.65M
only; all 20 seed checkpoints pruned; zero HOF anchors.** The league is a
~4M-episode sliding window of recent selves — much closer to plain self-play
with a short history buffer than to a league. The classic cycling failure mode
is left open, and §5's bidding oscillation is consistent with it.

### F2. Exploiter pressure was inert for generations 1–11
Pre-fix, the EMA-driven seat share ratcheted to 0.00 immediately after each
insertion (gen-1 and gen-3 exploiters therefore never applied table pressure).
Only since `67e2821` + the gen-12 insertion has the mechanism actually run.
Consequently the flat exploitability trend so far (see §5) mostly measures
PFSP-only training, not the league design. Interpretation, not a bug to fix —
but it means this run cannot confirm or refute the league; it is the control.

### F3. A failed exploiter run is indistinguishable from a robust main
Gens 4/5/6 produced **negative** gate edges (−0.258, −0.075, −0.171): the
best-response run made the challenger worse than its own warm start (fresh
schedule ⇒ LR 1.5e-4 + start entropies is hot for a warm start). Since the
exploiter *starts as* the main (edge ≡ 0 at init), the exploitability
estimate should never be negative. Gate every saved exploiter checkpoint
(5k cadence) and take the running best → monotone, non-negative lower bound
that also flags exploiter-training divergence for free.

### F4. Gate power is marginal at the threshold
SE ≈ 0.065–0.10 makes the effective bar max(0.10, 2·SE) ≈ 0.13–0.20. Edges of
0.080/0.097/0.112/0.074 (gens 2/8/10/11) were discarded — plausibly-real small
edges the league never trains against, and the exploitability record
undercounts. Greedy deals are cheap: 3–4k gate deals ≈ halved SE.

### F5. Exploiter retirement counts insertions, not elapsed generations
`_manage_size` uses `current_gen = max(member.generation)`, which advances only
on a successful insertion. The gen-1 exploiter stayed "active" through gen 11.
With frozen-edge share this means a beaten exploiter can hold ~15% of seats
indefinitely while gates keep failing. Intended as a floor; needs a ceiling —
count elapsed generation boundaries instead.

### F6. Bookkeeping
Generation 7 was silently skipped by the absolute-episode renumbering on the
restart from 7.65M (`first_gen = 7.65M // 1M + 1 = 8`) — the gap in
`exploitability.csv` is renumbering, not a failed gate. Training μ (`mu_jd`,
`mu_ca`) has drifted to a scale where cross-time comparison is meaningless;
it is currently logged as if informative.

### F7. No absolute-strength signal anywhere in the run
Every logged metric is relative to a moving field (picker_avg, ratings) or
behavioral self-play stats (greedy probes). Nothing answers "is the main
stronger than it was 5M episodes ago?" — the single question that gates all
other decisions.

## 4. Run health at ~13.68M episodes

- **Throughput**: steady 12.2 eps/s throughout; no degradation.
- **Exploitability**: successful gates 0.171 (gen 1), 0.148 (gen 3), 0.154
  (gen 12) — flat over 11M episodes, but see F2 (mechanism inert for most of
  it). Gen 13 = −0.043 is a failed exploiter run (F3), not evidence of
  robustness.
- **No collapse; play head recovered**: the play-logit-spread gate fired
  constantly in gens 0–2 (spread < 0.5, the terminal-only collapse signature
  from 035f716) but spread has since climbed to ~2.2–2.9. Trump-lead mostly
  0–7% recently (one 55% spike at 8.55M, self-corrected).
- **Bidding policy oscillates far beyond probe noise**: greedy PICK swings
  12%→64% between 50k probes (binomial noise at n=200 ≈ 3pp), leaster 0%→52%,
  ALONE spiking past the 12% gate repeatedly since 12.8M (up to 27%).
  picker_avg flat ~+1.0 since ~200k. Cycling signature, consistent with F1.
- **Gate violations are warnings only** — ~1 per 2–3 probes; nothing
  aggregates or acts on them, so they've become log noise.

## 5. Scripted agent: useful or superfluous?

Decision: **superfluous as a strength yardstick; useful as a sanity floor and
as a carrier for scripted exploit probes.** (`scripted_agent.py` in the repo
root is a placeholder stub, so this is a build-from-scratch decision.)

- Against (as a rating opponent): `rigorous_eval.py`'s frozen reference field
  already provides the static scale, and the 30M reference is far stronger
  than any feasible heuristic agent. Score is zero-sum relative to the field:
  vs a weak scripted field, strong candidates' differences compress toward
  noise and the number measures weak-player-farming efficiency, which is not
  monotone in true strength.
- For (three narrow uses):
  1. **Lineage decorrelation** — every NN anchor shares one training lineage,
     which demonstrably shares blind spots (the trump-lead leak survived 30M
     episodes because every opponent had it too). A rule-based agent is the
     only instrument that cannot share those blind spots; include it as ONE
     panel member in the reference field, not the whole field.
  2. **Scripted exploit probes (the strongest case)** — hand-code each
     *diagnosed* exploit (first: a defender that reads an information-revealing
     trump lead and acts on the inference) as a static probe. Answers "is this
     known hole closed, and by how much?" in fixed units, in seconds, forever —
     complementary to the learned exploiter's "does any exploit exist?", and
     immune to F3-style exploiter-run noise. Grows into a permanent regression
     suite as new flaws are diagnosed.
  3. **Legible floor + smoke test** — "beats a solid conventions player by X
     score/hand" is the only externally meaningful claim available, and losing
     to it flags breakage that μ-vs-own-checkpoints cannot.
- Scope if built: sound-amateur conventions only (real pick threshold,
  sensible bury, follow/schmear/trump-in logic). ~A day. Not a strong engine.

## 6. Work plan (one at a time, before any new run)

**Step-1 RESULT (2026-07-01, runs/rigorous_baseline_202607/):** the run IS
climbing on the PANEL-A scale — score/hand CA: 1M −0.304 → 7M −0.238 →
13.65M −0.120; JD: 1M −0.165 → 7M −0.200 (ns dip, failed-exploiter era) →
13.65M +0.038. All 13.65M paired diffs significant (+0.12..+0.24, p<0.001).
Verdict: bidding oscillation = heat not sickness; endpoint ≈ panel average
(clearly above selfplay seed, not yet clearly above the 30M lineage's mid
rungs). Since exploiter pressure was inert for gens 1–11 (F2), this curve is
effectively the PFSP-only control trajectory the fixed league must beat.
**Matched-episode comparison (reference_lineage_*.csv, same seed-42 deals +
field):** the repro trajectory is BEHIND the original pfsp lineage at matched
training. Reference pfsp-15M scores CA +0.120 / JD +0.111 vs the league
13.65M's CA −0.120 / JD +0.038 (gap ≈ 0.24 CA / 0.07 JD); reference pfsp-5M
(CA +0.033 / JD +0.066) already matches-or-beats the league endpoint. Two
non-exclusive readings: (a) the original run's shaped per-trick reward buys
early learning speed that terminal-only gives up (accepted trade for
unbiased signals); (b) the repro's slow start (play-head collapse gens 0–2,
inert exploiters, failed-exploiter era) cost real ground. Consequence for
step 7: success criteria must compare the new run against BOTH trajectories
at matched episodes, and reaching 30M-reference strength by ~14M episodes is
not the realistic bar — beating the repro control curve is.

**Step-6b RESULT (2026-07-01, analysis/trump_lead_probe.py, frozen seed
20260702, 2000 deals × 5 seats, scripted field):** the diagnosed trump-lead
hole is still open in both lineages, concentrated in trump-rich hands as
documented. 30M reference: JD 1.89% of defender-lead opportunities (6.74%
trump-rich), CA 0.80% (1.61%). League 13.65M: JD 1.21% (3.70%) — better —
but CA 2.88% (6.28%) — ~3.6x WORSE than the reference. Implied EV −0.25..
−0.99 score/1000 hands: real but ~100x smaller than the strength-curve
moves, i.e. a canary/regression marker, not a strength lever. 13.65M
episodes of league PPO did not close it (consistent with the documented
PPO-can't-learn-small-early-gaps mechanism); track per-checkpoint in the
next run via this probe's fixed units. JSONs:
runs/rigorous_baseline_202607/trump_lead_probe_{30M,league13650k}.json.

**Status 2026-07-01:** step 1 done (results above);
step 2 done (76599a6); step 3 done (acffc1e); step 4 done (d88b7bb); step 5
done (e67306c); step 6 agent done (1facce1, exploit probe pending baseline);
step 7 pending. Bonus: the scripted agent's self-play smoke exposed a
leaster tie-break bug — `get_leaster_winner` re-rolled the tie on every
call, so tied leasters scored two +4s / none and broke zero-sum across
every score-based tool — fixed + regression-tested in f67a827.
Scripted-agent placement (150 paired deals): beats selfplay-100k
+0.63±0.24; the 13.65M league main clears it by only +0.34±0.21 — it is
mid-ladder, not a floor.

1. **Diagnose before changing anything** (cheap, gating): run
   `analysis/rigorous_eval.py` (committed f4ce6f7) with heroes at
   ~1M / 7M / 13.65M vs PANEL-A (below). Flat curve ⇒ F1 is confirmed as the
   binding constraint; rising curve ⇒ the oscillation is heat, not sickness.
   Either way this sets the baseline the next run must beat. Power note from
   the smoke run: ~531 deals resolve a 0.10 score/hand gap (paired), so 1000
   deals ⇒ MDE ≈ 0.07. Run both partner modes.

   **PANEL-A (frozen reference field — never change; sha256 prefixes):**
   | member | path | sha256 |
   |---|---|---|
   | 30M reference (top rung) | `final_pfsp_swish_ppo.pt` | `cc644b7109d5896b` |
   | pfsp mid rung | `runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt` | `6951cd42c52a9a84` |
   | pfsp early rung | `runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_5000000.pt` | `c8ec0d1df24875c8` |
   | selfplay common ancestor (floor) | `runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt` | `7f5426b68d3ebc6b` |

   Selection rationale (validity constraints, in order of importance):
   (a) *equal relatedness* — every member branches at/before the selfplay
   seed or on the independent 30M-run branch, so no candidate on the
   repro-league curve is more related to the field than any other (a panel
   member drawn from the run under test would bias late checkpoints; this
   also disqualifies the gen-12 exploiter, the only surviving gate-passed
   exploiter, which warm-started from the repro main at 12M — and gen-1/3
   exploiter weights are lost: run dirs deleted, league copies pruned by F1);
   (b) *strength ladder* (100k → 5M → 15M → 30M) so the metric has dynamic
   range for weak and strong candidates and a single-opponent quirk exploit
   can't dominate — with ≥4 anchors rigorous_eval seats the full panel in
   every game, so every hand contains all four rungs;
   (c) *frozen bytes* — hashes above; re-verify before any future run.
   Known limitation: all four share one family tree (no independent lineage
   exists yet). Scripted agent (step 6) will be the decorrelated probe, kept
   OUT of PANEL-A so the scale never changes; if a future PANEL-B adds it,
   both panels can be run side by side for continuity.
2. **Fix roster retention (F1)**: initialize snapshot ratings from the current
   training rating (not the μ=25 prior) so skill pruning compares on one
   scale; add an HOF promotion rule (e.g., periodically tag the best
   evaluated member; seeds eligible). Verify: after a simulated few
   generations, roster spans old + new, HOF non-empty.
3. **Make the exploitability measurement trustworthy (F3 + F4)**: gate the
   running-best over exploiter checkpoints; raise gate deals to ~3–4k.
   Optionally try a tempered exploiter LR afterward — measurement fix first.
4. **Retirement ceiling (F5)**: key `exploiter_retire_generations` to elapsed
   generation boundaries, not successful insertions.
5. **Add the absolute anchor to the trainer (F7)**: small periodic CRN paired
   eval vs the frozen 30M reference at probe cadence, logged next to the
   greedy probe; alert on decline. Drop or de-emphasize the drifting μ logs (F6).
6. **Scripted agent (§5 scope)**: conventions-level `ScriptedAgent` as
   rigorous_eval panel member + smoke test; then the first scripted exploit
   probe (trump-lead inference defender).
7. **Fresh league run** with fixes 2–5, treating the current run as the
   PFSP-only control arm for comparison at matched episode counts.

## 7. Reference numbers (this run)

| gen | main ep | edge ± SE | inserted |
|----:|--------:|----------:|:---------|
| 0 | 1.1M | +0.050 ± 0.066 | no |
| 1 | 2.1M | +0.171 ± 0.061 | yes |
| 2 | 3.1M | +0.080 ± 0.067 | no |
| 3 | 4.1M | +0.148 ± 0.062 | yes |
| 4 | 5.1M | −0.258 ± 0.103 | no (failed run) |
| 5 | 6.1M | −0.075 ± 0.064 | no (failed run) |
| 6 | 7.1M | −0.171 ± 0.089 | no (failed run) |
| 7 | — | skipped (restart renumbering) | — |
| 8 | 8.0M | +0.097 ± 0.062 | no |
| 9 | 9.0M | −0.072 ± 0.068 | no |
| 10 | 10.0M | +0.112 ± 0.074 | no |
| 11 | 11.0M | +0.074 ± 0.102 | no |
| 12 | 12.0M | +0.154 ± 0.067 | yes |
| 13 | 13.0M | −0.043 ± 0.077 | no |
