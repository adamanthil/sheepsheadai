# Defender Trump-Lead Investigation (final_pfsp_swish_ppo.pt)

Date: 2026-06-17 · Baseline commit: `64e2b36` · Author: analysis session log

## 0. Summary

A from-scratch re-investigation of the "defender leads trump on an early trick
when a fail lead is available" behavior, this time on the **30M sister policy**
`final_pfsp_swish_ppo.pt` and extended to **tricks 0 and 1** (not just trick 0).

**Result:** the behavior is a **real but mild systematic suboptimality**, not a
per-hand blunder. Over 31 first-two-trick defender trump leads, the
hindsight-free belief-pool estimate puts the trump lead at **−0.19 leader game
score (SE 0.08), −3.1 defender card points, −4.5% win rate** versus the best
fail lead, and a properly-tuned ISMCTS search prefers a **fail lead in 22/30**
ESS-valid cases (and rates the policy's *specific* lead optimal in only 4/30).
The leak is concentrated in **trump-rich hands** (leading from 3–4 trump is
clearly worse; from 1–2 trump it is fine). This is directionally and roughly
quantitatively consistent with the original trick-0 finding (~−0.23 score) in
§1 of the refactor plan, now reproduced on the sister checkpoint with an
independent toolchain.

A methodological caution surfaced and was corrected mid-investigation: at the
**default ISMCTS audit settings** (`root_explore_frac=0.25`, recommendation by
visit count) the search *appeared* to endorse the trump lead — but that was a
**prior-domination artifact**. Flattening the root prior (`root_explore_frac=1.0`)
and reading the **value (top@Q)** rather than visit counts reversed the verdict.

---

## 1. Relationship to the original investigation

This is a **from-scratch followup** to the initial investigation referenced in
**§1 of `notebooks/ISMCTS_Teacher_Refactor_Plan.md`**. Differences:

| | Original (refactor plan §1) | This followup |
|---|---|---|
| Policy | `pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt` (round-number 30M checkpoint) | `final_pfsp_swish_ppo.pt` (the **30M sister policy**) |
| Scope | trick 0 only | tricks 0 **and** 1 |
| Method | paired MC on **true deals** only (`validation/counterfactual_trump_leads.py`) | 3-rung ladder: true-deal MC **+** belief-pool MC **+** ISMCTS-by-Q |
| Tooling | `validation/...`, `gate0_determinizer.py`, `critic_probe.py` | new `analysis/scan_defender_trump_leads.py`, `analysis/counterfactual_trump_leads.py` |

The two checkpoints are **sisters**: per the project's provenance notes,
`final_pfsp_swish_ppo.pt` is the 30M run's `final_swish.pt` (30M training +
one flush update / +6 optimizer steps), behaviorally near-identical to the
round-number checkpoint (median behavioral KL ≈ 0, ~98.6% argmax agreement). So
this is a genuine independent reproduction on a sibling model, not a re-run on
the same artifact.

Original §1 finding being followed up (verbatim context): *"The 30M PFSP agent …
leads trump as a defender on the first trick when a fail lead is available …
Paired counterfactual rollouts on true deals showed this is genuinely
EV-negative (~−0.23 game score, ~−4 card points; n=86). It is rare (~1.6% of
trick-0 defender leads) but it is a real mistake …"* The refactor plan's thesis
is that the leak is a **partial-observability ceiling** the critic cannot fix,
and that an **inference-weighted determinized search teacher** is the principled
remedy. This log neither re-derives nor depends on that thesis; it independently
measures whether the behavior is EV-negative on the sister policy and whether a
deploy-time search corrects it.

---

## 2. Toolchain built for this investigation

1. **`analysis/scan_defender_trump_leads.py`** — sweeps seeds through the exact
   deterministic `/analyze` path (`server.services.analyze.simulate_game`) and
   flags every spot where a **defender** (not picker, not revealed partner, not
   secret partner; non-leaster) *leads* trump on a trick while a fail lead was
   legal. Every reported `(seed, partnerMode, stepIndex)` reproduces
   byte-for-byte in the web `/analyze` viewer.

2. **`analysis/counterfactual_trump_leads.py`** — the analysis engine. For each
   first-two-trick defender lead with both a trump and a fail option, it runs the
   three estimators below from one snapshot of the game + per-seat recurrent
   memory at the decision node, plus an ISMCTS `--explore-sweep` diagnostic.

### Determinism fix (prerequisite)
`simulate_game` constructed its `Game` without threading the request seed, so the
deal was reshuffled from OS entropy on every call — seeds were **not**
reproducible. Fixed in `server/services/analyze.py` (`Game(..., seed=req.seed)`),
which also fixed `/analyze` reproducibility. All results below postdate this fix.

---

## 3. The three-rung ladder (method)

For a defender lead node, with the deal fixed and only the *first card* forced to
differ between branches, we estimate Δ = (trump lead) − (best fail lead):

| Rung | Continuation | Hidden hands | Isolates |
|---|---|---|---|
| **1. true-deal MC** | raw policy, all 5 seats sampled | the **one real deal** | realized outcome incl. deal luck (hindsight) |
| **2. belief-pool MC** | raw policy, all 5 seats sampled | **determinized** worlds from the ISMCTS belief pool (`_build_pool`, sampled ∝ `exp(log_w)`) | EV over the agent's posterior — hindsight removed |
| **3. ISMCTS (by Q)** | **search** continuation for the observer | same determinized belief worlds | value of a search-optimized rest-of-hand |

- **(1)→(2) gap** = hindsight / true-deal luck.
- **(2)→(3) gap** = search-continuation value over the raw policy.
- **Determinization independence:** rungs 2 and 3 share the determinizer, so a
  determinization bug would corrupt *both* in a correlated way. Rung 1 (and the
  single deterministic rollout) need no determinization and are the independent
  anchors. They agree in direction here, so the conclusion does **not** hinge on
  determinization correctness (formal check: `tests/test_ismcts_exit_regression.py`).

### ISMCTS audit-config correction (important)
A `root_explore_frac` × `iters` sweep on two trick-0 cases (seeds 266, 679)
showed that at the **default** `root_explore_frac=0.25`, the policy's high-prior
trump lead hogged visits (≈57% share on ~5 visits to the fail alternative), and
the **visit-count recommendation never matched the value recommendation**
(`top@N ≠ top@Q` at every frac). Flattening the prior (`frac=1.0`) collapsed the
trump's visit share and the recommendation moved off the policy's lead. The Q of
the under-explored fail action *rose* as it finally got visits.

**Consequences applied to the audit search:**
- `root_explore_frac = 1.0` (root prior fully uniform → visits track value, not
  the policy's biased confidence). Note: this only neutralizes the **root** prior;
  the policy still drives opponent play, deeper-node priors, the determinization
  belief weights, the leaf critic, and the observer rollout.
- `iters = 512` (up from 384, to offset the extra variance of uniform root
  exploration).
- **Primary verdict = `top@Q`** (best action by mean action-value among actions
  with ≥1% of visits); `top@N` (visit count) retained for information only.

The belief-pool MC rung is **tuning-free** (no tree prior / PUCT / FPU), so it is
unaffected by this and serves as the cleanest EV estimate.

---

## 4. Test setup (3200-seed run)

| Parameter | Value |
|---|---|
| Model | `final_pfsp_swish_ppo.pt` (30M sister policy) |
| Partner mode | 1 (Called Ace) |
| Seeds scanned | 0–3199 (deterministic, greedy argmax) |
| Trick scope | 0 and 1 (`--max-trick 1`) |
| Defender def. | not picker / not revealed partner / not secret partner / non-leaster; both a trump and a fail lead legal |
| Cases found | **31 TRUMP-PREF** (policy argmax lead is trump); **31 FAIL-PREF** control (argmax is fail), subsampled 1:1 |
| MC rollouts | 50 per branch (true-deal and belief), all seats sampled |
| Belief pool | 512 determinized worlds (`= iters`), sampled ∝ `exp(log_w)` |
| ISMCTS | 512 iters, `root_explore_frac=1.0`, roll-to-terminal early (`d_rollout = 6 − trick`), pure self-play (`seat_policies=None`); verdict by `top@Q`, min-visit 1% |
| Win threshold | defenders win with ≥60 card points |
| Output | `runs/counterfactual_trump_leads_3200_pm1.json` |

Command:
```
uv run python analysis/counterfactual_trump_leads.py \
  --num-seeds 3200 --partner-mode 1 --rollouts 50 --iters 512 \
  --root-explore-frac 1.0 --control-ratio 1.0 \
  --out runs/counterfactual_trump_leads_3200_pm1.json
```

Note: a defender leading trump on tricks 0–1 with a fail option is rare (the
3200-seed scan yielded only 31 such trump-pref states), consistent with the
original "~1.6% of trick-0 defender leads" rarity.

---

## 5. Results

### 5.1 TRUMP-PREF — the behavior under scrutiny (n = 31)

Δ = (trump lead) − (best fail lead); negative ⇒ the trump lead is worse.

| Estimator | Δ defender pts | Δ leader score | Δ win rate | trump better (pts / score / win) | abs EV trump vs fail |
|---|---|---|---|---|---|
| true-deal MC | −4.36 (SE 2.54) | −0.28 (SE 0.17) | −7.0% (SE 4.9) | 48% / 39% / 26% | 55.0 vs 59.3 |
| **belief-pool MC** | **−3.08 (SE 1.03)** | **−0.19 (SE 0.08)** | **−4.5% (SE 1.7)** | 29% / 32% / 23% | 68.3 vs 71.4 (ESS 44.4) |
| single det. rollout | −6.55 (SE 2.83) | −0.45 (SE 0.23) | — | 29% / — / — | — |

**ISMCTS @ 512 it, frac 1.0 (ESS-valid 30/31):**
- **by Q (primary): top is trump 8, fail 22, other 0; agrees with policy's exact lead 4/30.**
- by visits (info): top is trump 12, fail 18.

**By trump count in the leading hand (belief/MC Δ):**

| trump in hand | n | MC Δ pts | MC Δ score |
|---|---|---|---|
| 1 | 4 | **+3.28** | −0.435 |
| 2 | 5 | −1.02 | −0.116 |
| 3 | 11 | **−9.15** | −0.385 |
| 4 | 11 | −3.87 | −0.195 |

**Example states (largest \|MC Δscore\|):**

| seed | trick | trump vs fail | MC Δpts | MC Δscore | trump pts/score | fail pts/score | ISMCTS top@Q |
|---|---|---|---|---|---|---|---|
| 2635 | 1 | AD vs 9C | −29.1 | −2.54 | 60 / +0.42 | 89 / +2.96 | FAIL |
| 266 | 1 | KD vs KS | −29.3 | −2.44 | 48 / −0.56 | 77 / +1.88 | FAIL |
| 3015 | 2 | JD vs KS | −26.1 | −2.28 | 44 / −0.82 | 70 / +1.46 | FAIL |
| 2902 | 2 | 10D vs 9H | −4.9 | −1.74 | 58 / −0.04 | 62 / +1.70 | TRUMP |
| 1199 | 2 | 8D vs 10S | +8.2 | +1.20 | 58 / +0.20 | 50 / −1.00 | FAIL |
| 679 | 1 | 10D vs 10C | −38.3 | −1.14 | 11 / −1.96 | 49 / −0.82 | TRUMP |

(`trick` is 1-based in the table; these are trickIndex 0 and 1.)

### 5.2 FAIL-PREF — control / method check (n = 31)

When the policy's argmax lead is already a fail, Δ = trump − fail should be ≤0.

| Estimator | Δ defender pts | Δ leader score | Δ win rate | abs EV trump vs fail |
|---|---|---|---|---|
| true-deal MC | −9.03 (SE 2.76) | −0.77 (SE 0.24) | −19.6% (SE 7.1) | 43.6 vs 52.7 |
| belief-pool MC | −3.74 (SE 1.04) | −0.20 (SE 0.09) | −4.8% (SE 2.3) | 57.8 vs 61.5 (ESS 72.9) |
| single det. rollout | −11.58 (SE 3.27) | −0.97 (SE 0.33) | — | — |

**ISMCTS @ 512 it, frac 1.0 (ESS-valid 31/31):** by Q: top is trump 5, fail 26;
agrees with policy's exact lead 10/31. by visits: trump 5, fail 26.

The control behaves as required: all rungs Δ≤0 and the search keeps a fail lead in
26/31 — the method is **not** manufacturing a blanket "fail is always better."

---

## 6. Interpretation

1. **Real, mild, systematic.** Belief-pool MC (hindsight-free) shows a small but
   statistically significant loss for the early defender trump lead (Δscore −0.19,
   SE 0.08 ≈ 2.4σ; ~−3 pts; ~−4.5% win). The corrected ISMCTS prefers a fail lead
   in ~73% of these states and the policy's *specific* lead in only 4/30. This is
   a leak, not a blunder — and it matches the original trick-0 magnitude
   (~−0.23 score) on the sister checkpoint.

2. **Per-instance it is mostly deal luck; the *pattern* is the signal.** Flashy
   cases (266: true-deal −29 → belief −1.4; ~break-even) are dominated by
   hindsight. The (1)→(2) aggregate gap is modest (−4.36 → −3.08) because the big
   per-case hindsight swings cancel, leaving a consistent small −EV underneath.

3. **The leak is concentrated in trump-rich hands.** Leading from 3–4 trump is
   clearly worse (−9.15 / −3.87 pts); from 1–2 trump it is fine or even slightly
   +EV. This card-sense gradient is strong evidence the measurement captures real
   strategy rather than a fail-tilted artifact.

4. **Deploy-time search would recover most of it.** With the corrected audit
   config, search picks the better (usually fail) lead in ~73% of cases — i.e. a
   search wrapper at deploy corrects the leak the standalone policy exhibits,
   consistent with the refactor plan's search-teacher thesis.

---

## 7. Caveats and open items

- **Power:** n=31 per group; belief ESS ~44 (trump) / ~73 (control) — peaky. The
  **trump-vs-fail class** verdict is reliable; the *specific* best card is noisy
  (exact-card agreement is only 4/30 trump-pref, 10/31 control). Read class, not
  card.
- **Both groups favor fail**, so the control alone does not exclude a mild
  systematic fail-tilt; the by-trump-count gradient (item 6.3) is the stronger
  evidence against that.
- **Single partner mode / model.** Called-Ace only, on the 30M sister policy.
  Jack-of-Diamonds mode and other checkpoints not yet swept.
- **ISMCTS-as-oracle is bounded by the policy** (priors, belief weights, opponent
  rollouts, critic all come from the same policy). Needing `frac=1.0` to surface
  the better action is itself a symptom of the biased prior; the structural fix
  remains improving the policy (the ExIt/search-teacher path), with deploy-time
  search as a complement.

## 8. Artifacts

- `analysis/scan_defender_trump_leads.py` — case finder (reproducible to `/analyze`).
- `analysis/counterfactual_trump_leads.py` — 3-rung ladder + ISMCTS audit + `--explore-sweep`.
- `runs/counterfactual_trump_leads_3200_pm1.json` — full per-case detail for the run above.
- `runs/defender_trump_leads_pm1.json` — raw trump-lead scan (800-seed, all tricks).
- Fix: `server/services/analyze.py` now threads `req.seed` into `Game` (deal reproducibility).
