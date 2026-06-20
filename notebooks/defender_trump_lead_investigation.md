# Defender Trump-Lead Investigation (final_pfsp_swish_ppo.pt)

Date: 2026-06-17 (leak EV) · 2026-06-18/19 (deploy-time search, §8) · Baseline commit: `64e2b36` · Author: analysis session log

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

## 8. Deploy-time search: does it fix the leak in *realized play*? (June 18–19 2026)

§6.4 claimed deploy-time search "would recover most of it" — but that was an
**EV-ranking over belief worlds** (corrected ISMCTS top@Q prefers fail in 22/30).
It never measured whether *forcing the search's lead and playing the hand out*
improves the defender's **realized game score**. This section tests that directly,
and the answer is more nuanced than §6.4 implied.

### 8.1 Prerequisite: a silent critic-load bug (had to be fixed first)

`final_pfsp_swish_ppo.pt` (the `pfsp-ppo-30M-baseline` arch) **predates the deep
`value_trunk`** in the current `ppo.py`; it trained the value head as
`value = value_head(critic_adapter(x))` with no `value_trunk`. `PPOAgent.load`
loads the critic with `strict=False`, so the new `value_trunk.*` params stay at
**random init** and `forward` bootstraps the ISMCTS leaf on noise. Verified on real
states: the random-trunk value is biased negative, under-dispersed (std 0.105 vs
0.165), and **r = 0.13 — essentially uncorrelated** with the trained value. Only the
critic is affected (the actor loads strict), so policy play is fine; it corrupts only
the ISMCTS critic-bootstrap. The §5 audit dodged it by rolling to terminal early-game
(minimal critic use); a deploy search at `d_rollout=2` would have walked into it.

**Fix (`ppo.py`):** on detecting a checkpoint with no `value_trunk` keys, re-point the
value path through the trained `critic_adapter` (reconstructs the original pathway
exactly) and print a note; warn on any other critic mismatch instead of swallowing it.
**All of §8 uses the corrected critic.**

### 8.2 New tooling

- `analysis/tune_deploy_search.py` — **paired A/B tournament**: a hero seat plays
  search vs the *identical raw policy* on the *same deal* (other seats raw policy in
  both); Δscore = realized strength gain. Sweeps head-set × frac × iters × d_rollout ×
  c_puct × max_depth; selector top@Q or top@visits.
- `analysis/targeted_trump_lead_search.py` — conditions on the §5 leak nodes; per spot,
  paired realized Δ = defender score(search lead) − score(policy lead), deterministic
  argmax continuation on the **true deal**; + FAIL-PREF control. (Measurement is
  bias-free — the search is determinized, but the continuation/scoring is the real deal.)

### 8.3 All-deals pilot (play head) — search is score-neutral on random deals

16 seeds × 5 seats, iters 256, top@Q, frac∈{0,0.1,0.25,0.5} × d_rollout∈{2,6}: **every
config within ~1σ of 0** (best +0.14 ± 0.20). The strong policy + rare/cheap leak
dilute into the noise. Two soft signals: **`d_rollout=2` ≥ `d_rollout=6`** (cheap depth
is fine once the critic is correct) and **frac wants low**.

### 8.4 Targeted at the deploy budget (iters 384) — neutral-to-harmful early

Δ = defender score(search lead) − score(policy lead); paired, true deal. Selector top@Q.

| group | frac | d_rollout | fix% | Δscore (all) | Δscore \| changed |
|---|---|---|---|---|---|
| TRUMP-PREF | 0.1 | 2 | 65 | −0.29 ± 0.28 | −0.45 |
| TRUMP-PREF | 0.1 | term | 97 | +0.03 ± 0.32 | +0.03 |
| TRUMP-PREF | 1.0 | term | 87 | +0.06 ± 0.35 | +0.07 |
| **FAIL-PREF** (control) | 1.0 | 2 | 68 | **−0.52 ± 0.20 (−2.6σ)** | −0.76 |
| **FAIL-PREF** (control) | 0.5 | term | 74 | **−0.68 ± 0.24 (−2.8σ)** | −0.91 |

**No config gives a significant positive Δ on the leak**, and the control is
**significantly negative** — search overrides *correct* fail leads 35–74% of the time
and that costs up to −0.68 (−2.8σ). At the deploy budget, search at trick 0–1 is
neutral-to-harmful: the trained prior is the best estimator at the partial-observability
ceiling, and top@Q overrides it with a noisier signal.

### 8.5 Compute-ceiling + selection-rule test (iters 4096, rolled to terminal) — the edge appears

| group | select | frac | fix% | Δscore (all) | Δscore \| changed |
|---|---|---|---|---|---|
| TRUMP-PREF | q | 1.0 | 87 | +0.32 ± 0.26 (+1.2σ) | +0.37 |
| **TRUMP-PREF** | **visits** | **0.1** | **35** | **+0.23 ± 0.14 (+1.6σ)** | **+0.64** |
| TRUMP-PREF | visits | 1.0 | 84 | +0.32 ± 0.26 (+1.2σ) | +0.38 |
| FAIL-PREF (control) | q | 0.1 | 65 | +0.00 ± 0.26 | +0.00 |
| **FAIL-PREF** (control) | **visits** | **0.1** | **3** | **+0.00 ± 0.00** | +0.00 |
| FAIL-PREF (control) | q | 1.0 | 68 | −0.29 ± 0.24 | −0.43 |

At ~11× the deploy iters rolled to terminal, the **leak Δ turns positive across all four
TRUMP configs** (+0.23…+0.32, ≈1.2–1.6σ) where at 384 it was ~0 — so it was **partly
compute/anchor-limited, not pure determinization bias** (terminal rollouts supply a
low-variance, critic-free value exactly where the critic is blind). The standout is
**`visits` + `frac 0.1`**: it overrides only the **most-confident 35%** of leak nodes
(Δ|chg +0.64; +0.23 overall, tightest SE) and touches just **3%** of correct leads
→ **zero collateral harm**. The control harm at high frac/top@Q (−2.4…−2.8σ at 384)
flips to ~0 — confirming the damage was the **top@Q/high-frac override of a strong
prior**, not search itself.

### 8.6 Confirmation at scale (n = 101, single best config)

The §8.5 wins were n=31 / ~1.6σ — suggestive. Re-run at **9600 seeds** (→ 101
TRUMP-PREF + 101 FAIL-PREF control) at the single best config (`visits`,
`frac 0.1`, 4096 iters, terminal):

| group | fix% | →fail% | Δscore (all) | Δscore \| changed | Δpts | Δwin |
|---|---|---|---|---|---|---|
| **TRUMP-PREF** (leak) | 32 | 25 | **+0.16 ± 0.07 (+2.4σ)** | +0.50 (n=32) | +1.5 | +5% |
| FAIL-PREF (control) | 12 | 12 | −0.01 ± 0.05 | −0.08 (n=12) | −0.2 | 0% |

The leak edge **confirms and is now significant** (+0.16, +2.4σ). The point
estimate regressed from +0.23 (n=31) to +0.16 — textbook regression-to-the-mean
off a small sample — but the tighter SE makes it real. The control is **dead flat**
(−0.01 ± 0.05): the conservative `visits`+low-frac selector overrides only 12% of
*correct* leads and does them no harm. So at scale both halves hold: a real +0.16
game-score/occurrence gain on the leak (+0.50 on the 32% it actually overrides),
with **zero collateral damage**. This is what makes the config a *trustworthy*
signal — the anti-Arm-B — rather than churn.

### 8.7 Synthesis (refines §6.4)

1. **§6.4's "search recovers it" was an EV-ranking, not realized play.** Fail *is*
   marginally better in expectation over belief worlds, but at deploy budgets, forcing
   the search's lead does **not** improve realized score — and overriding correct leads
   actively hurts.
2. **The leak is compute-and-selection-gated, not a no-search ceiling.** Cheap deploy
   (384 iters, `d_rollout=2`, top@Q) = neutral-to-harmful early; **expensive** (4096
   iters, to terminal, `visits` + low frac) = a real, harmless edge (**+0.16, +2.4σ at
   n=101**, §8.6) — but ~70 s/decision, i.e. **offline/analysis-grade, not real-time**.
3. **Why cheap search hurts (and SOTA doesn't):** AlphaZero's "search improves the
   policy" is a *perfect-information* theorem; ISMCTS/PIMC carries strategy-fusion and
   non-locality bias and helps only when the per-world value signal is strong and policy
   headroom is large. At trick 0–1 (max hidden info, near-ceiling policy) those
   conditions fail until you pay for terminal rollouts and defer to the prior via
   visit-count selection.
4. **Deploy posture:** trust the policy at trick 0–1; if a search wrapper is used, prefer
   **visit-count selection + low root-explore-frac** (neutral early, helpful at later
   tricks where the prior work shows +0.103/deal on tactical blunders).
5. **Significance:** the §8.5 grid was n=31 (suggestive); §8.6 confirms the best config
   at n=101 (**+0.16, +2.4σ**, control flat). That config is also a validated candidate
   **clean teacher signal** (real gain on the leak, zero collateral on correct leads — the
   property the ExIt Arm B target lacked).

### 8.8 Implications for the training-time teacher

This investigation **revises the original ISMCTS-soft-target-everywhere plan** (which
Arm B already falsified) and points at a lighter design. Three constraints surfaced:
cheap determinized search is *noisier than the policy* at the trick-0/1 ceiling
(distilling it everywhere injects error); the binding constraint is the early-game
**value** signal, not search width; and conservative confidence-gated selection is what
made search safe.

- **Leading candidate — privileged (oracle) value as the GAE baseline.** Train a value
  `V_oracle(s)` on the *full* state (all hands + blind) and use it as the advantage
  baseline, `A_t = G_t − V_oracle(s_t)`. This is an **unbiased** state baseline (it is
  action-independent given the observation — distinct from action-dependent baselines,
  Tucker et al. 2018) and it **removes the hidden-card variance** from the advantage —
  exactly the variance that, per this project's standing finding, blocks PPO from
  resolving small early-game EV gaps like this leak (Arm A: 50k episodes, leak flat,
  because SNR at the node ≈ 0). Precedent: AlphaStar's value used privileged opponent
  information; asymmetric actor-critic (Pinto et al. 2017); CTDE/centralized-critic MARL.
  Cleanest form is the λ=1 control-variate (high λ to avoid bootstrapping through
  privileged values). **This may fix the leak with no search teacher at all** — it
  attacks the named mechanism directly. *Note:* the point is not to improve `V_obs` (it
  is unused at deploy); it is to use the privileged value as the training baseline.
- **Fallback — cheap oracle-leaf ISMCTS teacher + confidence-gated distillation.** If
  variance reduction alone can't move the *mode*, evaluate ISMCTS leaves with
  `V_oracle` on the determinized (full-info) world — which makes shallow search reliable,
  so the teacher no longer needs the 4096-to-terminal budget — and distill **only where
  the search is confident** (the §8.6 config is the trusted signal). This avoids the
  Arm B floor-mass reinjection.
- **Why it's worth doing despite the tiny EV:** a defender leading trump is a human
  *convention* signalling "I am the partner." A non-partner doing it for −EV emits a
  **false signal** an expert immediately reads — a legibility/credibility failure, not
  just −0.2 score.

Experiment order: **privileged baseline first** (cheapest, most grounded), escalate to
the search teacher only if needed. Validate any retrain with the rigorous-eval gauntlet
(no overall-strength regression) **and** a leak-rate re-scan.

---

## 9. Artifacts

- `analysis/scan_defender_trump_leads.py` — case finder (reproducible to `/analyze`).
- `analysis/counterfactual_trump_leads.py` — 3-rung ladder + ISMCTS audit + `--explore-sweep`.
- `analysis/tune_deploy_search.py` — paired deploy-search strength tournament (§8.2).
- `analysis/targeted_trump_lead_search.py` — realized-play leak assessment + control (§8.2).
- `runs/counterfactual_trump_leads_3200_pm1.json` — full per-case detail (§5 run).
- `runs/defender_trump_leads_pm1.json` — raw trump-lead scan (800-seed, all tricks).
- `runs/targeted_trump_lead_search.json` — §8.4 deploy-budget run (384 iters).
- `runs/targeted_compute_ceiling.json` — §8.5 compute-ceiling + selection run (4096 iters).
- `runs/targeted_confirm_visits_f01.json` — §8.6 confirmation (n=101, best config, 9600 seeds).
- Fix: `server/services/analyze.py` threads `req.seed` into `Game` (deal reproducibility).
- Fix: `ppo.py` `PPOAgent.load` — legacy critic compatibility shim (§8.1); required for any
  ISMCTS work on `final_pfsp_swish_ppo.pt`.
