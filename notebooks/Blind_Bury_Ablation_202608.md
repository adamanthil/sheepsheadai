# Blind/Bury Observation Ablation — 2026-08-30

Status: **MEASURED. The tokens are load-bearing at zero fine-tuning — the
drop is NOT free, and the v2 lineage leans on them ~6× harder than prod 30M.**

## 1. Motivation

A founding constraint of this project is that the agent plays under
*human* restrictions of both state observation and state recall: no
imperative bookkeeping, no access to the full state history at each
decision. Play history satisfies this — the network stores it in the GRU
memory. But the picker does not: `game.py` re-injects the picker's
blind and bury cards into **every** observation (`blind_ids`/`bury_ids`,
populated only when `is_picker`; PAD for every other seat). A human picker
saw the blind once and remembers it; our picker gets it re-shown each step.
That is a (minor) violation of the recall constraint.

The eventual fix is to remove the two keys from the observation dict and
the four tokens from the encoder layout (19 → 15), registering a new
architecture. Before committing to a retrain, the operator asked for the
cheap gate: **how much EV do the current trained nets actually lose if the
tokens are masked with no retraining at all?** That bounds how much
information the tokens carry beyond what the memory already holds.

## 2. Design

Instrument: duplicate-bridge h2h (the `h2h_duplicate` design — candidate in
all 5 seats per CRN deal, both partner modes, deal seed schedule 42), with
one twist: the hero plays the **same weights** as the field, but with
`blind_ids`/`bury_ids` PAD-masked at the encoder input (pure input
ablation; PAD produces all-False masks, the tokens drop out of attention
and pooling exactly as they already do for non-picker seats — no weights
touched, checkpoints load unchanged).

The masking gives the instrument an unusually sharp null:

* pre-pick observations are identical under the mask (blind/bury are PAD
  until the actor becomes picker), so the pick sequence — and therefore
  role assignment — is identical in both arms;
* play is deterministic (argmax) and the field is unablated;
* therefore the two arms are **bit-identical on every hand where the hero
  does not pick**. The paired per-hand diff is nonzero only on hero-picker
  hands, where the ablation binds: the bury decision, the called-card
  decision, and all picker play.

The ablated arm is replayed only on hero-picker hands (~19% of cells).
`--verify-identity` pilots (25 and 20 deals/mode) replayed *every* hand for
both checkpoints and asserted the zero null and role identity — both
passed, for both architectures.

Tooling: `sheepshead/analysis/blind_bury_ablation.py` (committed alongside
this note). Bootstrap is deal-clustered per `analysis/bootstrap.py`
conventions; picker-conditional means use a deal-clustered ratio bootstrap.

Subjects, 1000 deals/mode each (2000 CRN deals, 10k hero cells):

* **prod 30M** — `final_pfsp_swish_ppo.pt` (full architecture, strongest
  deployed artifact);
* **v2 8M seed** — `runs/league_retention_pg/checkpoints/`
  `pfsp_perceiver-shared-v2_checkpoint_8000000.pt` (the certified seed of
  the current teaching program, and the lineage any new architecture
  derives from).

Artifacts: `runs/blind_bury_ablation_202608/{prod30m,v2seed8m}_full.{json,log}`.

## 3. Results

Score units are per-hand zero-sum payoff. "Edge" is the overall duplicate
edge (ablated − baseline, per-deal seat-mean; this equals pick-rate ×
picker-hand cost by construction). CIs are 95% bootstrap.

| | prod 30M | v2 8M seed |
|---|---|---|
| overall edge | **−0.054** [−0.078, −0.029] | **−0.360** [−0.402, −0.319] |
| picker-hand EV diff | **−0.300** [−0.446, −0.156] | **−1.900** [−2.121, −1.680] |
| picker hands, changed outcome | 573/1796 (32%) | 941/1895 (50%) |
| hero-picker rate | 0.180 | 0.190 |
| leaster rate (cells) | 0.102 | 0.052 |

Per mode:

| | called | jd |
|---|---|---|
| prod 30M edge | −0.043 (se 0.019) | −0.065 (se 0.018) |
| prod 30M picker diff | −0.245 [−0.458, −0.030] | −0.351 [−0.542, −0.165] |
| v2 seed edge | −0.474 (se 0.033) | −0.246 (se 0.028) |
| v2 seed picker diff | −2.456 [−2.793, −2.123] | −1.322 [−1.614, −1.038] |

## 4. Reading

1. **The tokens carry real information the memory is not currently
   substituting for.** Even prod 30M — whose picker is far stronger —
   loses ~0.30 per picker hand (4σ from zero), with a third of picker
   hands changing outcome. This is not "realistically not much
   information" at the behavioral level; the *content* is small (4 cards
   the picker already saw) but the current policies genuinely consume the
   re-injection.
2. **The v2 seed is ~6× more dependent** (−1.9 per picker hand; half of
   picker hands change outcome; the called-ace mode, where bury/call
   interact with partner choice, is nearly twice as bad as jd). A picker
   hand is worth roughly +1–2 on average, so −1.9 means ablated v2
   picker play is close to throwing the hand. Plausible reading: 8M
   episodes vs 30M, and the perceiver readout, leave the v2 policy more
   reliant on the always-present shortcut; nothing ever pressured either
   net to route blind/bury through the GRU, because the shortcut was
   always there.
3. **Implication for the migration path.** The "drop the keys and
   fine-tune briefly" path is *not* a formality — the warm-start begins
   from a picker that has lost 0.3–1.9 per picked hand. The information
   is fully recoverable in-episode (the blind passes through `hand_ids`
   during the bury phase; the bury is the picker's own action), and the
   GRU demonstrably carries comparable history, so fine-tuning has a
   clear mechanism to close the gap — but it must actually be given the
   episodes to relearn the routing, and picker EV must be gated, not
   assumed. A from-scratch run remains the only clean version of the
   architectural claim ("trains to this level under human recall
   constraints"); the warm-start is the cheap feasibility rung, not the
   proof.

## 5. Planned change (not yet built)

When the change is made:

* register a new architecture, **perceiver-recall** (operator-chosen
  2026-08-30 — names the constraint it enforces), deriving from
  perceiver-shared-v2;
* remove `blind_ids`/`bury_ids` from `get_state_dict` (keep them in the
  oracle path — the privileged critic is CTDE and exempt), the token
  layout (19 → 15; `token_layout.py`, oracle encoder's 51-token layout,
  `compiled_encoder`), and the encoder's simple-bag machinery;
* transformer/readout/GRU weights are sequence-length-agnostic and
  masked keys contribute nothing to attention, so a 15-token forward
  pass is numerically identical to today's masked pass — the existing
  checkpoints remain loadable for warm-start experiments;
* arch goldens re-capture required (`capture_arch_goldens`), and the
  environment-stamp/skip-guard rules apply.

## 6. Open follow-ups

* Localize the v2 loss: is it the bury/call decisions or picker play?
  (Cheap variant: mask only after `play_started`.)
* Warm-start fine-tune feasibility run: how many episodes until picker
  EV recovers to parity on this instrument?
* Re-run this probe on the distill program's current artifact (e.g.
  `interp_a50`) before any migration decision, since that lineage will be
  the actual seed.
