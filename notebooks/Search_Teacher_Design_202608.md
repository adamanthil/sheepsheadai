# Search Teacher Design (2026-08)

Successor to the Stage-C exit plan's teacher lane, redesigned on the
August evidence base. Companion runbooks: Learning_System_Redesign_202607
(the league run this teacher will graft onto), Convention_Optimality_202607
(E6-E9), Search_Readout_Comparison_202607 (pi_gumbel adoption).

## 1. Why the plan is being refined (evidence summary)

- E7 (fail-lead logit ladder): the low-vs-fat lead ordering is
  ANTI-FORMED — league training built the wrong-side preference from a
  neutral warmstart; entropy steps amplify it.
- E8 (ecology probe): no population effect (contrast −0.002
  [−0.088,+0.078]) — self-play search targets are valid. AND: the
  low-lead edge is ~ZERO under current-policy continuations; it exists
  only under search-improved follow-up. Terminal-reward PG cannot climb
  this (joint-improvement chicken-and-egg); search resolves it inside
  the tree by construction.
- Conclusion (operator, 2026-08-11): search is REQUIRED at these nodes —
  both for noise (small edges under trajectory variance the oracle
  baseline cannot remove) and for ABSENT pressure (edges that only
  materialize under corrected continuations).

The prior teacher design was node-SELECTIVE (defender leads only)
because the 2026-06/07 studies found search harmful at deploy budgets on
non-target nodes. Those studies predate two upgrades:

1. **pi_gumbel readout** (adopted, Search_Readout): completed-Q over the
   UNMIXED prior — the PUCT visit-count target's two harm mechanisms
   (root_explore_frac floor leaking into the target; Q-inversion at ties
   promoted by visit counts) are structurally absent. Near-ties default
   to the policy prior instead of inverting.
2. **Mature critic** (8M-episode league lineage + repaired aux stack):
   leaf bootstraps (d_rollout=2) are much better calibrated than the
   2.8M-era critic the harm numbers were measured against.

Operator directive: reassess WHICH nodes search helps, at WHAT depth,
with HOW MUCH search — aiming for coverage of every node class where
search provides genuine signal advantage, not just selected leads.

## 2. Existing machinery (verified in code 2026-08-11)

`sheepshead/ismcts.py` already supports everything Phase 1 needs:
- `pi_gumbel` emitted on every search (readout-only; selection PUCT).
- `d_rollout` per-call override: N observer play-plies then LIMITED
  critic bootstrap (Stage-B d_short=2 design); large value = terminal.
- Leaf-parallel batching (Tier 1/2 throughput work).
- `seat_policies` population grounding (E8 says: not needed).
- Determinizers for pre-pick and leaster (Stage-C P4) — pick/bury/call
  nodes are searchable (out of Phase-1 scope, noted for Phase 3).

NOT yet implemented: **oracle leaves** — evaluating truncated rollout
leaves with the privileged OracleValueNetwork on the determinized world
(legitimate: the sampled world is fully known inside the search). This
is Phase 2; it should reduce leaf bias at shallow d_rollout and may
move the depth/budget frontier substantially.

## 3. E9: search help/harm matrix (Phase 1; pre-registered 2026-08-11)

Instrument: `sheepshead/analysis/search_help_matrix.py`. Driver (7M
boundary ckpt) replays seeds greedily; play decisions with >=2 legal
actions in standard called-ace games are sampled into cells
(trick 0-4 x role picker/partner/defender x lead/follow) up to a
per-cell quota. Per node, grid = iters {128,384,1024} x d_rollout
{2, terminal}; reference = 4096/terminal pi_gumbel (the offline-grade
instrument from the targeted-search study and E6).

Judgment per node (Qref = reference root Q, a_policy = driver argmax):
headroom = Qref(a_ref) − Qref(a_policy); uplift(cfg) =
Qref(a_cfg) − Qref(a_policy); harm = uplift < −0.01 Q-units.

**Coverage rule (pre-registered):** include a cell in teacher coverage
where mean headroom is materially positive AND some grid config
captures >=60% of it with harm rate <=5%; per covered cell adopt the
CHEAPEST such config. Cells with ~zero headroom are left to PG (search
would spend compute confirming the policy). Expected qualitative
priors, falsifiable by the matrix: headroom concentrated at early
leads (all roles) and early defender follows; late-trick cells ~zero
headroom (bookkeeping-determined); depth=2 adequate late but not early
(deep-rollout-early, Stage-B); 384 sufficient at most cells under
pi_gumbel (the old 384-iter harm was a visit-count-target artifact).

Caveats (registered): reference self-consistency (judging by 4096-iter
root Q assumes the reference ordering; E6's falsifier bounded this),
driver-distribution nodes only, limited-critic leaves (Phase-2 rerun
decides oracle-leaf deltas), single checkpoint (7M), quota-limited n
per cell (screening-grade, not certification-grade — the winning
configs get certified later on bigger n at the covered cells only).

## 4. Phases

- **Phase 1 (now):** E9 matrix -> coverage map + per-cell (iters, depth).
- **Phase 2:** wire oracle leaves into ismcts.py (flagged, default-off,
  regression-gated on the exit-regression test), rerun the frontier
  configs on the covered cells, re-read the depth/budget frontier.
- **Phase 3:** distillation integration on the league trainer
  (confidence-gated pi_gumbel targets at covered-cell nodes, PG-mask
  per Stage-C; node selection now BY CELL CLASS from the E9 map instead
  of hand-picked scanner nodes), plus the pick/bury/call extension of
  the matrix if play-phase results warrant.
- Gate for any trainer graft: operator sign-off + the standing league
  gates (h2h, exploiter, panel) on a branch run before mainline.

**E9 amendment (2026-08-10, pre-results):** driver changed 7M ->
checkpoint_8000000. The extended run was suspended at the gen-8
boundary (Learning_System_Redesign 7.19) and the teacher grafts onto
8M, so headroom must be measured against the graft policy. Recorded
before any full-matrix results were seen (smoke at 7M disclosed; its
per-cell n=1 numbers carry no weight). Full run: quota 8, seeds 0-499.

**Phase-2 trigger rule (2026-08-10, pre-results):** the E9 reference
and headroom map are critic-free (terminal rollouts), so oracle leaves
are NOT a validation requirement — they refine only the d_rollout=2
arms. Rule: if d=2 configs qualify (>=60% capture, <=5% harm) at all
covered cells with limited-critic leaves, SKIP oracle leaves for
coverage (optional quality bump later). If covered cells require
terminal rollouts, wire oracle leaves and rerun ONLY those cells'
d=2 arms against the existing reference — terminal rollouts cost
~10-25x d=2 per search, and the trainer graft's budget hinges on
shallow search qualifying. Oracle leaves are expected to be strongest
exactly there: determinized worlds are full-information, the oracle
critic's native input regime.

**Phase-2 SUPERSEDED by operator directive (2026-08-10):** oracle
leaves are implemented and DEFAULT-ON in ismcts.py (commit 9674e48;
config `leaf_evaluator="oracle"`, silent fallback for agents without
an oracle head, terminal-depth searches unaffected; exit-regression
suite passes 15/15). Rationale: strictly better value estimator in
determinized worlds (the oracle's native full-information regime), so
pre-terminal leaf targets should always use it — the conditional
skip rule above is void. Consequence for E9: the in-flight Phase-1
matrix (launched pre-change, limited-critic process image) becomes
the LEGACY-LEAF baseline; after it completes, the d=2 arms rerun
under oracle leaves (now simply the default) on the same frozen
nodes/reference — a clean leaf-evaluator A/B at zero extra design
cost. Terminal arms and the 4096 reference are identical across both
(never consult a critic), so the two matrices share ground truth.

## 5. E9 Phase-1 results (legacy limited-critic leaves; read 2026-08-11)

Run: driver checkpoint_8000000, seeds 0-499, quota 8 -> 240 nodes, all 30
cells at full quota. Artifact:
`runs/convention_optimality_202607/search_help_matrix_e9.json`. This is
the LEGACY-LEAF baseline (process launched before oracle leaves landed).

Operationalizations decided at read time (disclosed, not pre-registered
numerically): "materially positive" headroom = mean >= 0.010 Q-units
(same epsilon as the harm threshold); config cost order (cheapest
first, iters x est. plies) = 128/2 < 384/2 < 128/term < 1024/2 <
384/term < 1024/term. At n=8, harm rate <= 5% means 0/8 harmed.

**Coverage map (pre-registered rule): 7 of 30 cells covered,**
holding 53% of total mean-headroom mass:

| cell | headroom | config | capture | harm |
|---|---|---|---|---|
| t0-partner-follow | +0.023 | 384/term | 82% | 0% |
| t2-defender-follow | +0.013 | 384/2 | 94% | 0% |
| t2-partner-follow | +0.014 | 128/2 | 73% | 0% |
| t2-picker-follow | +0.026 | 1024/term | 71% | 0% |
| t3-partner-follow | +0.015 | 128/2 | 71% | 0% |
| t3-picker-follow | +0.012 | 128/2 | 100% | 0% |
| t4-picker-lead | +0.039 | 128/2 | 89% | 0% |

Sub-material cells (headroom 0.005-0.010) where a config would qualify:
t1-partner-follow (128/2), t1-partner-lead (384/term), t1-picker-follow
(1024/2), t1-picker-lead (128/2), t3-defender-lead (1024/term),
t4-defender-follow (128/2), t4-partner-lead (384/2) — certification
candidates at larger n.

**Hard cells — material headroom NO grid config captures:**
t1-defender-lead (+0.0101; best arm 384/term captures ~75% of nodes'
agreement but only 0.008 mean uplift at 12% harm — fails the rule) and
t0-defender-lead (+0.0090, no arm >= 60%). These are exactly the
E7/E8 convention-target nodes: consistent with the E8 finding that the
defender-lead edge exists only under improved continuations, budgets
<= 1024 with limited-critic leaves don't resolve it (the 4096/term
reference does). The Phase-2 oracle-leaf rerun is the live test of
whether cheap search becomes adequate there.

Priors vs outcome: "headroom concentrated at early leads" FALSIFIED —
it concentrates at FOLLOWS (t0-partner, t2 all-roles, t3 partner/
picker) plus t4-picker-lead (+0.039, the largest cell — also
falsifying "late-trick ~zero"; likely endgame counting/schmear
decisions). "Depth=2 adequate late but not early" roughly holds: the
covered early cell needs terminal rollouts, mid/late cells go d=2.
"384 sufficient under pi_gumbel" mostly holds (5/7 covered cells at
<= 384 iters). Harm is MILD everywhere (worst mean uplift −0.014;
no June-style catastrophic deploy-budget harm) — consistent with the
pi_gumbel + mature-critic reassessment motivating E9.

Structure: headroom is broad-based (2-6 of 8 nodes per cell nonzero),
not single-node artifacts. Caveats stand: screening-grade n, reference
self-consistency, driver-distribution nodes, single checkpoint.

**Phase-2 launched (2026-08-11):** `--reuse-ref` mode added to the
instrument (commit 08bd0ea) — freezes the 240 nodes and the stored
4096/term reference root-Q, reruns ONLY the d=2 arms under the new
oracle-leaf default with identical per-node RNG (world pools identical;
arms differ only through leaf values). Smoke (5 nodes) matched all
nodes with identical headroom; early signal: oracle 1024/2 captured
reference actions legacy d=2 arms missed. Output:
`runs/convention_optimality_202607/search_help_matrix_e9_oracle.json`.


## 6. E9 Phase-2 results: oracle-leaf A/B (read 2026-08-11)

All 240 frozen nodes matched (no unmatched rows); the operator's ISMCTS
refactor (aa59cd1..5d88c78) was verified behavior-preserving first —
21 ISMCTS tests pass and a critic-free 128/term arm rerun on frozen
nodes BIT-MATCHES the pre-refactor legacy matrix (argmax, uplift, ESS
to float precision), so world-pool identity holds across the A/B.

Oracle leaves changed a d=2 outcome at 114/240 nodes (t0 47, t1 31,
t2 36; t3/t4 identical BY CONSTRUCTION — d=2 reaches terminal there,
no critic consulted). **The effect is budget-dependent** (paired mean
delta-uplift over t0-t2 nodes, harm legacy->oracle):

| arm | mean dUplift | improved/worse | harm |
|---|---|---|---|
| 128/2 | −0.0010 | 40/28 | 10% -> 15% |
| 384/2 | +0.0001 | 37/37 | 11% -> 12% |
| 1024/2 | +0.0010 | 34/21 | 10% -> 8% |

Reading: the oracle's sharp, policy-conditioned leaf values amplify
whatever continuation the tree has built. Under a shallow tree (128)
they mislead more decisively than the noisier limited critic; under a
deeper tree (1024) they help — harm drops and capture rises. This is
the E8 mechanism surfacing inside the search: the leaf cannot supply
the improved continuation, only value it with less noise once the
tree supplies it.

Hard cells: **t1-defender-lead 1024/2 jumps −1% -> 73% capture** but
at 12% harm (1/8 nodes) — near-qualifying, fails the strict <=5%
gate. t0-defender-lead remains uncaptured (24% at 1024/2).
t2-defender-lead (sub-material) turns −404% -> +63%. Coverage under
the operative regime (oracle d=2 arms + terminal arms): same 7 cells,
53% of headroom mass; one config swap — t2-partner-follow moves
128/2 -> 384/2 (oracle 128/2 fails at 51%/12% harm; 384/2 86%/0%).
Since production leaves ARE oracle now, the oracle matrix is the
operative one for d=2 arms; legacy d=2 numbers are historical.

**Verdict: oracle leaves alone do not qualify the convention-target
cells at <=1024 iters / d=2.** They open the high-iters path at t1
and argue depth/budget, not leaf quality, is now binding.

**Depth-ladder probe launched (2026-08-11, pre-registered before
results):** arms {1024/1, 1024/3, 1024/6, 2048/2} (oracle leaves) on
the 48 frozen nodes of the six unresolved-or-terminal-only cells
(t0/t1/t2-defender-lead, t2-picker-lead, t0-partner-follow,
t2-picker-follow), same frozen reference, judged by the same coverage
rule. Rationale: leaf values are policy-conditioned, so the improved
continuation must form in-tree — intermediate d is the midpoint
toward terminal (policy plays the decisive middle tricks, oracle
values the endgame); d=1 tests the pure variance-reduction
hypothesis; 2048/2 tests iterations as the binding constraint.
Non-grid arms support: commit 7f93213. Output:
`search_help_matrix_e9_depth.json`.

## 7. E9 depth-ladder results (read 2026-08-11)

Arms {1024/1, 1024/3, 1024/6, 2048/2} (oracle leaves) on the 48
frozen nodes of the six unresolved-or-terminal-only cells. Semantics
caveat discovered at read time: ``obs_plays`` counts ROLLOUT-phase
observer plays only, so d=6 from trick 0-1 exceeds the remaining
plays — **1024/6 there is a terminal-1024 search under a different
RNG seed**, i.e. an unplanned replicate probe. That replicate is the
most important finding:

**Search-seed noise at defender leads is headroom-sized.** On the
same 8 t0-defender-lead nodes, legacy 1024/term scored −8% capture
@25% harm while its d=6 replicate scored 84% @0% (per-node swings
±0.02 Q on ~0.01 headroom). At near-tie nodes, single-search
capture/harm at n=8 is replicate-noisy; certification there needs
seed-replicate averaging, not just more nodes.

Genuine depth findings (d=1 arms are true shallow-oracle searches):

- **t0-partner-follow: 1024/1 = 87% @0% harm — QUALIFIES**, replacing
  384/term (~3x cheaper).
- **t2-picker-follow: 1024/1 = 63% @0% — QUALIFIES**, replacing
  1024/term (~10x cheaper).
- **t0-defender-lead: 1024/1 = 61% @0% — qualifies by the mechanical
  rule** (newly covered), flagged PROVISIONAL: near-threshold and
  replicate-noisy.
- t1-defender-lead: no arm qualifies (best 1024/6 33%@0%; 2048/2
  57%@12%; oracle-1024/2's 73%@12% stands as best known).
- t2-picker-lead: resistant across all arms (12-25% harm everywhere).
- t2-defender-lead (sub-material): depth arms don't help;
  oracle-1024/2 63%@0% stands.

**Updated coverage map (8 of 30 cells, ~56% of headroom mass) — no
terminal rollouts required anywhere:** t0-partner-follow 1024/1;
t0-defender-lead 1024/1 (provisional); t2-defender-follow 384/2;
t2-partner-follow 384/2; t2-picker-follow 1024/1; t3-partner-follow
128/2; t3-picker-follow 128/2; t4-picker-lead 128/2. All arms are
d<=2-or-1 at <=1024 iters with oracle leaves — the trainer-graft
search budget is cheap-search-only. Unresolved material cells:
t1-defender-lead and t2-picker-lead (~7.5% of headroom mass
combined); both rare in play (~1-2 decisions/game), so even a
reference-grade fallback there would be subsampled, not blanket.

Proposed certification pass (AWAITS OPERATOR SIGN-OFF; est ~3-4h):
expand the three contested cells (t0/t1-defender-lead,
t2-picker-lead) to n~32 nodes with 3 search-seed replicates of the
candidate arms AND the 4096/term reference (replicate-averaged
pi_gumbel), then re-apply the coverage rule at meaningful harm-gate
resolution. Artifact: search_help_matrix_e9_depth.json.

## 8. Certification pass (launched 2026-08-11, pre-registered before results)

Operator approved the §7 proposal. Instrument: certification mode
(commit 9c4aa8a — --cells sampling filter with early stop,
--replicates with disjoint deterministic RNG streams,
--ref-replicates with root-Q averaging). Run: driver 8M, cells
{t0-defender-lead, t1-defender-lead, t2-picker-lead}, quota 24
(fresh deterministic sampling from seed 0, so the original 8
nodes/cell are a subset), arms {1024/1, 1024/2} x 3 replicates,
reference 4096/term x 2 replicates averaged.

Design deltas vs the §7 sketch, chosen for the ~4h budget BEFORE
results (disclosed): n=24/cell not 32; ref replicates 2 not 3
(averaging 2x4096 root-Q ~ sharper than any single search used to
date); dropped 2048/2 (dominated by oracle-1024/2 in both prior
runs: t1 57%@12% vs 73%@12%) and terminal arms (if cheap arms fail
certification, the fallback is subsampled reference-grade labeling
per §7, not a mid-terminal arm). Judgment: a_ref = argmax of the
REPLICATE-AVERAGED reference root-Q (not single-search pi_gumbel);
headroom from averaged Q; per arm-replicate uplift/harm vs that
ground truth — deployment semantics (the trainer runs ONE search),
so capture = E_rep[uplift]/headroom and the harm gate is judged on
24x3 = 72 node-replicate samples per arm per cell (5% now
distinguishable from 12%).

Decision rule (same coverage rule): a contested cell joins coverage
iff some arm captures >=60% of (materially positive) headroom at
<=5% harm; cheapest qualifying arm wins. Cells failing with both
cheap arms are left to PG for the initial graft, with subsampled
reference-grade labeling recorded as the contingency. Output:
`search_help_matrix_e9_cert.json`.

**Gamma amendment (2026-08-11, operator-prompted):** search-time
discounting was silently γ=0.99 on all loaded league checkpoints —
the trainer trains UNDISCOUNTED (--gamma default 1.0, runtime
override) but checkpoints never persisted gamma and the constructor
default resurrected 0.99, which the teacher then applied to
leaf/terminal values against a critic trained on undiscounted
returns. Fixed in 6c08eb7: gamma persists in checkpoints, load
restores it, and the matrix instrument pins driver.gamma = 1.0.
Impact on existing results: hands are FIXED-LENGTH, so from any node
every continuation spans the same observer-ply count and the
discount is near-uniform across root actions — action orderings
(and thus capture/harm/coverage) in the completed matrices stand;
only Q magnitudes carried a ~1-5% uniform shrink at early tricks,
which cancels in capture ratios. The in-flight certification run
(launched pre-fix) runs at 0.99 by the same argument; NOT restarted.
All post-fix runs (and the Phase-3 trainer graft, whose CLI already
defaults to 1.0) use γ=1.

### 8.1 Certification results (read 2026-08-11)

72/72 nodes sampled (quota met at seed ~70; process image predates the
gamma pin, so search ran at 0.99 — ordering-neutral per the fixed-length
argument). **All three contested cells FAIL certification for both
cheap arms** (capture ~0-40% vs required 60%; harm 7-14% vs 5%):

| cell | headroom (cert, n=24, 2-rep ref) | headroom (screen, n=8, 1-rep ref) | 1024/1 | 1024/2 |
|---|---|---|---|---|
| t0-defender-lead | +0.0072 | +0.0090 | +0.001 up, 14% harm | +0.000 up, 14% harm |
| t1-defender-lead | +0.0075 | +0.0101 | +0.002, 12% | +0.003, 10% |
| t2-picker-lead | +0.0049 | +0.0100 | +0.001, 7% | +0.002, 8% |

Decisions per the pre-registered rule: the three cells stay OUT of
coverage (left to PG; subsampled reference-grade labeling remains the
recorded contingency), and t0-defender-lead's PROVISIONAL 1024/1
coverage from the depth ladder is REVOKED. **Final coverage map: 7
cells** (§7 list minus t0-defender-lead), ~53% of screening headroom
mass, all cheap arms.

Two deeper findings from the replicate structure:

1. **Screening headroom was partly reference-noise artifact.** All
   three cells' headroom fell below the material threshold under the
   averaged reference (argmax over a noisy Q inflates apparent gaps).
   And the reference itself is soft here: two independent 4096/term
   searches agree on the best action at only **38/72 (53%)** of these
   nodes. Much of the "uncaptured headroom" at the contested cells
   sits inside the reference's own noise floor.

2. **Replicate agreement is a powerful label gate (calibration for
   the Phase-3 confidence-gate design; operator decision pending).**
   Pooling all single searches at these failed cells: 1024/1 mean
   uplift +0.0010 at 11.1% harm. Gating on 2-of-3 replicate agreement
   with a NON-POLICY action: 22/72 nodes yield a label, mean uplift
   **+0.0112 at 0/22 harm** (95% CI upper ~13%). The same gate on
   1024/2 is weaker (+0.0076, 3/27 harm) — consistent with d=1's
   lower-variance leaves making agreement more evidential. The gate
   abstains on the rest (majority=policy or no majority), which
   distillation tolerates by construction. This directly supports the
   operator-observed per-node-vs-per-cell gap (§9 discussion): the
   contested cells are not unservable — they are unservable by a
   FIXED single-search arm.

### 8.2 Near-equivalence analysis (operator-prompted, 2026-08-11)

Operator observation: many Sheepshead plays are near-equivalent (7C
vs 8C; low fail of either suit), so exact-card agreement understates
search stability. Tested on the certification rows with lead classes
{QUEEN, JACK, TRUMP-PIP, FAIL-0, FAIL-K, FAIL-FAT} (suit ignored for
fail):

- Reference 2-rep self-agreement: 53% exact-card -> **74% class** —
  ~40% of apparent reference instability is equivalence-splitting,
  not genuine ambiguity. The instrument's noise floor is partly a
  card-identity artifact.
- Class-level 2-of-3 gate (1024/1): 23 labels, +0.0077, 1/23 harm —
  looser than the exact-card gate (22 labels, +0.0112, 0/22).
- **Card gate's false abstentions are cheap:** nodes class-consistent
  but card-split (where the exact-card gate wrongly abstains) number
  9/72, and their class-labels average only +0.0015 with 1/9 harm —
  when replicates agree on class but split on cards, the value edge
  is small/noisy and abstention is (empirically) nearly correct
  anyway. The equivalence problem bites the REFERENCE
  interpretation hard, the strict gate barely.

Design implication (pending operator sign-off with the §9 amendment):
keep the strict exact-card 2-of-3 gate for emission; make the TARGET
soft — the replicate-AVERAGED pi_gumbel distribution (root-
parallelization style), which spreads mass over equivalent cards
contextually without a hand-built taxonomy. A distribution-space gate
(fire on averaged-target mass decisively leaving the policy action)
would subsume both gates cleanly; calibrating it needs one arms-only
rerun on the frozen cert nodes with per-replicate pi_gumbel logged
(cheap; not yet run).

## 9. Phase-3 design: the agreement-gated teacher (operator-approved 2026-08-11)

**Amendment to the pre-registered coverage rule.** The per-cell
fixed-config rule (§3) is retired as the teacher's selection
mechanism and retained only as the WHERE-TO-SEARCH filter. Basis:
the per-node/per-cell gap (operator-observed: 51/52 material nodes
captured by SOME cheap arm vs 18 by the adopted per-cell arms — the
union is multiple-comparisons-biased, but certification showed the
gate converts exactly that luck into signal), certification's failure
of fixed cheap arms at the contested cells, and the §8.1-8.2 gate
calibration. Implemented in commit 235c0c3.

**Mechanism** (SearchConfig mode="gated"; pfsp_runtime.
_attach_gated_search_target; --search-teacher on train_league_ppo):

1. WHERE: main-agent PLAY decisions in BOTH partner-selection modes
   (called-ace AND jack-of-diamonds — operator directive 2026-08-11;
   eligibility never filters on partner mode, and play_cell role
   detection is mode-aware via is_secret_partner). Leaster and alone
   games excluded. >= 2 legal actions, node class (play_cell: trick
   x role x lead/follow — same classifier as the E9 instrument, now
   shared code) in ``gate_cells`` = the 23 classes with mean headroom
   >= ~0.003 in E9. Subsampled at ``gate_node_prob`` (default 0.02)
   — the budget knob. CALIBRATION-DOMAIN CAVEAT: the E9 map and gate
   calibration were measured on called-ace deals; JD-mode labels
   extrapolate it. Defensible — the committee mechanism is
   mode-agnostic and abstains at splits — but a JD-mode spot-check
   (E9-style, JD deals, the certified arm) is a recorded follow-up,
   and per-mode emission/agreement telemetry should be watched in
   the branch run.
2. SEARCH: 3 independent-RNG replicates of ONE calibrated arm —
   1024 iters, d_rollout=1, oracle leaves (engine default), gamma=1
   (persisted since 6c08eb7), SELF-PLAY worlds (E8: no ecology
   effect; the calibration searched self-play continuations, so
   population grounding would decalibrate the gate).
3. GATE: emit iff >= 2 of 3 replicates pick the SAME action AND it
   differs from the policy's greedy choice (argmax of the raw root
   prior, read from the search result — no extra forward pass, no
   memory hazard). Otherwise abstain (majority-backs-policy or
   split committee) — the designed common case (~70% at the hardest
   cells).
4. TARGET: the replicate-AVERAGED pi_gumbel distribution,
   renormalized. Soft, so near-equivalent cards share mass
   contextually (§8.2) — no hand-built card-class taxonomy.
5. LOSS: existing Stage-C path — forward-KL distillation toward the
   target on labeled transitions, hard PG-mask there
   (ppo.pg_mask_mix=0.0 default), value loss everywhere.
6. Diagnostics: play-head count/accepted = gate attempts/emissions;
   ess_sum repurposed as summed committee-agreement rate;
   entropy_sum = emitted-target entropy.

Constraints: changing gate_iters, depth, replicate count, or the
agreement threshold voids the E9 certification calibration.
PARALLEL WORKERS SUPPORTED (db87a7d, operator-prompted before
launch): weight payloads carry the oracle head + gamma; workers are
constructed oracle-mode and load both on every refresh
(strict=False, headed/headless tolerant); each worker runs its own
teacher with per-job deterministic RNG. The earlier sequential-only
restriction is retired — at the prior 8-worker collection the
teacher's search cost amortizes across workers instead of serializing
the trainer.

**Literature anchors** (also in the config/code comments): Expert
Iteration (Anthony, Tian & Barber 2017) for the apprentice/expert
loop; AlphaZero (Silver et al. 2017) for soft search-derived policy
targets on on-policy states; DAgger (Ross et al. 2011) for labeling
the learner's own state distribution; Gumbel MuZero (Danihelka et
al. 2022) for the completed-Q readout and its small-simulation
policy-improvement guarantee (a guarantee about the improved
DISTRIBUTION — the basis for soft targets over argmax labels);
MCTS-as-regularized-policy-optimization (Grill et al. 2020);
root parallelization (Chaslot, Winands & van den Herik 2008) for
replicate averaging; query-by-committee (Seung, Opper &
Sompolinsky 1992) for agreement-based label selection. The joint —
committee-gated emission for search distillation at selected node
classes — is not a named technique in the game-RL canon; it is this
program's response to a regime (true edges at or below single-search
noise) that large-budget AlphaZero-family pipelines do not face.

**Branch-run protocol (pre-registered; launch awaits operator):**
resume from checkpoint_8000000 on a branch run-name with
--search-teacher, entropy sidecar held (0.476 pending revisit),
standing gates unchanged (panel, h2h vs seed AND vs gen-8 endpoint
as paired control, duplicate-bridge exploiter). Success = h2h vs
gen-8 endpoint positive at the usual CI with no exploiter
regression; mechanism-level readout = C1/C2 pooled telemetry + the
E7 logit-ladder endpoint (does the low-vs-fat ordering finally
form?). Label telemetry (emission rate, agreement rate) watched for
drift as the policy sharpens — emission collapsing to ~0 is the
gate's built-in retirement, exactly as headroom self-retires.

**Branch-run launch + throughput correction (2026-08-11):** attempt 1
(prob 0.02) measured the gate EXACTLY on-design in its first window —
104 firings / 1,445 episodes (0.072/ep, the predicted rate), 36%
emission (calibration said ~30%), committee agreement 0.85 — but
throughput was 1.0 eps/s vs the 14.7 eps/s gen-8 baseline (~15x, vs
the ~1.1-1.25x estimated pre-launch). Root cause of the estimate
error: single-threaded workers run a 1024-iter search in ~35s (3-4x
the full-thread instrument timing), and each firing paid for 3.
Remediation, applied at episode 1,445 (no checkpoint yet; restarted
clean from 8M): (a) committee EARLY-STOP (6572938) — once an action
reaches gate_agreement picks the outcome is decided, so the remaining
replicate is skipped; identical gate decisions, ~25% less search at
the measured 0.85 agreement (the emitted target averages the
replicates actually run); (b) gate_node_prob 0.02 -> 0.005 (the
designated budget knob; calibration-locked parameters untouched).
Projected: ~2.4 worker-s/ep -> ~3-4 eps/s, ~3 days/gen, ~6k labels
per 1M episodes. Attempt-1 log kept as train_attempt1_prob020.log;
its 1-row telemetry CSV was cleared (fresh run re-creates it).

**Operator re-dial (2026-08-11):** gate_node_prob 0.005 -> 0.01,
accepting ~2x wall time for ~2x label volume ("I'd rather have a
better chance of actual improvement after the full generation").
Restarted at ~ep 3k of attempt 2 (no checkpoint written; clean from
8M; attempt-2 log kept as train_attempt2_prob005.log). Attempt-2
first window had confirmed the correction: 2.6 eps/s in the
startup-burdened window, 0.018 firings/ep on-design, 27% emission,
agreement 0.92 under early-stop. Projected at 0.01: ~2 eps/s
steady-state, ~5-6 days/gen, ~12k labels.

## 10. Attempt-3 entropy runaway and target-form correction (2026-08-12)

**Observed (attempt 3, prob 0.01, soft avg-pi_gumbel targets, coeff
1.0):** Hn_play climbed monotonically 0.52 -> 0.96 within ~25k
episodes and pinned; picker_avg +1.17 -> +0.08 (partial recovery
~+0.7); emission climbed 30% -> ~55% (noisier policy -> more
committee-vs-argmax disagreement -> more labels: a positive feedback
loop). Greedy probe at 8.05M showed ARGMAX damage, not just sampling
noise: partner trump-lead 15% (gen-8 ~100%), t0 trump-lead 37%,
called-suit lead 30%, median play logit spread 0.164 — the play head
was globally FLATTENED by ~800 labels. (Teacher-free gen-9 reference
at the same point: Hn_play ~0.65, conventions intact.)

**Mechanism:** two design errors compounding. (1) At near-tie nodes —
exactly where the gate fires — the completed-Q soft target is close
to uniform, so forward-KL toward it is an entropy-injection term on
states with shared representation (E7: one representation object).
(2) The Stage-C distill loss is a mean over SEARCHED transitions
(coeff 1.0, sized for ~30% search fractions), so its gradient scale
never dilutes with label sparsity — 25 labels/update carried
full-loss-scale gradient.

**Correction (2616ff9, disclosed):** label = smoothed ONE-HOT on the
committee's agreed action (eps=0.05 over other legal actions) — the
exact semantics E9 certification validated (+0.0112 uplift measured
for the agreed ACTION; the soft distribution was never calibrated).
This retracts the §8.2/§9 soft-target rationale: at near-ties
"equivalence mass" and "maximize entropy" are the same thing, and the
one-hot pushes the agreed card up without actively pushing
equivalents down (they lose mass only via normalization).
--search-distill-coeff, new default 0.25. The soft target remains as
gate_target="avg_gumbel" for study.

**Attempt 4 launched** from a clean 8M resume with the branch league
re-copied (attempt 3 had inserted one degraded snapshot member) and
the entropy sidecar restored (play target 0.476). Same budget knobs
(prob 0.01). Watch list: Hn_play vs the ~0.5-0.65 teacher-free band,
emission rate vs ~30%, greedy conventions at 50k probes.

### 10.1 Attempt 4 + teacher-free control: mechanism reattributed (2026-08-12)

Attempt 4 (one-hot targets, coeff 0.25) SLOWED but did not stop the
runaway: Hn_play 0.52 -> 0.82 by 16k episodes (attempt 3: ~0.90 by
13k), emission creeping 33% -> 42%. A ~10x cut in target-entropy
pressure bought only ~1.5x slope — falsifying "soft targets" as the
dominant mechanism. Loss plumbing verified correct by inspection
(mask and distill-mean strictly on labeled rows).

**Teacher-free CONTROL (byte-identical flags/seed/resume, isolated
league copy): Hn_play FLAT at 0.52 through 40k episodes, picker_avg
stable ~1.5.** The teacher path is unambiguously the driver, and the
gen-9 boundary is exonerated. (Also supersedes the earlier
0.65-at-49k comparison from the orchestrator's killed gen-9, which
resumed a mid-evolution sidecar.)

**Reattributed mechanism:** forward-KL distillation toward
committee-agreed actions is INTRINSICALLY entropy-raising while
adoption is incomplete, and its gradient magnitude does not shrink
with one-hot targets: the per-sample KL is ~ -log pi_theta(a*), which
is LARGE precisely because the gate selects disagreement states
(pi_theta(a*) small by construction). Each update pushes mass from
the argmax toward currently-low-probability actions at ~20-50 fresh
states, generalization spreads the flattening through the shared
representation (E7), rising entropy raises emission (the observed
feedback), and the entropy controller has no braking authority
(alpha >= 0; measured-above-target just parks alpha at ~0). One-hot
targets changed the direction of the asymptote, not the transient —
and the coeff cut (4x) accounts for the modest slowing observed.

**Proposed fix for attempt 5 (AWAITS OPERATOR SIGN-OFF — third
design iteration warrants it):** replace forward-KL with a MARGIN
RANKING loss on labeled transitions:

    L = max(0, m + log pi_theta(a_ref) - log pi_theta(a*))

with a_ref = the label-time policy argmax (already the gate's stored
referent) and margin m ~ 0.2-0.5 nats. This teaches exactly the
calibrated claim — "the committee prefers a* over the policy's
choice" — as an ORDERING constraint (E7's deficit is an ordering
deficit), is SELF-LIMITING (loss and gradient vanish once a* beats
a_ref by m; no pressure toward 0.95 mass, no continued flattening),
and leaves the rest of the distribution untouched except via
normalization. PG-mask retained. Literature: margin-based ranking /
large-margin classification (hinge); the self-limiting property is
the standard hinge saturation.

Meanwhile the control runs on as the TEACHER-FREE GEN-9 TWIN — same
seed and boundary as any future attempt, i.e. the ideal paired
control for the §9 gates (better than the gen-8 endpoint alone).

### 10.2 Loss mathematics: forward-KL vs margin ranking (implemented 5170e6f)

Setup: labeled transition with committee action a*, label-time policy
argmax a_ref (stored per transition), legal-action count n, policy
pi_theta with logits z.

**Old (forward-KL toward smoothed one-hot pi', 0.95 on a*, eps/(n-1)
elsewhere):**

    L_KL = sum_a pi'(a) (log pi'(a) - log pi_theta(a))
    dL/dz = pi_theta - pi'        (EVERY logit)

Zero only at pi_theta = pi' exactly. Three consequences:
1. Pressure continues AFTER the policy's argmax flips to a* — until
   pi(a*) reaches 0.95. Choice-agreement is not the stopping point;
   distribution-agreement at an arbitrary confidence profile is.
2. Every other action is pressed toward eps/(n-1): actions below it
   (junk) get pushed UP, near-equivalents above it get pushed DOWN —
   a distribution-reshaping term, entropy-injecting from below.
3. States are effectively unique (never revisited), so no state ever
   completes its trajectory to the profile; what transfers through
   the shared representation is the TRANSIENT gradient direction —
   argmax-suppression plus junk-inflation = global flattening.

Note the emission-level stop (the gate never emits where the policy
argmax already equals a*) is real but insufficient: it operates per
NEW state, while the overshoot happens inside every emitted label.

**New (margin ranking, DQfD form):**

    L_M = max(0, m + log pi_theta(a_ref) - log pi_theta(a*))
    dL/dz (active) = e_{a_ref} - e_{a*}     (softmax terms cancel)
    dL/dz (once log pi(a*) - log pi(a_ref) > m) = 0   exactly

Derivation of the cancellation: d log pi(a)/dz_b = delta_ab - pi(b);
the difference of two such terms kills the -pi(b) parts. So gradient
support is exactly the two logits in question, all other actions are
untouched (before weight coupling), and the loss saturates at the
ordering-plus-margin — pressure expires precisely when the policy
"chooses the search-approved label" with margin m. Verified by
autograd (test_margin_loss_gradient_support_and_saturation) and a
loss-math unit (hinge value + saturation).

Relation to PPO clipping: PPO's clip is a per-update RATE limiter
that re-arms each update (sustained pressure still moves the policy
arbitrarily far, which is how the runaway passed through it); the
hinge is a DESTINATION limiter — the objective itself is satisfied
and shuts off. Both bound what a small sample may demand; only the
hinge encodes a terminal state.

**Reference list (for the eventual writeup):**
- Anthony, Tian & Barber, NeurIPS 2017 — Expert Iteration (the loop).
- Silver et al., Nature 2017 (AlphaGo Zero) — soft search targets on
  on-policy states (dense-label regime).
- Danihelka, Guez, Schrittwieser & Silver, ICLR 2022 (Gumbel MuZero)
  — completed-Q readout; small-budget policy improvement.
- Grill et al., ICML 2020 — MCTS as regularized policy optimization.
- Chaslot, Winands & van den Herik, 2008 — root parallelization
  (replicate averaging).
- Seung, Opper & Sompolinsky, COLT 1992 — query by committee (the
  agreement gate).
- Ross, Gordon & Bagnell, AISTATS 2011 — DAgger (on-policy expert
  labeling).
- Zhang & Cho, AAAI 2017 — SafeDAgger (deviation-triggered queries);
  Menda et al., IROS 2019 — EnsembleDAgger (uncertainty gate);
  Hoque et al., CoRL 2021 — ThriftyDAgger (query budgets): the
  sparse-subset expert-labeling family this design belongs to.
- Schmitt et al., 2018 — Kickstarting (fixed-weight auxiliary distill
  hazard; adaptive weighting).
- Hester et al., AAAI 2018 — DQfD: the large-margin loss for sparse
  expert data beside an RL objective (the direct precedent for §10.2).
- Schrittwieser et al., 2021 — MuZero Reanalyse (partial search
  re-labeling at scale).

Also corrected this session: attempt 3's greedy gate DID fire on
trump-lead and play-spread — the violation prints lacked flush=True
and died in the stdout buffer at kill time (fixed in 5170e6f). June's
guards worked; the visibility bug was ours.

**Default flipped (operator-prompted, same day):** the agent-level
``search_distill_mode`` default is now "margin" — the safe form is
the default; KL is the explicit opt-in for the legacy dense-fraction
path. Made safe by excluding referent-less labeled rows from the
margin loss (fraction-path targets carry no referent) instead of
silently ranking against action index 0.

**Attempt 5 (operator-set, 2026-08-12):** margin ranking loss at
--search-distill-coeff 1.0 (not 0.25): the hinge's structural safety
(bounded two-logit gradient, saturation) carries the stability
burden, and 1.0 gives (a) a clean A/B against attempt 3's KL@1.0
entropy destabilization at matched scale, and (b) the fastest
teacher-learning read per expensive generation. Escalation ladder
inverted: if 1.0 destabilizes, 0.25 is the fallback arm. Same budget
knobs (prob 0.01, m=0.3); entropy tripwires and greedy gates as
before (violation prints now flush).

### 10.3 CORRECTION: attempt 5 launched with an inert teacher — referent stripped in event normalization (2026-08-16)

**Finding (conclusive, reproduced in a deterministic harness):** the
gated teacher stored the margin loss's ranking referent
(`search_ref_action`) on the raw transition dict, but the per-event
whitelist that converts transitions into `episode_events` (the
normalization every training event passes through, both sequential
and worker paths) did not carry the key. Every emitted label reached
the learner referent-less; the §10.2 hardening ("no referent → zero
loss, never rank against action 0") then zeroed its margin term. Net
effect since attempt-5 launch: **zero distillation pressure**, while
the PG-mask still fired on labeled rows — strictly worse than no
teacher (labeled rows lost their PG signal and got nothing back).
Nothing in the run telemetry could have shown this: the 🔍 gate line
reports worker-side firings/labels/agreement, which were all healthy,
and the distill hinge was not logged.

**How it was caught:** while removing the legacy fraction/KL path
(operator directive, this date) the exit-regression distillation
tests were rewired to drive a deterministic always-disagree stub
committee through `play_population_game` into `update()` — the first
coverage of the label pipeline THROUGH event normalization. The
rewired test asserted a positive mean hinge and got exactly 0.0 with
healthy `pg_masked_fraction`, isolating the dropped key. Prior tests
straddled the gap: gate tests asserted the referent pre-normalization,
loss-math tests injected post-normalization tensors directly.

**Why the earlier attempts are unaffected:** attempts 3-4 ran the
forward-KL form, which never used the referent; their runaway
diagnosis (§10-10.2) stands. The margin loss has never actually been
exercised in training until now.

**Interpretation note:** attempt 5's flat entropy (Hn_play ~0.52-0.54
through ~15k episodes) is NOT evidence the margin form is stable — it
is a rediscovery of the teacher-free control result with a PG-mask-only
perturbation. The margin-stability A/B (§10.2) remains unrun.

**Fixes (committed):** `2fe9616` carries `search_ref_action` through
event normalization; the end-to-end smoke asserts the referent
survives, and the rewired regression test asserts hinge > 0 through a
real `update()`. Attempt 5 restarted from `checkpoint_8000000` with
the fixed code as **attempt 5b** (same config: margin m=0.3, coeff
1.0, prob 0.01, 8 workers); the inert ~15k-episode run's log/CSV
preserved as `train_attempt5a_inert.log` / `progress_attempt5a_inert.csv`.

### 11. Legacy fraction/KL search-distill path removed (operator directive, 2026-08-16)

`06b6f7a` deletes the Stage-C/ExIt-era teacher wholesale: the dense
per-head-fraction scheduler (`head_search_fractions`/`t_full`/
`d_short`/`SearchConfig.mode`, `_attach_search_target`, the
fraction-only seat-policies grounding map in the runtime) and the
forward-KL distillation branch (`search_distill_mode`; the stale
`teacher_kl` metric renamed `hinge`). Rationale: both halves are
proven dead ends (fraction coverage: Exit_Arms_202606; forward-KL
under sparse disagreement-selected labels: §10), and the "kl"
compatibility shim had exactly zero production consumers — worse, a
revived fraction run would have silently no-op'd against the
referent-hardened margin loss. Engine-level `seat_policies`
(population-grounded rollouts in `ismcts.py`) is retained for the
analysis probes; the gated committee + margin hinge is now the only
teacher path.

### 10.4 Attempt 5b (margin, coeff 1.0): greedy orderings scramble in ~3 updates — scale, not loss form, is the driver (2026-08-16)

**Run:** relaunched 13:50 with the referent fix (§10.3). Hn_play
0.52 → 0.58 → 0.65 over the first three update windows
(+0.065/window, ≥ attempt 3's KL pace; control flat 0.52). Window
outcome stats stayed clean throughout (picker_avg +1.4–1.6, pick
19–20%) — they lag badly and are NOT a usable tripwire.

**Operator challenge (recorded):** is rising entropy even a problem,
given the teacher deliberately re-weights actions? Watch outcome
indicators instead. Partially sustained: entropy is instrumental, the
0.60 tripwire was KL-calibrated, and the margin form has a real
plateau mechanism (pair-swap satisfaction) that KL lacked. But
arithmetic rules out the benign reading of the magnitude: ~20 labels
per window vs ~9k hero play rows (~0.2%) can move average Hn_play by
~0.002 if fully uniformed; the observed +0.13 is ~60x that, i.e.
almost entirely generalization to unlabeled states.

**Decisive instrument — offline greedy health probe on published
worker weights** (no trainer change; ~1 min per probe; validated on
the untouched 8M checkpoint, same seed 98765, 300 games):

| metric | 8M seed ckpt | 5b @ v4 (~3-4 updates) | control @8.05M |
|---|---|---|---|
| t0 defender trump-lead | 0.7% | **61.5%** | 3.7% |
| partner trump-lead | 97.1% | **41.2%** | 98.4% |
| called-suit lead | 46.8% | 27.5% | 42.7% |
| play logit spread (med) | 3.60 | **1.99** | 3.61 |
| pick / alone / leaster | 37.5/12.9/4.0 | 36.6/17.1/4.7 | 34.4/18.5/8.0 |

Greedy play ORDERINGS scrambled within 3-4 updates — the >8%
trump-lead violation gate would have fired ~8x over — while bidding
heads (PG-owned, unlabeled) stayed intact. Run killed at ~window 4
per the operator's own divergence criterion; artifacts
train_attempt5b_coeff1.log / progress_attempt5b_coeff1.csv; damaged
weights preserved (_league_worker_weights_v4.pt) for forensics.

**Mechanism reattributed (again):** the hinge's per-logit properties
were never the binding constraint. With PG-mask + mean-over-labeled
normalization, each labeled row carries ~1200x the weight of a PG row
at coeff 1.0; each row's gradient is "suppress the label-time argmax"
at disagreement-selected states, and the trunk generalizes that
direction wholesale — same yank family as §10, insensitive to loss
form. The label DIRECTION is certified good (+0.0112/0-harm) but at
this scale the trunk over-amplifies it (61% t0 trump leads vs a
certified-optimal rate far below that). Scale knob and coeff are the
same knob: per-labeled-row weight ≈ coeff x (batch_rows/labeled_rows)
≈ coeff x 1200.

**Open next-arm decision:** (a) pre-registered fallback coeff 0.25
(≈300x/row — only 4x below a scale that scrambled in 3 updates;
value = confirms linear scale dependence, cost ~a day); (b) jump to
coeff ~0.02–0.05 (≈25–60x/row) sized from damage speed; (c) per-row
parity (coeff/1200 ≈ 0.0008, likely inert at ~25 rows/update).
New standing instrument either way: offline greedy probe per weight
publish as the ordering-damage tripwire (entropy alone is neither
necessary nor sufficient).

**Attempt 6 launched (operator-set coeff 0.05, 2026-08-16 ~15:30):**
option (b) from §10.4, skipping the 0.25 attribution arm. Coefficient
semantics reframed for sizing (operator sanity-check on the 1200x
math, recorded): with distill = mean-over-labeled and policy loss =
mean-over-all, coeff IS the teacher's share of the total
actor-gradient budget (per-labeled-row weight ≈ coeff x
batch/labels ≈ coeff x 1000; the normalization is self-adjusting —
sparser emission ⇒ heavier per-label weight, total force constant).
Similar-node PG samples do NOT dilute it: labeled rows are PG-masked,
PG at near-tie nodes is advantage-noise with ~zero realized edge (E8)
so it cancels while the teacher's rows add coherently, and the
entropy controller cannot brake above target. coeff 0.05 ⇒ ~50x/row,
20x below the 3-update-scramble scale; linear extrapolation ⇒ ~60
updates to equivalent damage, vs a per-publish probe cadence of ~1
update. New standing instrument armed: offline greedy probe
(scratchpad probe_weights.py, 300 games, CRN seed 98765) on every
_league_worker_weights_v*.pt publish, alerting at t0-lead > 5% /
partner < 92% / spread < 3.2 (8M baseline: 0.7% / 97.1% / 3.60).
Entropy sidecar restored from the attempt-5a archive copy (5b's final
controller state had adapted to the scrambled policy).

### 10.5 Attempt 6 (coeff 0.05): coefficient scaling is near-inert under Adam (2026-08-16)

Killed after 2-3 updates on a per-publish probe ORDERING ALERT:
t0-lead 0.7% → 0.7% (v2) → **17.7%** (v3), partner 95.3% → 88.7%,
spread 3.73 → 3.24 → 2.54. A 20x coefficient cut bought ~2x slower
damage at best — decisive against per-coefficient scaling as the
safety mechanism. **Mechanism:** the actor optimizer is Adam; for
parameter directions dominated by the teacher's coherent, persistent
gradient, second-moment normalization re-inflates ANY gradient scale
to ~lr-sized steps. Damage therefore tracks the NUMBER of Adam steps
in the coherent direction (epochs x updates), not the coefficient —
retroactively explaining why 5b (coeff 1.0) and 6 (0.05) failed at
almost the same update count. Corollary: only a mechanism that ZEROES
the teacher gradient can bind (Adam cannot step on zero) — loss-scale
knobs of any size cannot. Artifacts: train_attempt6_coeff005.log,
progress_attempt6_coeff005.csv, attempt6_damaged_weights_v3.pt.

Supporting measurement (pre-sizing the trust region): label-time gap
g = log pi(argmax) - log pi(rank-2) at gate-eligible nodes under the
8M policy (2,011 stochastic-play nodes, seed 31337): p25 0.40 /
median 1.25 / p75 2.84 / p90 4.80 nats (rank-3 median 3.14). These
are NOT near-tie nodes on average — each label demands a median
~1.55-nat pair swing (g + m), which the unclipped hinge delivered in
one violent multi-epoch update.

### 12. Adopted loss form: evidence-proportional weight + pair-gap trust region (operator-approved 2026-08-16; commit 518331c)

Design discussion (operator-initiated: "I don't like that we have to
use a hand-tuned coefficient that has baked-in dependencies on our
PPO update size"):

**A. Evidence-proportional weight lambda** (`--search-label-weight`,
default 50): distill = lambda * sum-over-labeled / batch_rows — one
certified label is worth exactly lambda PG samples, the DQfD
per-sample form (Hester et al. 2018 mixed demos into the batch at
natural proportion; our mean-over-labeled was the deviation).
Removes batch-size/emission dependence, and total teacher force now
tracks label count: the gate's self-retirement finally anneals the
teacher in the LOSS, where mean-over-labeled had cancelled it by
re-amplifying the survivors. lambda 50 transfers attempt-6's
operating point (0.05 x ~1000).

**B. Pair-gap trust region delta** (`--search-clip-delta`, default
0.2): a labeled row earns gradient only while the pair gap
log pi(a*) - log pi(a_ref) has improved < delta nats over its stored
label-time value (anchors = the search's mean unmixed root prior,
captured free at emission). The PPO-clip analog (Schulman et al.
2017) and — per §10.5 — the only mechanism in this family that binds
under Adam, because it zeroes the gradient. Each label is consumed in
one on-policy update, so delta is also its lifetime budget; a median
archetype flip (g+m ~ 1.55 nats) accumulates across ~8 labels,
DAgger-style (Ross et al. 2011).

**Design correction discovered by autograd test (recorded):** the
first implementation clamped each LEG to anchor +/- delta. Wrong: the
hinge's two-logit support relies on the pair's softmax terms
CANCELLING; clamping one leg leaves the other leg's full softmax
gradient (e_a - pi over every logit) — suppress-a_ref-and-boost-
everything, precisely the entropy-injection direction of §10. The
adopted form gates the intact pair to zero on gap_gain >= delta,
preserving exact two-logit support while active.

**delta sizing** (operator question: maximize learning per expensive
label subject to stability): candidate anchors were (1) PPO parity
ln(1+eps) ~ 0.2, (2) median-completion (median g + m)/2 ~ 0.78, (3)
budget accounting (teacher total movement K*delta << PG's
N*ln(1+eps) even at generous delta). Pre-§10.5 the recommendation
was (2); the attempt-6 damage speed argues for starting at (1) 0.2 —
the probe gives one-update feedback, so delta can be raised
closed-loop against a healthy ordering probe rather than guessed.
Instrumentation shipped for exactly that: per-window mean emission
gap in the 🔍 line, realized per-label displacement d_star/d_ref +
mean active hinge in distill stats.

**Attempt 7 = this form at lambda 50, delta 0.2, prob 0.01,** same
budget/gate knobs, per-publish ordering probe + log monitors armed.
Success signature: probe flat at baseline, entropy drift bounded,
emission rate decaying (teacher winning at archetypes); failure
signature: gap_gain pinned at delta with probe drift -> lower delta.

### 12.1 Attempt 7 verdict: clip validated, but the expert is non-stationary (2026-08-16, killed at ~6 updates)

Run: λ50 / δ0.2 / prob 0.01, per-publish ordering probe + full gate
telemetry. Killed on a pre-announced stop-rule as t0 defender
trump-lead approached 15% with a linear trend.

**What the clip fixed (validated):** per-update movement pinned at
~δ (Δ*+Δref 0.13-0.32 across windows vs the unclipped ~1.5-2.5 nat
one-shot swings); entropy drift +0.017/window vs 5b's +0.065; damage
rate ~5-10x slower than attempt 6; damage LOCALIZED to taught cells —
untaught conventions held throughout (partner trump-lead 99-100% at
every probe, vs 41-89% collapses in 5b/6). The full mechanism stack
(two-logit hinge, gap gate, evidence weight) behaved exactly as
designed.

**What still failed (the remaining mechanism):** greedy t0 trump-lead
climbed linearly through the certified-optimal band without
inflection (0.7 → 4.5 → 5.3 → 10.5 → 14.1% across probes v2-v6;
healthy band ~4%), spread declined monotonically (3.66 → 2.84), and
the emission gap REBOUNDED at window 4 (2.22 → 1.25 → 0.97 → 1.49)
after an initial self-retirement trend — the disagreement-feedback
loop in slow motion. Diagnosis: **the expert is not stationary.** The
committee searches from the CURRENT policy's priors, rollouts, and
critic; as the policy shifts at taught cells, the committee shifts
with it and finds fresh disagreements at the moved policy, teaching
further in the same direction. The E9 certification (+0.0112/0-harm)
was measured AT the 8M policy — labels leave their certified regime
as soon as the policy drifts, and nothing in the loop re-anchors
them. DAgger's guarantees assume a FIXED expert labeling the
student's states (Ross et al. 2011); ours is ISMCTS seeded by the
student, i.e. the expert chases the student.

**Candidate next amendment (attempt 8, operator decision pending):
freeze the expert.** Pin the teacher's search to the frozen 8M
snapshot — priors, rollout policies, and critic leaves from the
anchor, labels applied to the LIVE policy's states (proper DAgger:
stationary expert, on-student state distribution). The gate's
calibration then remains valid for the whole generation, and the
feedback loop is structurally impossible (the expert's opinion of a
state never moves). Costs: the teacher stops improving with the
student within a generation (fine — E9's certified edge was measured
against exactly this teacher), and worker memory for one extra frozen
agent. Alternatives considered: EV-grounded label audit per window
(expensive, lagging), symmetric confirm-labels as counterweight
(dilutes the calibrated signal), larger m / smaller δ (rate knobs —
§10.5 says destination unchanged). Artifacts:
train_attempt7_lambda50_delta02.log, progress CSV,
attempt7_drifted_weights_v6.pt.

### 12.2 Frozen expert built (5c9819c) + literature positioning + generation-length redesign (2026-08-16)

**Implementation:** teacher wraps a frozen reconstruction of the
generation-start policy (checkpoint + oracle warm-start + gamma) in
both collection paths; weight refreshes touch only the live agent.
Referent + clip anchors moved to the LIVE policy via an act() stash
(a second forward pass would advance recurrent memory): emission
compares the committee to the student's CURRENT argmax — required for
self-retirement (frozen-vs-frozen disagreement never resolves) — and
anchors are truthful label-time log-probs. root_prior no longer
consumed. Full suite 528 green.

**How established systems handle a non-stationary expert (operator
question):** they mostly EMBRACE it, but only because their premises
differ. AlphaGo Zero / AlphaZero / ExIt make search-wrapping-the-
current-network the whole point (policy iteration): safe because
their search is a DENSE, reliable improvement operator — hundreds of
simulations at EVERY state, so drift anywhere is corrected by
supervision everywhere. They additionally smooth non-stationarity
with large replay windows spanning many past iterations, and AlphaGo
Zero used an explicit evaluator gate (new net must win 55% before
generating data) = periodic re-certification. MuZero Reanalyse
re-labels old states with fresh searches — again dense targets +
massive replay averaging. DAgger (our closest frame: sparse expert
labels on the student's state distribution) assumes a FIXED expert;
its descendants (SafeDAgger/EnsembleDAgger/ThriftyDAgger) gate
QUERIES but keep the expert stationary. Our setting violates the
AlphaZero premise twice — deployable-budget search is an improvement
operator only at ~7-23 certified cells (E9: ~11% harm ungated at
near-ties), and labels are sparse one-sided hinges, not dense
distributions — so drift is amplified, not corrected. Freezing per
generation + re-certifying at boundaries = Expert Iteration with the
outer loop at the generation timescale, which is the sound adaptation
of the standard designs to a weak, gated expert.

**Generation length (operator question — agreed 1M is wrong):**
three reasons. (1) Cost: 1M at ~1.5 eps/s = 7-8 days/gen. (2) The
teacher's useful work front-loads: greedy behavior at taught cells
moved within ~5 updates (~7k episodes) in attempt 7, and with a
frozen expert the destination is BOUNDED (the expert's fixed
preferences); once the student adopts them the gate abstains but the
search cost (~90% throughput tax) continues — paying 10x slowdown
for abstentions. (3) Boundary gates should measure a
PG-consolidated policy, not one mid-teaching. PROPOSAL (attempt 8):
two-phase generation — teacher phase ~150-250k episodes with an
adaptive exit (emission rate < ~1/3 of its initial level for 3
consecutive windows = the expert is satisfied wherever it is
confident; hard cap 250k), then teacher-OFF consolidation ~150-250k
at full speed (~14.7 eps/s, ~5 hours) before boundary gates +
re-certification (E9-style spot-check at the new endpoint) + refreeze
for the next generation. Operationally: two standalone launches with
existing flags (teacher run to N, resume with --search-teacher off);
no orchestrator changes needed for the first iteration. Wall-clock
per generation: ~2 days instead of ~8.

**§12.2 amendments (operator, 2026-08-16):** two-phase generation
APPROVED. Correction: full-speed league throughput is ~6 eps/s, not
~14.7 (the 14.7 figure came from a gen-8 baseline log measured under
different conditions) — consolidation 250k ≈ ~12h, full gen ≈ ~2.5
days. Follow-up (build AFTER gen-1 validates the pattern): an
automated generation-phase hook — teacher phase with the adaptive
emission exit (< ~1/3 of initial rate for 3 consecutive windows, hard
cap), automatic teacher-off consolidation relaunch, boundary gates +
E9-style re-certification + expert refreeze — likely as an
orchestrator mode alongside run_extended_league.py. For gen 1 the
phase transition is operated manually off the emission telemetry.

**Attempt 8 phase-1 LAUNCH (2026-08-16):** frozen expert (5c9819c),
clipped margin λ50/δ0.2/m0.3, prob 0.01, --main-episodes 250000
(hard cap; adaptive exit operated manually), 8 workers, per-publish
ordering probe + log monitors armed. Verdict instruments: emission
rate/gap decay (self-retirement under a FIXED expert is now
monotone-convergent by construction), ordering probe vs the
certified band, entropy drift, then consolidation + boundary gates.

### 12.3 Attempt 8 verdict: frozen expert fixes the ordering axis; global spread flattening persists — killed on the pre-registered spread stop-rule (2026-08-16, ~7 updates / ~8k episodes)

**KILLED at v7 (~Ep 8,008k) per the pre-registered stop rule (probe
spread < 2.7).** Seed-98765 spread trajectory across weight publishes:
3.66 → 3.61 → 3.52 → 3.41 → 3.13 → 3.02 → 2.72 (baseline 3.60) —
monotone, with the LARGEST single-step drop last (−0.30). Fresh-seed
replication on v7 (seeds 12345/55555/77777, 300 games each): 2.72 /
2.68 / 2.79 — median on the line, one seed below. Killed rather than
waiting for an unambiguous breach because the trend showed no
deceleration and damage tracks Adam step count.

**What the frozen expert FIXED (hypothesis confirmed):** the attempt-7
divergence axis is gone. t0 trump-lead rose into the certified band
and came back down — 0.7 → 4.8 → 5.6 → 3.2 → 2.5 → 2.1% — vs
attempt 7's linear climb to 14.1% at the same probe count. Partner
trump-lead held 94.9-100% (v6 reading of 94.9 replicated at
94.4/97.1/99.0 = noise, not erosion). Emission gap decayed
(1.80 → 1.42-ish band, oscillating) instead of rebounding. Window
outcome stats stayed healthy throughout: picker_avg +1.32 → +1.58,
pick 19-20%, leaster 5-6%, probe pick-rate 33-36%. The bounded
destination the frozen expert was built for is real.

**What it did NOT fix — the kill mechanism:** global play-head
flattening. The probe spread is a MEDIAN over all play nodes; labels
touch ~1% of eligible nodes in 23 cells, so a 0.9-nat median drop
cannot be direct label gradient — it is generalized softening, the
attempt-5b/6 entropy-injection signature slowed by the clip (7
updates to −0.9 vs scramble-in-2-3) but not stopped. Two reasons it
is NOT self-limiting as built:

1. **Emission never decayed.** Rate held at 39/33/21/41/38/41% —
   self-retirement stalled. With a frozen expert and a moving
   student, steady emission means the live argmax keeps disagreeing
   at fresh (or re-visited) nodes.
2. **Re-anchoring ratchet.** The pair-gap trust region anchors at
   LABEL time. Every re-label of a similar node re-anchors at the
   current policy and grants a fresh δ=0.2. δ caps movement per
   label cycle, not total movement. Steady 40% emission ⇒ unbounded
   cumulative drift budget, and the softmax redistribution en route
   leaks entropy into the trunk via generalization (the per-row
   two-logit support is exact; cross-state generalization is not).

Extrapolation: ~170 more updates in phase 1 at −0.1/update average
(accelerating) ends in a scramble long before the 8.25M cap.

**Standing conclusions for the next design round (NO new attempt
without operator review):**
- Frozen expert: KEEP. It is necessary (attempt 7) and its success
  signature appeared exactly as predicted.
- Pair-gap clip: KEEP but insufficient alone — it needs either
  (a) anchors fixed at GENERATION start (total-movement budget, not
  per-label), or (b) an emission-side cooldown (once a node/cell has
  been labeled N times, stop re-emitting) to break the ratchet.
- The stalled emission rate is itself diagnostic: if the student had
  converged to the expert at taught cells, emission would abstain
  (argmax match). It didn't — plausibly because flattening UNDOES
  earlier teaching at previously-taught nodes (spread collapse ⇒
  argmax instability ⇒ re-disagreement ⇒ re-label ⇒ more flattening:
  the ratchet is a closed loop).
- Operator's outcome-indicator framing held right up to the kill
  (picker_avg rising at v7); the spread probe led every lagging
  indicator by many updates. Spread median = the canonical early
  instrument for this failure mode.

Artifacts: train_attempt8_frozen_expert.log,
attempt8_flattened_weights_v7.pt,
checkpoints/progress_attempt8_frozen_expert.csv. No phase-1
checkpoint was written (first save at 8.05M; killed at ~8.008M) —
next attempt relaunches from the same 8M seed checkpoint.

### 12.4 CORRECTION to §12.3: the flattening is NOT systemic — it is concentrated at defender-lead cells (2026-08-16, per-cell analysis on v7)

**Process note:** the assistant reported this section as committed in
the previous working session before it was actually written; it is
recorded now, after the omission was caught against git log. The
analysis itself was run before the report.

§12.3 attributed the spread decline to "global play-head flattening
via generalization leak." A per-cell measurement (scratchpad
spread_by_cell.py: 300 CRN games seed 98765 DRIVEN by the v7 policy,
both v7 and the 8M baseline evaluated on identical states with
independent recurrent-memory streams, spread bucketed by play_cell)
falsifies the "global" claim:

| cell (sel.)         |    n | base |   v7 | delta | agree% |
|---------------------|------|------|------|-------|--------|
| t0-defender-lead    |  154 | 9.73 | 4.74 | -5.00 |  61.0  |
| t1-defender-lead    |  145 | 9.50 | 3.72 | -5.77 |  81.4  |
| t2-defender-lead    |  120 | 7.37 | 2.81 | -4.56 |  83.3  |
| t3-defender-lead    |  115 | 4.55 | 1.31 | -3.24 |  73.9  |
| def/ptn follows     |    — |  2.4-4.3 |  | -0.9..-1.4 | 69-80 |
| picker cells (all)  |    — |  3.0-7.4 |  | -0.06..+0.33 | 74-96 |
| ALL                 | 5094 | 3.94 | 2.84 | -1.10 |  78.3  |

- **Picker region untouched** (t2-picker-follow delta 0.00;
  t4-picker-lead -0.06 at 95.9% agreement): a trunk-level entropy
  leak cannot be this role-selective. §12.3's "global" mechanism is
  WRONG; the global median moved because defenders hold 3 of 5 seats.
- **The collapse sits exactly on the trained conventions**: defender
  leads had the sharpest convictions in the whole play head (9.5-9.7
  nats) and lost 3-6 nats in ~7 updates.
- **A fight without a stable winner**: 39% of t0-defender-lead
  argmaxes flipped, yet greedy t0 trump-lead is only 2.1% — flips
  scatter across fail cards instead of installing one taught action.
  Consistent with the E9 depth-ladder finding of headroom-sized
  SEARCH-SEED NOISE at defender leads and with certification having
  FAILED/REVOKED every cheap arm at those cells: 2-of-3 exact-card
  agreement occasionally coincides on noise, and each such label
  erodes the incumbent by up to m=0.3 without consistent
  replacement. Non-convergent by construction → emission never
  decayed. The kill stands; the mechanism is sharper than §12.3.
- **Operator framing correction**: earlier monitoring notes called the
  t0 trump-lead rise movement "into the certified band." Wrong: E9
  certified the VALUE of agreed labels (dominated by follow cells),
  never a target trump-lead rate. The human-convention optimum for
  t0 defender trump leads is 0%; any expert-induced rise is
  UNVALIDATED until deep search at a strong convention-adhering
  agent says otherwise (operator, 2026-08-16). The probe's t0>5%
  alert threshold reverts to being treated as a genuine alarm, not a
  "taught direction" tolerance.

### 12.5 Premise correction (operator, 2026-08-16): defender leads are the TARGET, not a nuisance — instrument redesign proposal

The assistant proposed removing defender-lead cells from gate_cells.
**REJECTED by operator**: the impetus for the search-teacher path is
a convention-adhering agent under terminal-only rewards — PARTICULARLY
at defender leads, where PG obeyed the called-suit lead convention at
only ~30% on the first two tricks (and falling as entropy dropped).
Excluding those nodes optimizes the instrument at the expense of the
mission. Proposal retracted.

Redesign (PENDING operator decision; nothing built):

1. **Split-arm gate — heavy committee at lead cells only.** Defender
   leads are rare (~0.5/game) and gate-subsampled, so 4096-iter /
   terminal-rollout committees at just those cells are affordable in
   a way they never were cell-wide; follows keep the certified
   1024/1. Rationale: E9 "edge-only-under-improved-continuations" +
   June targeted study (edge appears only at 4096-to-terminal). The
   frozen 8M expert supplies better continuations (conventions
   84-98%) than the 30M agent the June study used.
2. **Class-level agreement at lead cells.** The decision is
   categorical (called-suit fail vs other fail vs trump) and the
   convention is defined at that granularity. Require 3-replicate
   agreement on the CLASS, emit the highest-mean-Q card within the
   agreed class, abstain when the live argmax is already in-class.
   Noise-robust where exact-card agreement demonstrably is not;
   still terminal-grounded — the convention must win on outcomes to
   be emitted.
3. **Gating study (doubles as the convention-optimality probe).**
   Before any relaunch: harvest ~150-200 t0-t2 defender-lead nodes
   from 8M self-play; run the heavy committee TWICE with independent
   seeds; measure replicate stability, class-agreement rate, and
   label content vs conventions. Outcomes: (a) class-stable labels →
   relaunch with split-arm gate, teacher installs the convention
   with certified pressure at the ~30%-adherence nodes that motivated
   the program; also directly answers whether the optimal t0
   trump-lead rate is 0% or slightly above (deep search AT a strong
   convention-adhering agent = the operator's stated standard of
   evidence). (b) Labels still scatter → the optimum at these nodes
   is genuinely flat at this playing strength; convention adherence
   then needs tie-breaking pressure, not value pressure — a design
   fork for the operator, and a result worth having before spending
   more teacher compute.
