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
