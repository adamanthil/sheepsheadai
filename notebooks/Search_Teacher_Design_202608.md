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
