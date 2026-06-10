# Population-grounded teacher rollouts — implementation plan

Decision context: `Exit_Arms_202606.md`. The self-play teacher (all seats
modeled as pi_theta) certifies information-revealing play (the trump-lead leak)
because its rollout opponents never exploit it, and its Q is too weak/noisy to
starve the exploration floor out of the distill target. Three independent lines
of evidence (teacher audit +26/+28 SE; Arm A clean control; Arm B 9%→74.5% leak
explosion) say play distillation needs opponents that punish mistakes.

## Design

**Ground the teacher in the ACTUAL table, not a sampled abstraction.** At every
training game, `play_population_game` already holds `agents[seat-1]` — the live
seat→PPOAgent mapping (training agent + the 4 sampled population opponents).
The teacher receives that mapping and models each non-observer seat with the
agent that is actually controlling it this episode:

- Observer seat (tree decisions, observer rollout plies, critic bootstrap):
  `pi_theta` — self-modeling your own future decisions is correct, and the
  Q/V units stay the training agent's.
- Non-observer seats (pool-build forced replay + bidding belief weights,
  in-tree "advance" phase, rollouts): the seat's actual controller.
- Partner seat: just another seat with its actual controller (no special case).

This is statistically the most faithful grounding available — teacher Q becomes
"EV of this action against the field I am literally playing right now" — and it
needs no extra model loading, sampling machinery, or staleness policy: the
worker already holds those exact networks in memory for the game itself.

Scheme-B importance weights follow the same principle: the bidding
log-likelihood of a determinized world is computed under the policy that
actually produces those bids (the seat's controller), so belief and behavior
model are consistent.

## API

`ISMCTSTeacher.search(..., seat_policies=None)` — optional `{seat: PPOAgent}`
for non-observer seats. `None` (default) or missing seats → `self.agent`, i.e.
exact current self-play behavior; all existing tests/paths unchanged. The
mapping is per-search transient state (like `_d_rollout_override`), and
`search()` snapshot/restores `_player_memories` for **every distinct agent
involved** (sequential paths touch the controllers' memory dicts; the live
game's opponent memories must survive the search).

Internally a single accessor `self._controller(seat)` replaces direct
`self.agent` use at the seat-policy call sites:

| site | controller |
|---|---|
| `_build_worlds_lockstep` / `_build_world`: acting-seat encode + bidding weight | `controller(seat)` |
| `_after_action_batched` / `_after_action`: end-of-trick observes | per seat `controller(s)` |
| `_run_chunk` step 2-3: batched encode + actor | group requests by controller |
| `_advance_opponents`, `_rollout` non-observer plies | `controller(seat)` |
| `_observer_probs`, observer rollout plies, `_critic_value`, terminal/discount | `self.agent` (unchanged) |

## Throughput analysis

Search cost is ~95% encoder; total encode COUNT is unchanged — each seat-state
was already encoded exactly once, just by a different network now. What changes
is batch grouping:

- **Pool build (lockstep): zero loss.** Each replay decision already encodes
  one acting seat batched across all k worlds; the controller is a function of
  seat only, so the batch stays intact. End-of-trick observes were already
  per-seat loops.
- **Leaf-parallel rounds (`_run_chunk`): bounded fragmentation.** One batched
  call becomes up-to-5 grouped calls (acting seats in flight). Worst case all
  5 controllers present → ~batch/5 per group. Mitigations: (a) the critic head
  now runs only on the observer group's "critic" rows instead of every row
  (small win, was computed-and-discarded); (b) if the benchmark shows >1.5x
  search slowdown, raise `ISMCTSConfig.batch_size` 32→64 so groups stay ≥12.
- Measured before/after benchmark is part of acceptance (same seed, play-head
  search, batch 32): expect ≤1.5x.

## Tests

1. Backward compat: `seat_policies=None` → existing regression suite green
   (covers batched-pool==sequential, search contract, etc.).
2. Controllers reach the search: same root, controllers = differently-seeded
   fresh nets vs None → pi'/root_q differ (seeded, deterministic).
3. Memory isolation: opponents' `_player_memories` byte-identical across a
   search (the live game must be undisturbed).
4. Batched == sequential with controllers: extend the existing lockstep-vs-
   sequential pool check with a seat_policies variant.

## Acceptance

`teacher_trump_lead_audit.py --opponents <4 frozen reference-population ckpts>`:
re-run the n=150 audit on the pristine 30M with population-grounded rollouts.
Pass = root-Q gap (best trump − best fail) widens meaningfully negative and per-
node inversions drop from ~35%; pi' trump mass falls toward the tau-sharpened
range. That result (not the code) is what justifies turning play distillation
back on — next run = Arm C: population-grounded teacher + target_tau from the
B' read + anchor + ramp + guards.

## Out of scope (deliberately)

- Sampling abstractions / opponent pools inside the teacher (the live table is
  better and free).
- Deduplicating the sequential `_simulate` path into the batched one — it is
  the validation reference the batched path is tested against.
- Population-grounding the *determinizer's* deal proposals (hidden-hand
  sampling already honours the public record; the belief WEIGHTS get the
  controller fix, which is the part that matters).
