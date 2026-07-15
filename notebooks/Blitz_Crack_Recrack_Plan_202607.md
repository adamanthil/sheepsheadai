# Blitzing / Cracking / Recracking — Design & Training Plan (July 2026)

**Status (2026-07-15): PLAN ONLY — no code written. Blocked behind Stage 1
of the architecture ablation (winner determines which encoder lineage gets
the surgery). Everything below is implementable by a future session without
further design work; decisions that were judgment calls are marked D1–D6
with their rationale.**

## TL;DR — what this is and how to use it

Add three optional house rules from
[sheepshead.org — special situations & house rules](https://www.sheepshead.org/rules/special-situations-house-rules/)
to `sheepshead/game.py` and teach them to the already-trained PPO agents
**without retraining from scratch**:

* **Blitzing** (black and/or red): a player holding both queens of a color
  may declare before the pick round; stakes double and the two queens are
  publicly revealed.
* **Cracking**: after the picker is established (post-bury/call here, see
  D2), a non-picker may double the stakes.
* **Recracking**: after a crack, the picker may double again (picker-only
  in v1, see D3).

Each rule is an independent `Game` constructor flag, exactly like
`partner_selection_mode` / `double_on_the_bump`. Rules-off games must stay
**bit-identical** to today's engine and today's checkpoints.

The migration method is **function-preserving model surgery** (OpenAI Five
"surgery", Net2Net; references §R): widen input layers with zero-initialized
columns, add fresh heads for the new decision points, then fine-tune on a
mixed rules-off/rules-on episode distribution with a reduced trunk LR.
Rules-off forward passes are bit-identical after surgery, so the existing
golden gate verifies the transplant.

Implementation order: **Part 1 (engine) → Part 2 (obs/encoder) → Part 3
(heads) → Part 4 (surgery) → Part 5 (training)**. Each part ends with its
verification gate. Do not start Part 5 until Parts 1–4 gates all pass.

---

## Rule mechanics and design decisions

All three rules are **pure stake multipliers plus a few binary pre-play
decisions**. Trick play is completely untouched. Final stake multiplier:

```
stake_mult = 2 ** (blitzed_black + blitzed_red + cracked + recracked)   # 1..16
```

applied multiplicatively on top of the existing score (including
`double_on_the_bump`, which stacks — multiplication commutes, order
irrelevant).

Decisions (numbered for later reference; if a future session must deviate,
add a dated amendment here rather than editing in place):

* **D1 — Blitz window = one pass, seats 1..5, before the pick round.**
  The source rule also allows non-pickers to blitz after the pick; v1 uses a
  single pre-pick window to keep the phase machine simple. Only seats
  actually holding a qualifying pair get a decision node (others are
  auto-skipped by the engine — no no-op actions polluting rollouts). A
  player holding all four queens gets two consecutive nodes: black first,
  then red. Post-pick blitzing is a listed extension, not v1.
* **D2 — Crack window = after bury/call complete, immediately before the
  first lead.** The source says "after someone picks... before play
  begins", which is a window, not a point. We pick the latest point in that
  window (defenders know the called card — richer decision, single clean
  insertion point in the phase machine, right where `play_started` flips).
* **D3 — Crack eligibility = every non-picker seat, including the secret
  partner; at most ONE crack per game; recrack = picker only.** Partner
  eligibility is rule-faithful ("non-pickers") and avoids the engine
  consulting hidden partner identity for masks; the agent will learn not to
  crack its own team (and can exploit the bluff if it ever pays).
  Single-crack keeps the doubling tree finite and matches the common house
  form. Partner-recrack (which publicly reveals partnership — a real
  strategic option) is a listed extension, not v1.
* **D4 — Blitz multipliers apply to leaster.** A pre-pick blitz risks
  landing in a leaster; letting the double ride keeps the blitz EV
  calculation honest (blitzing with a weak-side hand is properly punished).
  Cracking cannot occur in leaster (no picker).
* **D5 — Crack round order = seat order 1..5 skipping the picker; first
  CRACK ends the round.** Later seats that never got to act are simply
  never asked (unobservable to the net, no leakage: observers only see who
  cracked, and mask-eligibility is private).
* **D6 — Reward carries the multiplier.** The multiplied score IS the
  training reward. A multiplier that only conditions the critic gives the
  actor no reason to ever blitz/crack. Variance consequences and
  mitigations in Part 5.

Frequency sanity (why this is learnable at all): P(a given player holds
both queens of a color) = (6/32)(5/31) ≈ 3.0%, so ≈ 14% of deals contain a
black-blitz node and likewise red (≈ 27% either); every picker game
contains 1–4 crack nodes. Signal is terminal, outcome-grounded, and
one-step — far better SNR than the play conventions studied in
`Convention_Optimality_202607.md`.

---

## Part 1 — Engine (`sheepshead/game.py`)

### 1.1 Actions — APPEND ONLY

`ACTIONS` currently has 110 entries (ids 1..110: 4 base + 9 CALL + 32 UNDER
+ 32 BURY + 33 PLAY). **Append the seven new actions at the END** so every
existing action id is preserved (ids are baked into trained heads, replay
buffers, and analysis tooling):

```python
BLITZ_ACTIONS = ["BLITZ BLACK", "BLITZ RED", "NO BLITZ"]      # ids 111-113
CRACK_ACTIONS = ["CRACK", "NO CRACK"]                          # ids 114-115
RECRACK_ACTIONS = ["RECRACK", "NO RECRACK"]                    # ids 116-117
ACTIONS.extend(BLITZ_ACTIONS + CRACK_ACTIONS + RECRACK_ACTIONS)
BLACK_QUEENS = ["QC", "QS"]
RED_QUEENS = ["QH", "QD"]
```

### 1.2 Config and state

`Game.__init__` gains keyword args (all default **False** — a default
`Game()` is byte-identical to today, which the regression test in 1.6
enforces):

```python
blitzing=False, cracking=False, recracking=False
# assert not recracking or cracking
```

New state:

```python
self.blitz_black = 0        # seat that blitzed black, 0 = none
self.blitz_red = 0
self.cracked_by = 0         # seat that cracked, 0 = none
self.recracked = False
self.blitz_turn = 1 if blitzing else 0   # 0 = blitz phase over
self.crack_turn = 0         # 0 = not in crack phase
```

Helper properties:

```python
@property
def stake_multiplier(self):
    return 2 ** (bool(self.blitz_black) + bool(self.blitz_red)
                 + bool(self.cracked_by) + self.recracked)

def _advance_blitz_turn(self):
    # skip seats with no undecided qualifying pair; set 0 when past seat 5
```

A seat "qualifies" for color X if it holds both queens of X **and** has not
yet decided that color. `__init__` must call `_advance_blitz_turn()` once
so games with no qualifying holder skip the phase entirely.

### 1.3 Phase logic (`Player.get_valid_actions`, game.py:1115)

Insert two gates, in this order, keeping every existing branch untouched:

1. **Blitz phase** (top of the function): if `self.game.blitz_turn` is my
   seat → return the qualifying subset of
   `{"BLITZ BLACK", "BLITZ RED", "NO BLITZ"}` (black offered before red if
   both qualify: offer only black's node first). If `blitz_turn` is another
   seat → return `set()`. The existing pick/pass branch is only reachable
   once `blitz_turn == 0`.
2. **Crack phase**: entered when the bury completes (or `ALONE`+bury path
   completes) — i.e. at the two sites in `act()` that set
   `play_started = True` for picker games (game.py:1274), set
   `crack_turn = 1` instead if `cracking` is on, and only set
   `play_started` when the crack phase resolves. While `crack_turn` is my
   seat and I am not the picker → `{"CRACK", "NO CRACK"}`; picker seats are
   skipped by the advance helper. After a `CRACK`: if `recracking` →
   picker's node `{"RECRACK", "NO RECRACK"}` (represent as
   `crack_turn = -1` meaning "picker recrack pending"); else phase over.
   Phase over → `play_started = True`, leader/leaders exactly as the
   current code does.

Leaster entry (5th PASS, game.py:1262) is unchanged — no crack phase.

### 1.4 Action handlers (`Player.act`, game.py:1249)

```python
if action == "BLITZ BLACK":  self.game.blitz_black = self.position; advance
if action == "BLITZ RED":    self.game.blitz_red = self.position; advance
if action == "NO BLITZ":     mark this seat/color decided; advance
if action == "CRACK":        self.game.cracked_by = self.position; → recrack node or end phase
if action == "NO CRACK":     advance crack_turn
if action == "RECRACK":      self.game.recracked = True; end phase
if action == "NO RECRACK":   end phase
```

"Advance" = the `_advance_*_turn` helpers; ending the crack phase sets
`play_started = True` and the trick-0 leader fields.

### 1.5 Scoring (`Player.get_score`, game.py:1391)

* Picker games: multiply the final per-player return by
  `self.game.stake_multiplier` (after the existing `double_on_the_bump`
  doubling). Zero-sum is preserved (a common scale factor).
* Leaster: multiply the existing `4 / -1` payouts by the blitz-only part of
  the multiplier: `2 ** (bool(blitz_black) + bool(blitz_red))` (D4).

### 1.6 Ancillary engine surfaces

* **Revealed cards**: blitzed queens are public from declaration until
  played. No new engine state needed — derivable from
  `blitz_black/blitz_red` + hands — but `get_state_dict` (Part 2) and
  `sample_determinization` need it: **determinizers must pin the revealed,
  not-yet-played queens to the blitzer's seat** (touch
  `_sample_deal_attempt` / `_determinization_context`, game.py:580–735).
  ISMCTS correctness depends on this.
* **`scripted_agent.py`**: must handle the new decision nodes. Default
  heuristic: never blitz-blind... no — scripted default = always `NO
  BLITZ` / `NO CRACK` / `NO RECRACK` (keeps it a stable sanity floor);
  add optional aggressive variants later as exploit probes.
* **Regression test (gate for Part 1)**: seeded `Game(seed=k)` with flags
  off must produce byte-identical action masks, state dicts, histories, and
  scores versus the pre-change engine for a few hundred seeds (record
  digests before changing anything). Plus unit tests: phase transitions,
  auto-skip (no decision node for non-qualifying seats), single-crack,
  zero-sum with multipliers, D4 leaster payout, all-four-queens double
  node.

---

## Part 2 — Observation & encoder

### 2.1 `get_state_dict` additions (game.py:958)

Append (never reorder existing keys):

```python
"blitzing": uint8, "cracking": uint8, "recracking": uint8,   # rule config
"blitz_black_rel": uint8,   # rel seat 1..5 of black blitzer, 0 = none
"blitz_red_rel": uint8,
"cracker_rel": uint8,
"recracked": uint8,
"stake_mult_log2": uint8,   # 0..4
"revealed_card_ids": (4,) uint8,   # blitzed queens not yet played, 0-padded
"revealed_seat_rel": (4,) uint8,   # owner rel seat per slot, 0 = pad
"revealed_is_picker": (4,) uint8,
"revealed_is_partner_known": (4,) uint8,
```

Slot order: black pair then red pair; a queen leaves the array once played
(the trick tokens + GRU memory take over from there). All-zero when rules
are off or nothing declared — this is what makes zero-init surgery
function-preserving. `get_oracle_state_dict` inherits everything via its
`get_state_dict` call; it needs nothing extra (the oracle already sees all
hands — the *header* fields are what matter to it, esp. `stake_mult_log2`,
which changes the value target).

### 2.2 Encoder changes (`sheepshead/agent/encoder.py`)

The token-construction stage is shared by ALL architectures (full,
perceiver*, shared-readout), so each change is made once:

1. **Header** (encoder.py:505): append the 8 new scalar fields AFTER the
   existing 10 (append-order = surgery-order). Extend the norm vector
   (encoder.py:522) with `[1, 1, 1, 5, 5, 5, 1, 4]`. `context_mlp` input
   becomes `Linear(18 + d_card, d_token)`.
2. **Revealed-cards bag**: build ≤4 tokens shaped exactly like trick
   tokens — `card_emb(revealed_card_ids) + seat_emb(revealed_seat_rel) +
   role_emb(from is_picker/is_partner_known)` — **through
   `token_mlp_trick`** (weight sharing is the right prior: both token kinds
   mean "card at a known seat", and it avoids a new module). Mask = card id
   != 0. New `card_type` id **6** ("revealed"): expand the embedding table
   6 → 7 (encoder.py:242).
3. **Append the bag at the END of `all_tokens`** (after bury, positions
   19:23) and extend `all_mask`/`type_ids` accordingly. Every existing
   consumer slices fixed leading indices (context 0, memory 1, hand 2:10,
   trick 10:15, blind/bury 15:19 — encoder.py:635-639, encoders.py:196/
   233/324), so appending breaks nothing. Update the stale `(B, 19)`
   comments/docstrings (encoder.py:39, encoders.py:84, SharedReadout
   docstring).
4. **Per-seat declaration flags at the consumption site**: widen the
   trick-token input (`token_mlp_trick`, encoder.py:265) with 3 flag
   columns appended after role — `is_cracker`, `is_blitzer_black`,
   `is_blitzer_red` — derived IN THE ENCODER from the header (trick slots
   are in rel-seat order 1..5, so slot r's flag is `r == cracker_rel`
   etc.; same derivation for the revealed bag and, via `*_rel == 1`, for
   the hand tokens' 3 new "I am the cracker/blitzer" columns appended to
   `token_mlp_hand`'s input, encoder.py:261). No new obs fields needed for
   these.

Why this shape (and not one-hots in the context token alone): revealed
queens are card-located facts; encoding them as card tokens reuses the
shared card-embedding table so all existing card-to-card attention
(dominance, trump counting) applies immediately, instead of the net having
to learn from reward that "flag k entails QC+QS at seat 3". Header scalars
alone would also sit two attention hops from the hand tokens — the exact
multi-hop-inference weakness documented in
`defender_trump_lead_investigation.md`. Header + token flags + card tokens
is deliberately redundant; OpenAI Five / AlphaStar observation design is
redundantly encoded on purpose (§R).

### 2.3 Per-architecture notes

* **perceiver-shared-v2 / all perceiver variants**: nothing else. The
  shared readout and the per-network readouts attend over `all_tokens`
  with `key_padding_mask` (encoders.py:329-335); MHA is length-agnostic,
  so the new tokens are visible to actor and critic with zero new
  parameters. Empty bag = fully masked = bit-identical rules-off pass.
* **full**: information reaches the fused features indirectly (revealed
  tokens reshape hand/trick/context tokens via self-attention before
  pooling). Optionally add a dedicated `AttentionPool` for the bag +
  zero-init columns on `feature_proj` — do this ONLY if Phase-2 training
  shows the indirect path insufficient; it complicates the surgery.
* **oracle critic** (`sheepshead/agent/oracle.py`): mirrors the layout with
  51 tokens and the same fixed-leading-slice pattern (oracle.py:296-300);
  append the revealed bag + header fields there too. The revealed *cards*
  are redundant for the oracle (it sees all hands); the header fields are
  not.
* **`convention_wrapper.py` and any mask-shaped tooling**: action-space
  size changes 110 → 117; audit for hardcoded action counts.

---

## Part 3 — Heads (`sheepshead/agent/ppo.py`)

`MultiHeadRecurrentActorNetwork` (ppo.py:29) gains three linear heads,
following the `pick_head` pattern exactly (ppo.py:68):

```python
self.blitz_head = nn.Linear(d_model, 3)     # BLITZ BLACK / BLITZ RED / NO BLITZ
self.crack_head = nn.Linear(d_model, 2)     # CRACK / NO CRACK
self.recrack_head = nn.Linear(d_model, 2)   # RECRACK / NO RECRACK
```

* `action_groups` gains `'blitz'`, `'crack'`, `'recrack'` entries listing
  the new global indices (110..116 zero-based); `_build_logits_from_features`
  scatters them into the full logit vector exactly like `pick_head`'s
  outputs. Masking needs no changes — it flows from
  `get_valid_action_ids()`.
* Per-head temperatures: add `temperature_blitz/crack/recrack`, default 1.0.
* Fresh heads are safe by construction: they only fire at decision points
  that don't exist in rules-off games, so they cannot perturb existing
  behavior.
* Critics need no head changes — value conditioning on multiplier/rule
  flags arrives through the encoder (context token + readouts).

---

## Part 4 — Checkpoint surgery (function-preserving)

Write `sheepshead/analysis/surgery_blitz_crack.py`: loads an old
checkpoint, emits a new-architecture checkpoint computing the **identical
function on rules-off inputs**. The literature basis is OpenAI Five's
surgery and Net2Net (§R): preserve trained weights, initialize every new
parameter so new inputs contribute exactly zero.

Per-tensor recipe (`nn.Linear.weight` is `(out, in)`; new input columns
were appended at the END of each concat in Part 2, so old columns stay
contiguous):

| Tensor | Old shape | New shape | Init of new part |
|---|---|---|---|
| `context_mlp.0.weight` | (d_token, 10+d_card) | (d_token, 18+d_card) | old copied verbatim, 8 zero cols appended — REQUIRES the encoder concat order `[old-10-header, called_emb, new-8-header]` (see note below) |
| `token_mlp_trick.0.weight` | (d_token, d_card+8) | (d_token, d_card+11) | old copied, 3 zero cols appended |
| `token_mlp_hand.0.weight` | (d_token, d_card+4) | (d_token, d_card+7) | old copied, 3 zero cols appended |
| `card_type.weight` | (6, d_token) | (7, d_token) | old 6 rows copied, row 6 fresh normal-init (never touched when bag empty — masked) |
| `blitz/crack/recrack_head.*` | — | new | standard init (unreachable in rules-off games) |
| everything else | — | — | copied verbatim |

**Concat-order note (important for the implementer):** Part 2.1 says
"append the 8 new header fields after the existing 10", but the existing 10
are followed by `called_emb` in the `context_mlp` input concat
(encoder.py:536). To keep the surgery trivial, concatenate as
`[old-10-header, called_emb, new-8-header]` (new fields AFTER the called
card embedding) — then EVERY widened matrix in this table is "old weights
verbatim + zero columns appended at the right edge". Keep the norm-vector
entries paired with wherever the scalar fields actually sit.

Verification gates (all must pass before Part 5):

1. `uv run python -m sheepshead.analysis.capture_arch_goldens --check`
   — capture goldens on master BEFORE starting, check after surgery (this
   is the standing rule for any arch/ppo structural change).
2. Rules-off forward equality: for a batch of recorded rules-off
   observations, old-model and surgered-model logits & values agree to
   0 ULP (exact equality — zero columns × zero features is exact).
3. Rules-off behavioral replay: seeded self-play episodes, old vs surgered
   → identical action sequences.
4. `rigorous_eval` Panel-A gauntlet on the surgered (untrained) model =
   the old model's numbers exactly (same seeds → same games).

---

## Part 5 — Training

### 5.0 Phase 0 — cheap diagnostics BEFORE any training (standing lab style)

* **Critic-threshold crack oracle.** The crack decision is the backgammon
  doubling-cube problem (§R: Keeler & Spencer 1975; TD-Gammon derived cube
  actions from the value function). Build a zero-training baseline: at each
  crack node, crack iff the frozen critic's value for the acting defender
  exceeds a margin δ ≥ 0 (sweep δ ∈ {0, 0.1, 0.25}). Evaluate EV vs
  never-crack over ≥2000 CRN deals with the frozen policy playing all
  seats. This yields (a) the value at stake — if ≈0, deprioritize the
  whole training phase; (b) the sanity floor any learned crack policy must
  beat; (c) the distillation target if warm-starting the head (QDagger
  pattern, §R).
* **Blitz EV probe**: same idea — always-blitz vs never-blitz on deals
  containing a qualifying pair (≈27% of deals), CRN-paired. Blitzing is
  likely near-always-correct for black (two top trump) — the probe
  quantifies the red-blitz and leaster-risk margins.
* Record both in a `## Results` section of this file before proceeding.

### 5.1 Phase 1 — head warm-up (frozen trunk)

Train ONLY the new parameters (3 heads + the zero-init columns + card_type
row 6) on rules-on episodes; everything else frozen. Rules-off behavior is
exactly unchanged while frozen, so no forgetting risk exists yet. Purpose:
(a) get the new heads off random init before trunk gradients flow; (b) a
free probe — if frozen-trunk heads already match the Phase-0
critic-threshold baseline, the trunk barely needs to move in Phase 2.
Optional accelerant: distill the crack head toward the Phase-0 oracle for
the first N updates (kickstarting/QDagger, §R), then drop the loss.

### 5.2 Phase 2 — mixed-mode fine-tune

* **Episode mix** (the anti-forgetting mechanism — rehearsal, which
  empirically beats parameter-space penalties like EWC when you control
  the env distribution, §R): per episode sample
  `blitzing ~ Bern(0.5)`, `cracking ~ Bern(0.5)`,
  `recracking = cracking AND Bern(0.5)`; this leaves 25% pure rules-off
  episodes. Plumb through the trainer config the same way partner-mode
  mixing is plumbed (`pfsp_runtime`/config).
* **LRs**: trunk at reduced LR (~0.1×) via the existing `param_groups`
  mechanism (encoders.py:160/290 precedent — add the new heads to their
  own full-LR group); PPO clip is the trust region. Optional extra
  insurance: KL-to-frozen-anchor penalty computed only on rules-off states
  for the first M updates.
* **Reward/variance handling (D6 consequences)**: reward now spans up to
  ±(base × dotb × 16). Mitigations, in order of preference: (1) keep the
  existing advantage normalization and confirm per-mode advantage stats
  are sane early in the run; (2) config knob `max_stake_log2` (engine-side
  cap, default 4) if tails destabilize; (3) if the critic struggles,
  remember the exact factorization: once play starts the multiplier is
  fixed and public, so V(s) = stake_mult × V_base(s) for play-phase states
  — a critic that reads `stake_mult_log2` from the header can represent
  this, and `--critic-mode oracle` (CTDE, already built) sees it
  privileged.
* **Duration**: this is a fine-tune, not a fresh run — expect the crack/
  blitz heads to converge in O(100k) episodes given every picker game has
  crack nodes; gate on the Phase-5.4 metrics, not a fixed budget.

### 5.3 Phase 3 — league exposure

The decisions are game-theoretic (a crack reveals strength; *not* blitzing
conceals queens; partner-cracks can bluff). Pure self-play can settle on
exploitable conventions, so after Phase 2 stabilizes, fold rules-on
episodes into the PFSP/extended-league setup and add cheap exploit probes
to the league's sanity floor: scripted always-crack, scripted
crack-iff-picker-weak, always-blitz. (AlphaStar exploiter rationale, §R;
matches the existing exploiter-league findings that probes, not yardsticks,
are what scripted agents are good for.)

### 5.4 Gates & falsifiers (pre-registered)

1. **No classic-mode regression**: `rigorous_eval` Panel-A, 1000 CRN deals
   (MDE ≈ 0.07): fine-tuned model vs pre-surgery model on rules-off games
   — difference must be ≥ −MDE. FAIL → increase rules-off mix / lower
   trunk LR / re-run from Phase-1 checkpoint.
2. **Crack quality**: learned crack policy EV ≥ Phase-0 critic-threshold
   baseline EV on CRN deals. FAIL → the head learned noise; inspect
   advantage stats at crack nodes.
3. **Blitz sanity**: near-100% black-blitz rate on hands where Phase-0
   probe showed positive EV; never-blitz where it showed negative.
4. **Signal-use falsifier**: opponents' behavior must differ measurably
   after a blitz reveal (e.g., defender trump-lead rate vs a masked-input
   counterfactual — reuse the E1 adherence-scanner pattern from
   `Convention_Optimality_202607.md`). If behavior is invariant to the
   revealed bag, the encoding failed and the token path needs inspection
   before blaming the training.
5. **Exploitability floor**: league exploit probes gain < threshold vs the
   final policy (reuse exploiter-league harness).

---

## Extensions explicitly deferred (do not build in v1)

* Post-pick blitz window (D1) and partner recrack (D3).
* Multiple independent cracks (D5 variant).
* App/product surface (`app/`): UI for declaring/observing blitzes and
  cracks, table config, payout display — separate plan when the model side
  lands.
* Per-head temperature tuning for deploy (`tune_deploy_search` family).

## §R — References

* **Berner et al. 2019**, *Dota 2 with Large Scale Deep Reinforcement
  Learning* (OpenAI Five), arXiv:1912.06680 — "surgery": function-preserving
  parameter transplantation across observation/action-space changes
  mid-training, applied repeatedly over a 10-month run rather than
  restarting. The direct precedent for Part 4.
* **Chen, Goodfellow & Shlens 2016**, *Net2Net: Accelerating Learning via
  Knowledge Transfer*, ICLR, arXiv:1511.05641 — function-preserving network
  transformations (the theory behind zero-init widening).
* **Agarwal, Schwarzer, Castro, Courville & Bellemare 2022**, *Reincarnating
  Reinforcement Learning*, NeurIPS, arXiv:2206.01626 — umbrella framework
  for reusing prior computation; QDagger (distill-then-finetune) is the
  Phase-1 warm-start pattern.
* **Schmitt et al. 2018**, *Kickstarting Deep Reinforcement Learning*,
  arXiv:1803.03835 — auxiliary KL-to-teacher during fine-tune (the optional
  anchor loss in Phase 2).
* **Kirkpatrick et al. 2017**, *Overcoming Catastrophic Forgetting in
  Neural Networks* (EWC), PNAS, arXiv:1612.00796 — the rejected
  alternative; rehearsal via mixed episode distribution is simpler and
  applies because we control the env.
* **Tesauro 1995**, *Temporal Difference Learning and TD-Gammon*, CACM 38(3)
  — doubling-cube decisions derived from the learned value function; the
  model for the Phase-0 critic-threshold crack oracle.
* **Keeler & Spencer 1975**, *Optimal Doubling in Backgammon*, Operations
  Research 23(4) — the decision theory of stake-doubling thresholds.
* **Zhang, Rao & Agrawala 2023**, *Adding Conditional Control to
  Text-to-Image Diffusion Models* (ControlNet), ICCV, arXiv:2302.05543 —
  zero-initialized connections as the standard idiom for grafting new
  inputs onto trained networks without disturbing them.
* **Alayrac et al. 2022**, *Flamingo: a Visual Language Model for Few-Shot
  Learning*, NeurIPS, arXiv:2204.14198 — zero-init tanh gating for new
  cross-modal pathways (same idiom).
* **Vinyals et al. 2019**, *Grandmaster level in StarCraft II using
  multi-agent reinforcement learning* (AlphaStar), Nature 575 — league
  training with exploiter agents (Phase 3 rationale); also redundant
  observation encoding practice.
* **Nikishin et al. 2023**, *Deep Reinforcement Learning with Plasticity
  Injection*, NeurIPS, arXiv:2305.15555 — when added capacity IS needed for
  a new task; noted so a future session knows this exists and why it is
  NOT needed here (three binary EV decisions strain interface, not
  capacity).
* **Rules source**: sheepshead.org, *Special Situations / House Rules* —
  https://www.sheepshead.org/rules/special-situations-house-rules/
