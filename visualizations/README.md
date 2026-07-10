# PPO architecture 3D visualization

`ppo_architecture_3d.html` is a self-contained, offline three.js walkthrough of
the **perceiver-shared-v2** network's forward pass: input observation → card
embeddings → 19 tokens → a transformer *tunnel* (one ring per reasoning layer,
per-head attention chords, FFN lens behind each ring, residual rails carrying
each token through) → context-token GRU recurrence → ONE **shared 16-query ×
4-head readout** (a 4×4 gem grid; each gem draws its top-3 attended tokens,
and clicking a gem expands its full per-head attention fan) → a 256-d
features hub that both networks consume → the actor's heads (pick / two-tower
call over the card table / a fully opened-up **pointer stage** where the 8
post-reasoning hand tokens re-materialize as a ghost row, combine with the
Wg situation query through per-slot tanh nodes, and emerge as score bars) →
action output; and, on the critic side, the deep value trunk → V(s) crystal
plus a dedicated **aux stage** (win / E[return] / secret-partner gauges,
per-seat points bars, and the 14-chip trump tracker with the unseen-higher
lamp). A **Decision** bar switches between five scenarios captured from a
single real self-played hand (pick, partner call, bury, opening lead, late
follow), a guided tour steps through the 17 stages (dollying through the
tunnel layer by layer) with data-flow particle animation, and **H1–H4** chips
toggle individual attention heads' chords (tunnel and readout fans alike).

A **Network** toggle switches to an analogous 13-stage walkthrough of the
**oracle critic** (`oracle.py: OracleValueNetwork`, the CTDE privileged
critic) on the *same* five decision states, in a violet "privileged
information" identity: full-information observation (all five hands face up,
gold halo on the secret partner, true blind/bury/under, per-seat points) →
its own embedding table (zero parameter sharing) → 51 tokens (the familiar
19 + 32 seat-tinted opponent-hand tokens) → the same-shape 4-layer tunnel at
larger radius → memory-token GRU (U(h,s) recurrence) → 4-query readout →
value trunk → U(h,s) crystal. Until the league run produces a trained
oracle, the dump uses a seeded random init and the UI badges the network
**untrained** everywhere.

## Rebuilding

```sh
.venv/bin/python visualizations/dump_forward_pass.py   # plays a hand, writes ppo_forward_pass.json
.venv/bin/python visualizations/build_3d_html.py       # embeds JSON + vendored three.js → ppo_architecture_3d.html
```

Use the project venv — the system python lacks torch.

- `dump_forward_pass.py` loads a perceiver-shared-v2-arch checkpoint (default
  `runs/ablate_perceiver-shared-v2_s42/perceiver-shared-v2_checkpoint_175000.pt`,
  override with `--checkpoint`; loaded via `ppo.load_agent` so arch metadata
  is honored), plays one deterministic hand with the agent in all five seats
  (per-seat GRU memory), snapshots the five decision points with their
  pre-decision memory, and re-runs the forward pass manually (mirroring
  `encoder.py` / `architectures.py`) to capture every intermediate: per-layer
  **per-head** attention (L×H×19×19), per-layer token norms, FFN
  hidden-activation norms, the shared readout cross-attention (H×16×19) +
  features vector, the pointer's Bahdanau intermediates (g / per-slot Wt /
  tanh-hidden norms / slot scores), two-tower card scores, and the critic's
  value + full aux stack (win / return / secret / per-seat points /
  seen-trump probabilities / unseen-higher). If the checkpoint passes every
  hand into a leaster (early/mid training), the seed scan falls back to
  forcing the last seat's PICK so the call/bury phases exist; the
  pick-scenario text is marked "forced" when that happens.
- `build_3d_html.py` splices `ppo_forward_pass.json` and the vendored three.js
  sources into `ppo_3d_template.html` to produce the single ~1.6 MB HTML file.

The HTML is almost entirely data-driven: the scenario buttons, description
text, hand/trick cards, tunnel rings and chords, readout fans, output bars,
and value crystal are all built from the embedded JSON at load time (layer /
head / query counts come from its `dims` block), so most changes only touch
`dump_forward_pass.py` followed by a rerun of both scripts.

## Customizing

**Different checkpoint:** pass `--checkpoint path/to/model.pt` (must be a
shared-readout perceiver variant — the script refuses others). Re-run the
dump whenever a better checkpoint lands; the template needs no changes.

**Oracle critic:** by default the dump instantiates a fresh
`OracleValueNetwork()` with `torch.manual_seed(--oracle-seed)` (default
20260709) — the manual forward replication is asserted `allclose` against
the module's own `encode_batch`/readout, so every dump doubles as an
architecture smoke test. The oracle's memory is threaded per seat over the
same event stream the agent's memory sees (decisions + end-of-trick
observations, zero-init per hand). Once a league run saves oracle weights,
pass `--oracle-checkpoint path/to/ckpt.pt` (any checkpoint carrying an
`oracle_state_dict`, i.e. saved with `--critic-mode oracle`) and the
untrained badges disappear. Oracle transformer attention is stored sparse
(top-400 directed `[head, i, j, w]` triples per layer) to keep the JSON
small; the readout attention (4×4×51) stays dense.

**Different hand:** `find_hand()` scans seeds from 0 and keeps the first hand
containing all five decision types. To force another hand, start the scan past
the current winner (e.g. `range(4, max_seeds)`) or hardcode a seed. Any seed
works as long as someone picks, calls a partner, and the hand plays out
(leasters are rejected in `select_snapshots()`).

**Different moment per scenario:** edit `select_snapshots()`. It currently
takes the eventual picker's pick decision, the first call/bury/lead, and the
"richest" defender follow (`follow_richness` prefers more cards on screen,
then later tricks). E.g. show the second bury with `snapshots["bury"][1]`, or
a picker follow by dropping the defender filter.

**Adding a scenario** (a mid-hand lead, the partner's first play after being
revealed, …):

1. In `classify_decision()`, return a new kind string for the decision point
   you want — it sees the valid-action names and the full state dict, so you
   can key off trick number, cards on table, `state["partner_rel"]`, etc.
2. Add the kind to `SCENARIO_ORDER` / `SCENARIO_LABELS`, pick one snapshot for
   it in `select_snapshots()`, and add a branch in `describe_scenario()`.
3. Template side, the only thing to check is `liveHead()` in
   `ppo_3d_template.html`: it maps scenario kind → which policy head renders
   "live" (`pick`, `call`, or `pointer`). Any play/bury decision is already
   `pointer`; other kinds need a mapping.

Everything downstream — JSON schema, selector bar, stage text, top-8 output
bars with the ✓ on the chosen action — adapts automatically. Caveat: all
scenarios must come from the *same* hand, so a rare new kind may push the
chosen seed higher or require raising `max_seeds` in `find_hand()`.

## Headless verification

To eyeball the built page without a display: copy it to a temp dir, pin a
scenario/stage by replacing the boot line `showStage(0);` (find it with
`grep -n "^showStage(0);$"`) with e.g. `switchScenario(2); showStage(9);`
— prepend `switchNetwork(1);` to land in the oracle walkthrough (13 stages,
own index space) — then screenshot:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader \
  --window-size=1920,1080 --hide-scrollbars --virtual-time-budget=15000 \
  --screenshot=out.png file:///path/to/copy.html
```

Use the SwiftShader flags, not `--disable-gpu` (which kills WebGL context
creation). Headless captures can come out color-inverted — a capture artifact,
not a page bug. Add `--enable-logging=stderr` and grep for `CONSOLE.*error`
to catch JS errors.

## Files

- `ppo_architecture_3d.html` — built artifact (open directly in a browser)
- `ppo_3d_template.html` — page source with `__DATA_JSON__` etc. placeholders
- `dump_forward_pass.py` — perceiver forward-pass capture → `ppo_forward_pass.json`
- `build_3d_html.py` — template + JSON + vendor → single-file HTML
- `vendor/` — pinned three.js core, OrbitControls, and an esbuild addons
  bundle; see `vendor/README.md` for the rebuild recipe
