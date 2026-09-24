# PPO architecture 3D visualization

`ppo_architecture_3d.html` is a self-contained, offline three.js walkthrough of
the **perceiver-recall** network's forward pass (the release-candidate
architecture, Training_Program_Redesign_202609 §3): input observation → card
embeddings → 15 tokens (the human-recall layout — no blind/bury tokens) → a
transformer *tunnel* (one ring per reasoning layer, per-head attention
chords, FFN lens behind each ring, residual rails carrying each token
through) → memory-token GRU recurrence → ONE **shared 16-query ×
4-head readout** (a 4×4 gem grid; each gem draws its top-3 attended tokens,
and clicking a gem expands its full per-head attention fan) → a 256-d
features hub that both networks consume → the actor's heads (pick / two-tower
call over the card table / a fully opened-up **pointer stage** where the 8
post-reasoning hand tokens re-materialize as a ghost row and are scored by
two terms — the additive Bahdanau term, v·tanh(Wg·f + Wt·t), through
per-slot tanh nodes, and the bilinear term, (U·f)·(V·t)/√64, through
per-slot dot-product rings — whose sum is each slot's score bar) →
action output; and, on the critic side, the deep value trunk → V(s) crystal
plus a dedicated **aux stage** (win / E[return] / secret-partner gauges,
per-seat points bars, and the 14-chip trump tracker with the unseen-higher
lamp). A **Decision** bar switches between five scenarios captured from a
single real self-played hand (pick, partner call, bury, opening lead, late
follow), a guided tour steps through the 17 stages (dollying through the
tunnel layer by layer) with data-flow particle animation, and **H1–H4** chips
toggle individual attention heads' chords in every attention block: tunnel
self-attention, readout gem fans, and the gems' default top-3 chords (whose
head-average is recomputed over just the enabled heads — one chip on shows
that head's own top-3).

Cards throughout the scene render in the product app's visual language
(`app/web/lib/ds/PlayingCard`, copied into the template — no dependency):
paper faces with serif corner rank + suit, red/black ink, a gold rim and
gold point badge on trump, and the hatched green back for face-down cards.
Each face also carries its own 16-d embedding-table row as a color barcode
(grouped by the informed-init dims: suit / rank / pts / under / learned) —
clicking any card shows the exact values. The Observation stage lays the
situation out directly in the scene: the hand as an arced fan (with a hand-summary
line and a marker on the card the policy ends up choosing), trick cards with
seat/role attribution, the header scalars with their real values, and the
256-d memory as a heat strip. Once the picker has buried, its two buried
cards appear as ghosts beside the memory chip: known to the seat, absent
from the observation, so only the memory can carry them.

A **Network** toggle switches to an analogous 13-stage walkthrough of the
**oracle critic** (`oracle.py: OracleValueNetwork`, the CTDE privileged
critic) on the *same* five decision states, in a violet "privileged
information" identity: full-information observation (all five hands face up,
gold halo on the secret partner, true blind/bury/under, per-seat points) →
its own embedding table (zero parameter sharing) → 51 tokens (the policy's
15 + 4 true blind/bury + 32 seat-tinted opponent-hand tokens) → the
same-shape 4-layer tunnel at larger radius → memory-token GRU (U(h,s)
recurrence) → 4-query readout → value trunk → U(h,s) crystal, plus the two
deterministic team aux heads. The oracle weights come from the same league
checkpoint as the policy; a dump without them falls back to a seeded random
init and the UI badges the network **untrained** everywhere.

## Rebuilding

```sh
.venv/bin/python visualizations/dump_forward_pass.py   # plays a hand, writes ppo_forward_pass.json
.venv/bin/python visualizations/build_3d_html.py       # embeds JSON + vendored three.js → ppo_architecture_3d.html
```

Use the project venv — the system python lacks torch.

- `dump_forward_pass.py` loads a perceiver-recall checkpoint (default
  `runs/202609_recall_rc/league/checkpoints/checkpoint_2000000.pt`, the
  gen-2 boundary; override with `--checkpoint`; loaded via `ppo.load_agent`
  so arch metadata is honored), plays one deterministic hand with the agent
  in all five seats (per-seat GRU memory, observations via
  `observation_for`), snapshots the five decision points with their
  pre-decision memory and the picker's blind/bury knowledge, and re-runs
  the forward pass manually (mirroring `encoder.py` /
  `architectures/encoders.py`, asserted `allclose` against the modules'
  own forward) to capture every intermediate: per-layer **per-head**
  attention (L×H×15×15), per-layer token norms, FFN hidden-activation
  norms, the memory-token GRU write, the shared readout cross-attention
  (H×16×15) + features vector, the pointer's two terms (g / u norms,
  per-slot Wt / V norms, tanh-hidden norms, additive / bilinear / total
  slot scores), two-tower card scores, and the critic's
  value + full aux stack (win / return / secret / per-seat points /
  seen-trump probabilities / unseen-higher). If the checkpoint passes every
  hand into a leaster (early/mid training), the seed scan falls back to
  forcing the last seat's PICK so the call/bury phases exist; the
  pick-scenario text is marked "forced" when that happens.
- `build_3d_html.py` splices `ppo_forward_pass.json` and the vendored three.js
  sources into `ppo_3d_template.html` to produce the single ~1.9 MB HTML file.

The HTML is almost entirely data-driven: the scenario buttons, description
text, hand/trick cards, tunnel rings and chords, readout fans, output bars,
and value crystal are all built from the embedded JSON at load time (layer /
head / query counts come from its `dims` block), so most changes only touch
`dump_forward_pass.py` followed by a rerun of both scripts.

## Customizing

**Different checkpoint:** pass `--checkpoint path/to/model.pt` (must be a
perceiver-recall checkpoint — the script refuses others). Re-run the
dump whenever a better checkpoint lands; the template needs no changes.

**Oracle critic:** the oracle weights come from `--oracle-checkpoint`, or
from `--checkpoint` itself when it carries an `oracle_state_dict` (league
checkpoints do); only when neither does is a fresh `OracleValueNetwork()`
built with `torch.manual_seed(--oracle-seed)` (default 20260709) and badged
untrained. The manual forward replication is asserted `allclose` against
the module's own `encode_batch`/readout, so every dump doubles as an
architecture smoke test. The oracle's memory is threaded per seat over the
same event stream the agent's memory sees (decisions + end-of-trick
observations, zero-init per hand). Oracle transformer attention is stored sparse
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

# Training program infographic

`training_program.html` is a single, self-contained page (no build step, no
dependencies): a vertical flowchart of the training pipeline run by
`sheepshead/training/run_training_program.py`, written for the blog. Five
stage cards (bootstrap, oracle critic, league, search-guided policy
iteration, certification) with an icon, one line each and a pill naming the
stage's learning signal; decision diamonds between them, and loop-back
arrows on the two iterative stages. Clicking a card opens the same three
bullets on every stage (method with a paper reference, why, ends when);
clicking a dotted term opens a one-sentence popover. It deliberately names
no scripts, flags or files; update it when the pipeline's shape changes,
not when its parameters do.

It is light-only to sit on the blog's white article card, and all of its
CSS, ids and script are scoped under `figure.tp`, so the `<style>`,
`<figure>` and `<script>` blocks can be pasted inline into a post (it
picks up the blog's Transat fonts there) or the file can be iframed. Sizing
uses container queries on the figure, not the viewport. Open it directly in
a browser to preview.
