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
features hub that both networks consume, which feeds two clearly separated
sides of the scene (focusing either side pushes the other far back). The
**actor** side (+z, titled ACTOR) runs from the actor adapter into three
decision lanes, each holding exactly the heads whose logits share one
masked softmax. The top row, in game order, holds **pick** (PICK / PASS)
and **partner call** (the two-tower CALL scorer over the card table plus
ALONE / JD PARTNER). The bottom row is the **cards** lane: the 8
post-reasoning hand tokens re-materialize as a ghost row, left to right,
and are scored by two terms. The additive Bahdanau term,
v·tanh(Wg·f + Wt·t), goes through per-slot tanh nodes, and the bilinear
term, (U·f)·(V·t)/√64, through per-slot dot-product rings. Their sum is
each slot's score bar, and PLAY UNDER closes the row. Every legal action
then flows from the element that scored it into a single **softmax** ring,
which feeds a ranked list of horizontal π(a | s) bars: a summary of the
whole actor, deliberately unlike the heads' vertical logit bars. The
**critic** side (−z, lower, titled CRITIC) holds the deep value trunk →
V(s) crystal and the **aux** lane (win / E[return] / secret-partner
gauges, per-seat points bars, and the 14-chip trump tracker with the
unseen-higher lamp).

The page is laid out as a figure with a caption, sized for a blog column
(it is headed for an embedded writeup) and working down to phone width.
The canvas carries only two buttons, **⋯** (options) and **⤢** (expand);
everything else sits below it:

- a **Decision** select (five scenarios captured from one real self-played
  hand: pick, partner call, bury, a lead, a late follow) and a
  **Policy | Oracle critic** switch;
- a chapter rail: Overview, then Observation · Tokens · Transformer ·
  Memory · Readout · Policy · Critic. Prev/Next and **▶ Tour** step chapter
  to chapter. Chapters with more than one view show sub-step chips
  (Transformer: all layers / layer 1–4; Policy: heads / pointer / output;
  Critic: value / aux heads). The Policy chapter lands on whichever head
  decides the current scenario;
- the narration, shown in full. Clicking anything in the scene replaces it
  with that object's details and a **‹ Back** link.

The **⋯** menu holds the per-head attention toggles (H1–H4, filtering
chords in every attention block: tunnel self-attention, readout gem fans,
and the gems' default top-3 chords, whose head-average is recomputed over
the enabled heads), data-flow particles, auto-rotate, the floor stage
names (off by default), tour speed, the theme, and the color key.
**⤢** expands the figure to fill the current window (not OS fullscreen):
the scene takes the whole window and the caption becomes a floating panel
on the right (along the bottom on narrow windows) with a vertical chapter
list and the key. The camera's projection is offset so the diagram centers
in the space beside the panel.

Inline, the figure does not capture page scrolling. A mouse can orbit
immediately, but the wheel zooms only after a click in the figure. On
touch, the page scrolls until a tap hands gestures to the scene, and
tapping outside hands them back. Rendering pauses while the figure is
scrolled out of view.

**Themes.** `dark` (default) is the original glow look: additive blending,
emissive materials, bloom. `light` is a "paper diagram" for white pages in
the blog's palette (white ground, #555 ink, Transat type when the host page
provides it). It uses the same hues pulled down to ink strength, with normal
blending and no bloom (bloom would wash out a white ground). Depth comes
from a hemisphere fill plus a key light, soft shadows on an invisible floor
(cast only by the stage in focus), a faint backdrop fall-off, and bolder
connection strokes. Pick one at load with `?theme=light` or
`?theme=dark`, or switch in the ⋯ menu. A third option, **Dark scene**
(`?theme=mixed`), frames the dark glowing scene in the light page and
caption: the page chrome (CSS, `<html data-theme>`) and the scene
(`THEMES`) are chosen independently (`THEME_PARTS`). The light and mixed
themes are still experiments.

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

The **Oracle critic** switch changes to an analogous walkthrough of the
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
  `runs/202609_recall_rc/league/checkpoints/checkpoint_3000000.pt`, the
  gen-3 boundary; override with `--checkpoint`; loaded via `ppo.load_agent`
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
view by replacing the boot line `showStage(0);` (find it with
`grep -n "^showStage(0);$"`) with e.g.
`switchScenario(2); showStage(stageIndex('pointer'));`. Prepend
`switchNetwork(1);` for the oracle walkthrough (its stage labels are
`oObs`, `oXfL2`, …), `setMenu(true);` to open the options menu, or
`setExpanded(true);` for the expanded layout, and append `?theme=light`
to the file URL for the light theme. Then screenshot:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader \
  --window-size=1920,1080 --hide-scrollbars --virtual-time-budget=15000 \
  --screenshot=out.png file:///path/to/copy.html
```

Use the SwiftShader flags, not `--disable-gpu` (which kills WebGL context
creation). Headless captures can come out color-inverted — a capture artifact,
not a page bug. Add `--enable-logging=stderr` and grep for `CONSOLE.*error`
to catch JS errors. Headless Chrome will not make a window narrower than
about 500px, so to check phone width, load the page in a 375px-wide
`<iframe>` inside a small wrapper page. That also exercises the embed path
below.

## Embedding

Embed it in an `<iframe>`, which keeps the page's three.js payload and CSS
out of the host. Inside a frame, the page drops its own padding and posts
two messages to the parent. One reports its content height, so the frame
can fit the figure and caption as the caption changes length. The other
asks the host to expand or collapse the frame when **⤢** is used (a frame
cannot grow past its own box by itself):

```html
<iframe id="ppo-viz" src="ppo_architecture_3d.html?theme=light"
  style="width:100%;height:760px;border:0;display:block"
  loading="lazy" title="Sheepshead network architecture"></iframe>
<script>
  addEventListener("message", (e) => {
    const f = document.getElementById("ppo-viz");
    if (!e.data || e.source !== f.contentWindow) return;
    if (e.data.type === "sheepshead-viz:height" && !f.dataset.expanded)
      f.style.height = e.data.height + "px";
    if (e.data.type === "sheepshead-viz:expand") {
      f.dataset.expanded = e.data.expanded ? "1" : "";
      Object.assign(f.style, e.data.expanded
        ? { position: "fixed", inset: "0", width: "100vw", height: "100vh", zIndex: "1000" }
        : { position: "", inset: "", width: "100%", zIndex: "" });
      document.body.style.overflow = e.data.expanded ? "hidden" : "";
    }
  });
</script>
```

The blog's Transat webfonts don't reach into the frame, so the figure falls
back to the system sans unless the fonts are also served to it.

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
