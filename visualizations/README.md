# PPO architecture 3D visualization

`ppo_architecture_3d.html` is a self-contained, offline three.js walkthrough of
the **perceiver** network's forward pass: input observation → card embeddings →
19 tokens → a transformer *tunnel* (one ring per reasoning layer, per-head
attention chords, FFN lens behind each ring, residual rails carrying each token
through) → memory-token GRU recurrence → the actor's and critic's independent
4-query cross-attention readouts → actor heads (pick / two-tower call /
pointer) → action output, and critic value crystal. There are no pooling bags
or shared feature trunk — that is the point of the perceiver rung. A
**Decision** bar switches between five scenarios captured from a single real
self-played hand (pick, partner call, bury, opening lead, late follow), a
guided tour steps through the stages (dollying through the tunnel layer by
layer) with data-flow particle animation, and **H1–H4** chips toggle
individual attention heads' chords.

## Rebuilding

```sh
.venv/bin/python visualizations/dump_forward_pass.py   # plays a hand, writes ppo_forward_pass.json
.venv/bin/python visualizations/build_3d_html.py       # embeds JSON + vendored three.js → ppo_architecture_3d.html
```

Use the project venv — the system python lacks torch.

- `dump_forward_pass.py` loads a perceiver-arch checkpoint (default
  `runs/ablate_perceiver_s42/best_perceiver.pt`, override with
  `--checkpoint`; loaded via `ppo.load_agent` so arch metadata is honored),
  plays one deterministic hand with the agent in all five seats (per-seat GRU
  memory), snapshots the five decision points with their pre-decision memory,
  and re-runs the forward pass manually (mirroring `encoder.py` /
  `architectures.py`) to capture every intermediate: per-layer **per-head**
  attention (L×H×19×19), per-layer token norms, FFN hidden-activation norms,
  and the actor/critic readout cross-attention (H×4×19 each). If the
  checkpoint passes every hand into a leaster (early/mid training), the seed
  scan falls back to forcing the last seat's PICK so the call/bury phases
  exist; the pick-scenario text is marked "forced" when that happens.
- `build_3d_html.py` splices `ppo_forward_pass.json` and the vendored three.js
  sources into `ppo_3d_template.html` to produce the single ~1.6 MB HTML file.

The HTML is almost entirely data-driven: the scenario buttons, description
text, hand/trick cards, tunnel rings and chords, readout fans, output bars,
and value crystal are all built from the embedded JSON at load time (layer /
head / query counts come from its `dims` block), so most changes only touch
`dump_forward_pass.py` followed by a rerun of both scripts.

## Customizing

**Different checkpoint:** pass `--checkpoint path/to/model.pt` (must be a
perceiver-arch checkpoint — the script refuses others). Re-run the dump
whenever a better checkpoint lands; the template needs no changes.

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
`grep -n "^showStage(0);$"`) with e.g. `switchScenario(2); showStage(9);`,
then screenshot:

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
