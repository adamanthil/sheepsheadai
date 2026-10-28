# Model Artifacts as Release Assets — Plan (October 2026)

**Status (2026-10-27): IMPLEMENTED.** Both releases are published and
immutable, the `artifact/*` tag ruleset is active, and the manifest, fetch
tool, CI lint and panel resolution are committed with the root checkpoint
removed. Decisions taken during implementation are marked **[impl]**; the
one that changes the plan's promise is A4 (panel identity). Split out of `CD_Deployment_Plan_202608.md` (its former D5, D10–D12 and
§5) after the 10-27 audit. **This plan is a prerequisite of the CD
plan**: the deploy script reads the deployed model, its sha256 and its
sampling from the manifest defined here. It is also valuable on its
own: it makes the evaluation panels reproducible from a fresh clone and
puts the release checkpoint somewhere other than one disk under `runs/`.

## TL;DR — what this is

Every model artifact the repository depends on — the deployed
checkpoint AND the seven frozen panel agents — becomes a GitHub Release
asset, pinned by one committed manifest (sha256 + size + provenance +
role), and fetched by one stdlib-only tool. Panels are identified by
bytes, not by paths under gitignored `runs/`. Deploying a new model is
a two-step: upload a release, commit the manifest.

## 0. Decisions and rationale

* **A1 — Release assets, not committed blobs and not Git LFS.** The
  repository is public, so release assets download over plain HTTPS
  with no token (2 GB per asset, no storage or bandwidth quota); Git
  LFS on a public repository would burn its 1 GB/month bandwidth quota
  on every clone of ~130 MB of panels; committing the blobs would add
  130 MB to every clone forever. `final_pfsp_swish_ppo.pt` at the repo
  root was a stopgap and is **removed from the tree** (`git rm`) once
  it is published as an asset like every other model (operator 10-27);
  the ~50 scripts, README lines, the Dockerfile comment,
  `.env.prod.example` and `test_arch_golden`'s legacy-load test that
  name the root path point at `models/pfsp-swish-30m.pt` instead (the
  test keeps its skip-when-absent guard). The blob stays in history
  (added in 21954a9, 10.6 MB) — no history rewrite on a public
  repository with sixteen branches.
* **A2 — One manifest, one tool.** `models/manifest.json` (committed)
  is the single description of every artifact: release tag, asset
  name, sha256, size, roles, arch, provenance, and — on the deployed
  model — its deployment sampling. There is no second copy of any of
  it: the box reads the manifest with `jq` (CD plan §5), so there is no
  `model.env` and no consistency checker between two files.
  `sheepshead.eval.fetch_models` downloads and verifies into
  `models/` (gitignored) and also validates the manifest's shape in CI.
* **A3 — Release assets are made deletion-resistant three ways.**
  (1) Enable the repository's **immutable releases** setting
  (Settings → General → Releases; verify the toggle exists for this
  account at implementation time), which blocks asset deletion and
  re-upload once a release is published; (2) a **tag ruleset** on
  `artifact/*` that forbids deletion and non-fast-forward updates; (3)
  the sha256 pin in git, which turns any tampering into a loud failure
  rather than a silent swap. Belt: the operator's `model-history/`
  archive, and (once the box exists) the CD plan's off-site mirror of
  the box's model cache.
* **A4 — Panel identity is the bytes, not the path.**
  `sheepshead/eval/panels.py` stops naming `runs/...` paths and
  names manifest keys; the panel loader resolves a key to
  `models/<key>.pt` and refuses to run if the file's sha256 does not
  match the manifest. The membership, deal seed and deal list are
  unchanged.
  **[impl, operator 10-27] Members are identified by manifest key, and
  recorded reads are NOT per-deal pairable with new ones.** The plan
  assumed only the on-disk location moved, but two things are taken over
  the panel's path strings: the field's seat order
  (`eval_endpoint` and `rigorous_eval` sort the panel by path before
  drawing seats) and the endpoint design hash (`design_hash` hashes the
  sorted paths; `paired_delta` refuses endpoints whose hashes differ).
  The new paths sort differently (PANEL-A was 30M, 15M, 5M, 100k; it is
  now 15M, 30M, 5M, 100k), so new reads use different seat draws and a
  new hash: PANEL-A `f17b2dc28d0c` → `3e6443e4c5e9`, PANEL-B
  `5fd2564a6425` → `04046aaca26e` (3,996-deal protocol). The rejected
  alternative kept each member's historical path as a frozen ID
  (bit-identical, pairable); the operator chose clean key identity.
  Consequence: every row of `notebooks/panel_reference.csv` stays valid
  as recorded, and a new read is comparable to it **statistically**
  (same members, deals and protocol; different field draws), not by a
  per-deal paired delta. A paired comparison against an old agent means
  re-reading that agent under the new hash.

## 1. Current state (2026-10-27)

The release `runs/202609_recall_rc/final/release.pt`
(= `pi/iter6/bidding/final.pt`, byte-identical), PANEL-A (four files)
and PANEL-B (four files, one shared with A) — seven distinct files, six
under gitignored `runs/` and the 30M at the repo root as a committed
stopgap, ~106 MB total. No release has ever been created on the
repository (`gh release list` is empty). The app reads the deployment
sampling from five env vars (`SHEEPSHEAD_TEMPERATURE_{PICK,PARTNER,
BURY,PLAY}`, `SHEEPSHEAD_LOGIT_BAND`; migration 000008 makes
(model, sampling) the AI player's identity).

## 2. Tag namespace and assets

* Tag namespace `artifact/<name>`. The tag's commit is irrelevant;
  nobody checks out artifact tags. One release may carry several
  assets (the panel bundle); the manifest keys each asset individually.
* **Releases are immutable** (A3, §5 rule 1): a corrected or re-trained
  artifact gets a new name, never a re-upload.
* Bootstrap releases (runbook §6.1; hashes measured 10-27 on the files
  as they sit under `runs/`):

| key | release tag | asset | sha256 (prefix) | bytes | roles | provenance |
|---|---|---|---|---|---|---|
| `recall-rc-theta6` | `artifact/recall-rc-theta6` | `recall-rc-theta6.pt` | `d96a6224ecc3` | 21,140,707 | deploy | `runs/202609_recall_rc/final/release.pt` (= `pi/iter6/bidding/final.pt`); Training_Program_Redesign decision log 10-25 |
| `pfsp-swish-30m` | `artifact/panels-2026-10` | `pfsp-swish-30m.pt` | `cc644b7109d5` | 10,629,409 | panel-a, panel-b | `final_pfsp_swish_ppo.pt` (repo root until the bootstrap release; removed from the tree in §7 step 3) |
| `pfsp-swish-15m` | `artifact/panels-2026-10` | `pfsp-swish-15m.pt` | `6951cd42c52a` | 10,635,253 | panel-a | `runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt` |
| `pfsp-swish-5m` | `artifact/panels-2026-10` | `pfsp-swish-5m.pt` | `c8ec0d1df248` | 10,634,675 | panel-a | `.../pfsp_swish_checkpoint_5000000.pt` |
| `selfplay-swish-100k` | `artifact/panels-2026-10` | `selfplay-swish-100k.pt` | `7f5426b68d3e` | 10,619,055 | panel-a | `runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt` |
| `v2-release` | `artifact/panels-2026-10` | `v2-release.pt` | `6ce80a5e6bc5` | 21,155,651 | panel-b | `runs/rc_validate_v2/final/release.pt` (CE_Teacher §21) |
| `v2-iter11-p1` | `artifact/panels-2026-10` | `v2-iter11-p1.pt` | `348490d8df02` | 21,202,037 | panel-b | `runs/policy_iteration_202609/iter11/distill_epoch7.pt` |
| `v2-league-8m` | `artifact/panels-2026-10` | `v2-league-8m.pt` | `b56ba26c5c09` | 20,979,519 | panel-b | `runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt` |

Full hashes go in the manifest at implementation time (recompute with
`shasum -a 256`; the prefixes above are for cross-checking).

## 3. The manifest: `models/manifest.json` (committed)

```json
{
  "release_base": "https://github.com/adamanthil/sheepsheadai/releases/download",
  "artifacts": {
    "recall-rc-theta6": {
      "tag": "artifact/recall-rc-theta6",
      "asset": "recall-rc-theta6.pt",
      "sha256": "d96a6224ecc3…",
      "bytes": 21140707,
      "roles": ["deploy"],
      "arch": "perceiver-recall",
      "provenance": "runs/202609_recall_rc/final/release.pt; program concluded 2026-10-25",
      "sampling": {"pick": 0.5, "partner": 0.5, "bury": 0.0, "play": 0.5, "band": 1.0}
    },
    "pfsp-swish-30m": {
      "tag": "artifact/panels-2026-10", "asset": "pfsp-swish-30m.pt",
      "sha256": "cc644b7109d5…", "bytes": 10629409,
      "roles": ["panel-a", "panel-b"], "arch": "full",
      "provenance": "legacy swish PFSP 30M (+ flush update); project_final_model_provenance"
    },
    "...": "one entry per row of §2"
  },
  "panels": {
    "PANEL-A": ["pfsp-swish-30m", "pfsp-swish-15m", "pfsp-swish-5m", "selfplay-swish-100k"],
    "PANEL-B": ["pfsp-swish-30m", "v2-release", "v2-iter11-p1", "v2-league-8m"]
  }
}
```

* The artifact key doubles as the app's `SHEEPSHEAD_MODEL_LABEL`.
* `sampling` on the deploy entry is the per-checkpoint deployment
  sampler (§4.5 of the program notebook). **Decided by the operator
  10-27**: T 0.5 on pick, partner and play, bury 0 (the 10-26 split
  attributed the bidding cost to the bury head), global logit band 1.0
  nats. Measured cost vs the release's own argmax field ≈ −0.0155,
  about 17% of deals changed; vs the 30M still ≈ +0.09. A re-temper is
  a manifest commit, not a box edit; it creates a new `ai_player` row
  by design.
* `panels` is the frozen membership, in order. `panels.py` reads it.

## 4. The fetch tool: `sheepshead/eval/fetch_models.py`

```
uv run python -m sheepshead.eval.fetch_models               # everything in the manifest
uv run python -m sheepshead.eval.fetch_models --role panel-a --role panel-b
uv run python -m sheepshead.eval.fetch_models --key recall-rc-theta6
uv run python -m sheepshead.eval.fetch_models --check        # verify local cache, no network
uv run python -m sheepshead.eval.fetch_models --lint         # manifest shape only (CI)
```

Spec: stdlib only (`urllib.request`, `hashlib`, `json`; no new
dependency). For each selected key: if `models/<asset>` exists and
hashes to the manifest, skip; else download
`<release_base>/<tag>/<asset>` to a temp file in `models/`, verify
sha256 and size, `os.replace` into place; a mismatch deletes the temp
file and exits 1 naming the key. `--check` is what the panel loader
calls before every panel read, so a stale or corrupted cache fails
loudly instead of producing a number. `models/` is gitignored except
`manifest.json`.

**[impl]** Files land with mode 0644 (`mkstemp` would leave 0600).

`--lint` is a CI step in the `training` job. It asserts: every sha256
is 64 hex chars; every `bytes` is a positive int; **exactly one** entry
has role `deploy`, and it carries a `sampling` block with exactly the
five keys `pick`, `partner`, `bury`, `play`, `band` as non-negative
numbers (and no other entry carries `sampling`); every `arch` is a
registered architecture; every asset is named `<key>.pt` **[impl]**
(so path order equals key order, A4); `panels` lists only known keys,
and an artifact's `panel-a`/`panel-b` roles agree with the `panels`
lists in both directions. (It replaces the CD plan's former
`check_model_manifest.py`: with one file there is nothing to keep in
sync, only a shape to enforce.)

Panel resolution (A4) **[impl]**: `panels.py` exposes
`panel_files(name) -> list[str]` (repo-root-relative `models/<key>.pt`,
in manifest order), replacing `PANEL_A`/`PANEL_B`. Callers:
`program_config.py` (league panel defaults; `final.references.prod_30m`
→ `models/pfsp-swish-30m.pt`), `league_progress_eval.eval_endpoint`
(default field), `run_ablation_matrix.py`, and a new
`rigorous_eval --panel NAME` (mutually exclusive with `--anchors`).
`build_panel_reference` never read the lists (it reads `.npz` files),
so it is unchanged. The hash check sits in `rigorous_eval.ModelRegistry.get`
— the one load path every panel read goes through — via
`fetch_models.verify_managed`, which checks any file under `models/`
that the manifest names and passes every other path through. Saved run
configs under `runs/*/program/config.json` still name the old paths;
the programs that wrote them have concluded and are not resumed.

## 5. Invariants

1. **Artifact releases are immutable**: never re-upload an asset under
   an existing `artifact/<name>` tag — mint a new name. The sha256 pin
   turns violations into loud failures; the repository setting and tag
   ruleset (A3) make them impossible by accident.
2. **Panel membership is frozen by hash.** A panel read against a file
   that does not hash to the manifest is not a panel read; the loader
   refuses rather than records. A PANEL-C would be a new `panels` list,
   never an edit of A or B.
3. **The manifest is the only description of an artifact.** Nothing
   else in the repository restates a hash, asset name or sampling
   value.

## 6. Operator runbooks

### 6.1 Publish artifacts (bootstrap, once)

```bash
# The seven panel agents in one immutable release
mkdir -p /tmp/panels && cd /tmp/panels
cp final_pfsp_swish_ppo.pt pfsp-swish-30m.pt            # … one copy per §2 row, renamed
gh release create artifact/panels-2026-10 *.pt \
    --title "artifact: evaluation panels (PANEL-A, PANEL-B), 2026-10" \
    --notes "Frozen panel members; provenance per models/manifest.json. Immutable."

# The deployed model
cp runs/202609_recall_rc/final/release.pt recall-rc-theta6.pt
gh release create artifact/recall-rc-theta6 recall-rc-theta6.pt \
    --title "artifact: recall-rc-theta6" \
    --notes "202609_recall_rc release (theta_6); +0.046 vs v2 release, +0.092 vs 30M"

# Then fill models/manifest.json (shasum -a 256 each file) and prove it:
uv run python -m sheepshead.eval.fetch_models --lint
rm -rf models/*.pt && uv run python -m sheepshead.eval.fetch_models   # round-trip from GitHub
```

### 6.2 A later model

One new `artifact/<name>` release; in one commit, add its manifest
entry and move the `deploy` role (and `sampling`) onto it. Once the CD
plan is live, pushing that commit deploys it.

### 6.3 Run the panels from a fresh clone

```bash
git clone https://github.com/adamanthil/sheepsheadai && cd sheepsheadai
uv sync --frozen --extra dev
uv run python -m sheepshead.eval.fetch_models --role panel-a --role panel-b
uv run python -m sheepshead.eval.rigorous_eval ... --panel PANEL-A
```

Same deal seed and deal hash as every row in
`notebooks/panel_reference.csv`.

## 7. Implementation order (each its own commit, per house rules)

1. GitHub settings: immutable releases, `artifact/*` tag ruleset (A3)
   — before anything is published, so the bootstrap is born immutable.
   **[impl]** Immutable releases were already on; ruleset "artifact tags
   are permanent" (id 26009298) blocks deletion, update and
   non-fast-forward on `refs/tags/artifact/*`.
2. Publish the two bootstrap releases (§6.1). **[impl]** Done as drafts
   first, then published; GitHub's server-side sha256 digests and an
   anonymous download of all eight assets match the manifest. An
   interrupted first attempt left two unpublished duplicate drafts,
   since deleted.
3. `models/manifest.json` + `models/.gitignore` + `fetch_models.py` +
   the `--lint` CI step; then, in the same commit,
   `git rm final_pfsp_swish_ppo.pt` and re-point its references to
   `models/pfsp-swish-30m.pt` (A1), so the tree never has both copies.
4. `panels.py` → manifest keys (A4) and its callers. **[impl]** The
   bit-for-bit reproduction check is moot under key identity (A4); a
   small end-to-end read through `eval_endpoint` (PANEL-A) and
   `rigorous_eval --panel PANEL-B` stands in for it.

## 8. Deferred

* **Release-checkpoint regression fixture** (load `recall-rc-theta6`
  through the app's load path, fixed probe states, bit-identical
  logits; skipped when the asset is absent). Declined 10-27 as not
  worth carrying; the manifest hash check covers the bytes, not the
  load path. Re-raise if a checkpoint-load regression ever bites.
* Optional stricter CI (`workflow_dispatch` only): download the deploy
  asset and `torch.load` it against the architecture registry so an
  incompatible checkpoint fails CI, not the boot on the box.
