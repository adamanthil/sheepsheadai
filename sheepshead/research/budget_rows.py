#!/usr/bin/env python3
"""Would a larger search budget change the distill target at given nodes?
(Arm_D_Review §9.5)

``generate`` re-creates scanned /analyze decisions (e.g. the over-trump scan,
``scan_called_suit_overtrumps``) as corpus rows at several search budgets. It
runs the REAL ``distill_corpus.play_corpus_game`` with three patches: the
deal is the /analyze seed's, every seat's action is forced to the logged
trace action, and search fires only at the target nodes. Each game is
generated once per budget with the same det_rng (common random numbers), and
written as one corpus-format shard per budget.

``targets`` sends those rows through a finished iteration's own target-stage
pieces: row encoding with the encoder that iteration's AdvantageModel was fit
on, its AdvantageModel (the deals are fresh, so its predictions are honest),
its per-class sigma_u^2, the centered noise model against its class pools,
and its global eta tilt. Scenarios: each budget as configured; the higher
budget at these nodes only (class pools unchanged) or for the whole class
(pools scaled by the measured replicate-variance ratio); the cell's own
sigma_u^2; search only.

Usage (from repo root):

    uv run python -m sheepshead.research.budget_rows generate \
        --scan runs/202609_recall_rc/overtrump_probe/scan_iter3.jsonl \
        --ckpt runs/202609_recall_rc/pi/iter3/distill_epoch10.pt \
        --budgets 256 512 --out-dir runs/202609_recall_rc/overtrump_probe
    uv run python -m sheepshead.research.budget_rows targets \
        --scan runs/202609_recall_rc/overtrump_probe/scan_iter3.jsonl \
        --rows-dir runs/202609_recall_rc/overtrump_probe --budgets 256 512
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
import torch

from sheepshead.game import ACTIONS, DECK, PARTNER_BY_CALLED_ACE, TRUMP_POWER, Game
from sheepshead.rewards import RETURN_SCALE
from sheepshead.rl.search_advantage import (
    NOISE_POOL_DF,
    AdvantageModel,
    RowTable,
    apply_noise_model,
    blend_advantages,
    build_row_table,
    build_tilt_target,
    predict_rows,
)
from sheepshead.search.ismcts import (
    ISMCTSConfig,
    ISMCTSTeacher,
)
from sheepshead.training import distill_corpus as dc

DEFAULT_ITER = "runs/202609_recall_rc/pi/iter3"
NOISE_SHRINK_NU = 4.0
GENERATE_SEED = 20260916  # det_rng base for the regenerated games

# Per worker process: one teacher per budget or named engine variant, and
# the variant's rollout-depth override (None = the corpus config's).
_TEACHERS: dict[int | str, ISMCTSTeacher] = {}
_D_ROLLOUT: dict[int | str, int | None] = {}

# Engine knobs a ``--variant`` may set (Arm_D_Review §9.14 channel tests).
VARIANT_KEYS = ("iters", "leaf_evaluator", "max_depth_play", "d_rollout")


def parse_variant(spec: str) -> dict:
    """``label:key=value,key=value`` -> {"label": ..., key: value}. Keys:
    iters (int), leaf_evaluator (oracle|limited), max_depth_play (int),
    d_rollout (int; 99 = terminal, no bootstrap)."""
    label, _, body = spec.partition(":")
    if not label or not body:
        raise SystemExit(f"--variant needs label:key=value[,...], got {spec!r}")
    out: dict = {"label": label}
    for kv in body.split(","):
        k, _, v = kv.partition("=")
        if k not in VARIANT_KEYS:
            raise SystemExit(f"--variant key {k!r} not in {VARIANT_KEYS}")
        out[k] = v if k == "leaf_evaluator" else int(v)
    return out


def variant_teacher(agent, v: dict) -> ISMCTSTeacher:
    heads = ("pick", "partner", "bury", "play")
    kwargs: dict = {"iters": {h: int(v.get("iters", 256)) for h in heads}}
    if "leaf_evaluator" in v:
        kwargs["leaf_evaluator"] = v["leaf_evaluator"]
    if "max_depth_play" in v:
        depth = dict(ISMCTSConfig().max_depth)
        depth["play"] = int(v["max_depth_play"])
        kwargs["max_depth"] = depth
    return ISMCTSTeacher(agent, ISMCTSConfig(**kwargs))


def trace_actions(seeds: list[int], ckpt: str) -> dict[int, list[tuple[int, int, int]]]:
    """(stepIndex, seat, action id) of each seed's deterministic /analyze
    trace, called-ace mode."""
    from server.api.schemas import AnalyzeSimulateRequest
    from server.services.analyze import simulate_game
    from sheepshead.research.scan_defender_trump_leads import load_scan_agent

    agent = load_scan_agent(ckpt)
    out = {}
    for s in seeds:
        resp = simulate_game(
            AnalyzeSimulateRequest(seed=s, partnerMode=1, deterministic=True),
            agent=agent,
        )
        out[s] = [
            (ad.stepIndex, ad.seat, ACTIONS.index(ad.action) + 1) for ad in resp.trace
        ]
    return out


def _worker_init(init_args: dict) -> None:
    dc.worker_init(init_args)
    agent = dc._W["agent"]
    for b in init_args["budgets"]:
        _TEACHERS[b] = ISMCTSTeacher(
            agent,
            ISMCTSConfig(iters={h: b for h in ("pick", "partner", "bury", "play")}),
        )
        _D_ROLLOUT[b] = None
    for v in init_args.get("variants", []):
        _TEACHERS[v["label"]] = variant_teacher(agent, v)
        _D_ROLLOUT[v["label"]] = v.get("d_rollout")


def _run_game(task: dict) -> dict:
    """One /analyze deal through play_corpus_game, once per budget."""
    seed, actions, targets = task["seed"], task["actions"], task["targets"]
    by_seat: dict[int, list[int]] = defaultdict(list)
    step_of: dict[tuple[int, int], int] = {}
    for step, seat, a in actions:
        step_of[(seat, len(by_seat[seat]))] = step
        by_seat[seat].append(a)
    target_keys = {k for k, step in step_of.items() if step in targets}
    agent = dc._W["agent"]
    real_act = type(agent).act
    used: dict[int, int] = defaultdict(int)
    flag = {"target": False}

    def forced_act(self, state, valid, pos, *a, **kw):
        _, log_prob, value = real_act(self, state, valid, pos, *a, **kw)
        k = used[pos]
        used[pos] += 1
        flag["target"] = (pos, k) in target_keys
        act = by_seat[pos][k]
        if act not in valid:
            raise RuntimeError(
                f"seed {seed}: logged action {act} illegal for seat {pos} at its decision {k}"
            )
        return act, log_prob, value

    def deal(partner_selection_mode, seed=None):
        return Game(partner_selection_mode=partner_selection_mode, seed=task["seed"])

    out: dict = {"seed": seed, "by_budget": {}}
    orig_game, orig_sched = dc.Game, dc.schedule_p
    orig_d_rollout = dc._W["args"]["d_rollout"]
    try:
        dc.Game = deal
        dc.schedule_p = lambda *a, **kw: 1.0 if flag["target"] else 0.0
        agent.act = forced_act.__get__(agent)
        for b, teacher in _TEACHERS.items():
            used.clear()
            dc._W["teacher"] = teacher
            # play_corpus_game reads the rollout depth from the worker args.
            override = _D_ROLLOUT.get(b)
            dc._W["args"]["d_rollout"] = (
                orig_d_rollout if override is None else override
            )
            res = dc.play_corpus_game((seed, PARTNER_BY_CALLED_ACE, False))
            searched = sum(c["searched"] for c in res["counts"].values())
            if searched != len(target_keys):
                raise RuntimeError(
                    f"seed {seed}: searched {searched} nodes, expected {len(target_keys)}"
                )
            out["by_budget"][b] = {
                "games": {
                    "game": seed,
                    "mode": "called",
                    "is_leaster": res["is_leaster"],
                    "alone_called": res["alone_called"],
                },
                "episodes": res["episodes"],
            }
    finally:
        dc.Game, dc.schedule_p = orig_game, orig_sched
        dc._W["args"]["d_rollout"] = orig_d_rollout
        del agent.act
    return out


def generate(
    scan: str,
    ckpt: str,
    budgets: list[int],
    out_dir: str,
    manifest_iter: str,
    workers: int,
    limit: int,
    variants: list[dict] | None = None,
) -> None:
    variants = variants or []
    labels: list[int | str] = [*budgets, *(v["label"] for v in variants)]
    if not labels:
        raise SystemExit("generate needs at least one --budgets value or --variant")
    nodes = [json.loads(line) for line in open(scan)]
    targets: dict[int, set[int]] = defaultdict(set)
    for n in nodes:
        targets[n["seed"]].add(n["stepIndex"])
    seeds = sorted(targets)[: limit or None]
    acts = trace_actions(seeds, ckpt)
    tasks = [
        {"seed": s, "actions": acts[s], "targets": sorted(targets[s])} for s in seeds
    ]
    print(
        f"{len(tasks)} games, {sum(len(t['targets']) for t in tasks)} target nodes",
        flush=True,
    )
    with open(os.path.join(manifest_iter, "corpus", "manifest.json")) as f:
        config = json.load(f)["config"]
    init_args = {
        **config,
        "ckpt": ckpt,
        "iters_schedule": "",
        "budgets": budgets,
        "variants": variants,
        "routed_encoder": None,
        "seed": GENERATE_SEED,
    }
    shards: dict[int | str, dict[str, list]] = {
        b: {"games": [], "episodes": []} for b in labels
    }
    with get_context("spawn").Pool(
        workers, initializer=_worker_init, initargs=(init_args,)
    ) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_game, tasks), 1):
            for b, d in r["by_budget"].items():
                shards[b]["games"].append(d["games"])
                shards[b]["episodes"].extend(d["episodes"])
            if i % 20 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}]", flush=True)
    for b, s in shards.items():
        torch.save(s, os.path.join(out_dir, f"budget_corpus_{b}.pt"))


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def _load_budget_rows(
    path: str, scan_nodes: dict[int, list[dict]], agent, model: AdvantageModel
):
    """(row table, a_model, over-trump mask, node keys) of one budget's shard."""
    shard = torch.load(path, weights_only=False)
    games = [g["game"] for g in shard["games"] for _ in range(5)]
    t = build_row_table(agent, shard["episodes"], shard_idx=0, game_indices=games)
    with torch.no_grad():
        a_model, _ = predict_rows(model, t, torch.arange(len(t)))
    over = torch.zeros_like(t.masks, dtype=torch.bool)
    keys = []
    for i, (_, ep, ev) in enumerate(t.refs):
        e = shard["episodes"][ep][ev]
        seed = shard["games"][ep // 5]["game"]
        hand = sorted(DECK[int(x) - 1] for x in e["state"]["hand_ids"] if int(x))
        node = next(n for n in scan_nodes[seed] if sorted(n["hand"]) == hand)
        for a in e["valid_actions"]:
            c = ACTIONS[a - 1][5:]
            if c in TRUMP_POWER and TRUMP_POWER[c] > TRUMP_POWER[node["winner"]]:
                over[i, a - 1] = True
        keys.append((seed, node["stepIndex"]))
    return t, a_model, over, keys


def _noise(
    t: RowTable, pools: dict[str, float], s2_global: float, pool_scale: float = 1.0
) -> torch.Tensor:
    """apply_noise_model's centered formula against fixed class pools (the
    iteration's, not these few rows')."""
    lab = t.label_mask
    nobs = t.noise_nobs.to(torch.float64)
    ok = lab & (nobs >= 2) & torch.isfinite(t.noise_s2c)
    df = torch.where(ok, nobs - 1.0, torch.zeros_like(nobs))
    s2c = torch.where(ok, t.noise_s2c.double(), torch.zeros_like(nobs))
    pool = torch.tensor(
        [pools.get(c, s2_global) * pool_scale for c in t.node_class],
        dtype=torch.float64,
    )
    blend = (NOISE_SHRINK_NU * pool.unsqueeze(1) + df * s2c) / (NOISE_SHRINK_NU + df)
    per_action = torch.where(ok, blend / nobs.clamp(min=1), torch.zeros_like(nobs))
    return (per_action.sum(1) / ok.sum(1).clamp(min=1)).float()


def targets(
    scan: str, rows_dir: str, budgets: list[int], iter_dir: str, encoder_ckpt: str
) -> None:
    from sheepshead.agent.ppo import load_agent

    lo_b, hi_b = budgets
    scan_nodes: dict[int, list[dict]] = defaultdict(list)
    for line in open(scan):
        r = json.loads(line)
        scan_nodes[r["seed"]].append(r)
    agent = load_agent(encoder_ckpt)
    with open(os.path.join(iter_dir, "fit_report.json")) as f:
        fit = json.load(f)
    model = AdvantageModel(agent, fit["selected"], heteroscedastic=True, bilinear=True)
    model.load_state_dict(torch.load(os.path.join(iter_dir, "advantage_model.pt")))
    model.eval()
    with open(os.path.join(iter_dir, "target_report.json")) as f:
        rep = json.load(f)
    su2_class, eta, tilt_max = (
        rep["sigma_u2_by_class"],
        float(rep["eta"]),
        float(rep["tilt_max"]),
    )
    full = RowTable.load(os.path.join(iter_dir, "row_table.pt"))
    st = apply_noise_model(full, "centered", NOISE_SHRINK_NU, pool="class")
    pools, s2_global = st["s2_centered_by_class"], st["s2_centered_global"]
    del full
    print(f"(class pools from {iter_dir}: NOISE_POOL_DF {NOISE_POOL_DF:.0f})")

    rows = {
        b: _load_budget_rows(
            os.path.join(rows_dir, f"budget_corpus_{b}.pt"), scan_nodes, agent, model
        )
        for b in budgets
    }
    # A committee with no usable root statistics leaves a node untargetable,
    # at different nodes per budget: compare on the shared nodes.
    common = sorted(set(rows[lo_b][3]) & set(rows[hi_b][3]))
    print(
        f"targetable nodes: {lo_b} {len(rows[lo_b][3])}, {hi_b} {len(rows[hi_b][3])}, both {len(common)} (of {sum(len(v) for v in scan_nodes.values())})"
    )

    def align(b: int):
        t, a_model, over, keys = rows[b]
        pos = {k: i for i, k in enumerate(keys)}
        sel = torch.tensor([pos[k] for k in common])
        return t.subset(sel), a_model[sel], over[sel]

    t_lo, am_lo, over = align(lo_b)
    t_hi, am_hi, over_hi = align(hi_b)
    if not torch.equal(over, over_hi):
        raise RuntimeError("over-trump masks differ between budgets")
    prior, lab = t_lo.prior, t_lo.label_mask

    def over_minus_alt(t: RowTable) -> torch.Tensor:
        p = t.prior
        po = torch.where(over, p, torch.full_like(p, -1)).argmax(1)
        pn = torch.where(~over & t.masks.bool(), p, torch.full_like(p, -1)).argmax(1)
        r = torch.arange(len(t))
        return RETURN_SCALE * (t.advantage[r, po] - t.advantage[r, pn])

    d_lo, d_hi = over_minus_alt(t_lo), over_minus_alt(t_hi)
    ratio = float(
        np.nanmean(t_hi.noise_s2c[lab].numpy())
        / np.nanmean(t_lo.noise_s2c[lab].numpy())
    )
    nv_lo = _noise(t_lo, pools, s2_global)
    nv_hi_local = _noise(t_hi, pools, s2_global)
    nv_hi_class = _noise(t_hi, pools, s2_global, ratio)
    su2_cls = torch.tensor([su2_class[c] for c in t_lo.node_class])
    r2_hi = ((t_hi.advantage - am_hi) ** 2 * lab).sum(1) / lab.sum(1)
    r2_lo = ((t_lo.advantage - am_lo) ** 2 * lab).sum(1) / lab.sum(1)
    su2_cell_hi = max(float(r2_hi.mean() - nv_hi_class.mean()), 1e-7)
    su2_cell_lo = max(float(r2_lo.mean() - nv_lo.mean()), 1e-7)
    n = len(t_lo)
    print(f"{n} nodes; policy over-trump mass {float(prior[over].sum() / n):.3f}")
    print(
        f"search verdict, over-trump minus policy alternative (score): {lo_b} {d_lo.mean():+.3f}±{d_lo.std() / n**0.5:.3f}  "
        f"{hi_b} {d_hi.mean():+.3f}±{d_hi.std() / n**0.5:.3f}  corr {np.corrcoef(d_lo, d_hi)[0, 1]:.2f}"
    )
    print(
        f"replicate variance of centered Q, {hi_b}/{lo_b}: {ratio:.2f};  row noise {lo_b} {nv_lo.mean():.2e}  "
        f"{hi_b} local {nv_hi_local.mean():.2e}  {hi_b} class {nv_hi_class.mean():.2e}"
    )
    print(
        f"model error s_u2: class mean {su2_cls.mean():.2e}  cell ({hi_b} labels) {su2_cell_hi:.2e}  cell ({lo_b} labels) {su2_cell_lo:.2e}"
    )
    print(
        f"\n{'scenario':34s} {'gamma':>6s} {'over mass prior->target':>24s} {'target argmax over':>19s}"
    )
    pmass = float(prior[over].sum() / n)
    for label, t, a_m, nv, su2 in (
        (f"{lo_b} (as configured)", t_lo, am_lo, nv_lo, su2_cls),
        (f"{hi_b} at these nodes only", t_hi, am_hi, nv_hi_local, su2_cls),
        (f"{hi_b} for the whole class", t_hi, am_hi, nv_hi_class, su2_cls),
        (
            f"{lo_b} + cell s_u2",
            t_lo,
            am_lo,
            nv_lo,
            torch.full_like(su2_cls, su2_cell_lo),
        ),
        (
            f"{hi_b} class + cell s_u2",
            t_hi,
            am_hi,
            nv_hi_class,
            torch.full_like(su2_cls, su2_cell_hi),
        ),
        (
            "search only (gamma = 1)",
            t_hi,
            am_hi,
            torch.zeros_like(nv_hi_class),
            su2_cls,
        ),
    ):
        a_hat, v_post, g = blend_advantages(t.advantage, a_m, t.has_q, nv, su2)
        tgt, _ = build_tilt_target(
            t.prior,
            a_hat,
            v_post,
            t.masks.bool(),
            kappa=1.0,
            tilt_max=tilt_max,
            eta=eta,
        )
        tm = float(tgt[over].sum() / n)
        am = float(over.gather(1, tgt.argmax(1, keepdim=True)).float().mean())
        print(f"{label:34s} {g.mean():6.2f} {pmass:10.3f} -> {tm:.3f}      {am:12.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser(
        "generate", help="regenerate scanned nodes as corpus rows at several budgets"
    )
    g.add_argument(
        "--scan", required=True, help="scan JSONL (seed, stepIndex per node)"
    )
    g.add_argument(
        "--ckpt",
        required=True,
        help="acting + searching checkpoint (the scanned policy)",
    )
    g.add_argument("--budgets", nargs="*", type=int, default=[256, 512])
    g.add_argument(
        "--variant",
        action="append",
        default=[],
        help="engine variant label:key=value[,...] (keys: iters, leaf_evaluator, "
        "max_depth_play, d_rollout); written as budget_corpus_<label>.pt",
    )
    g.add_argument("--out-dir", required=True)
    g.add_argument(
        "--manifest-iter",
        default=DEFAULT_ITER,
        help="iteration whose corpus config to copy",
    )
    g.add_argument("--workers", type=int, default=2)
    g.add_argument(
        "--limit", type=int, default=0, help="first N games only (smoke runs)"
    )
    tg = sub.add_parser("targets", help="target scenarios on the regenerated rows")
    tg.add_argument("--scan", required=True)
    tg.add_argument(
        "--rows-dir", required=True, help="directory holding budget_corpus_<b>.pt"
    )
    tg.add_argument(
        "--budgets", nargs=2, type=int, default=[256, 512], help="low, high"
    )
    tg.add_argument(
        "--iter-dir", default=DEFAULT_ITER, help="iteration whose target stage to apply"
    )
    tg.add_argument(
        "--encoder-ckpt",
        default="runs/202609_recall_rc/pi/iter2/bidding/final.pt",
        help="the checkpoint --iter-dir's row table was encoded with (its corpus's theta_k)",
    )
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        generate(
            args.scan,
            args.ckpt,
            args.budgets,
            args.out_dir,
            args.manifest_iter,
            args.workers,
            args.limit,
            [parse_variant(s) for s in args.variant],
        )
    else:
        targets(
            args.scan, args.rows_dir, args.budgets, args.iter_dir, args.encoder_ckpt
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
