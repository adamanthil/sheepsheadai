#!/usr/bin/env python3
"""Offline supervised bake-off: shared oracle critic vs per-phase experts.

Question (Learning_System_Redesign_202607, post-Phase-A): how much of the
oracle critic's early-node EV gap (e.g. play_lead_t02 0.458 vs a ~0.9
playout-noise ceiling) is *shared-capacity interference* — coherent
late-play gradients shaping one trunk at the expense of the rare/noisy
early strata — versus effective-sample starvation? Interference is fixable
by architecture; starvation is not.

Method: a frozen self-play dataset (stochastic acting from one checkpoint,
oracle observations, empirical discounted return G — identical semantics to
``critic_stratified_ev``), then train from scratch on identical data:

  * shared — one production-shape ``OracleValueNetwork``.
  * moe    — five fresh ``OracleValueNetwork`` instances hard-routed by
             decision phase: pick, partner-call, bury, play tricks 0-2,
             play tricks 3-5 (operator spec 2026-07-21). Routing is
             observable (action head + trick), so this is per-phase heads,
             not learned-gate MoE. Each expert consumes each episode's
             event stream only up to its last routed step (the recurrence
             needs the prefix; truncation keeps compute ~2.5x shared, not
             5x). Precedent: GNU Backgammon per-phase nets, Stockfish NNUE
             material buckets, Suphx per-action-type models.
  * ref    — the checkpoint's own online-trained oracle head, evaluated
             untouched on the same test episodes (no training).

All arms report per-stratum EV on a held-out test split (same strata as
``critic_stratified_ev``). The MoE arm's capacity is 5x the shared arm's —
deliberately unmatched: the production question is "do routed experts beat
the production critic on identical data", and oracle-critic capacity is
deploy-free (the oracle never ships).

Usage (from repo root):

    uv run python -m sheepshead.analysis.diagnostics.oracle_moe_offline \\
        generate --ckpt <ckpt> --episodes 36000 --workers 8 \\
        --out runs/oracle_moe_offline/dataset.pt
    uv run python -m sheepshead.analysis.diagnostics.oracle_moe_offline \\
        train --dataset runs/oracle_moe_offline/dataset.pt --ckpt <ckpt> \\
        --out runs/oracle_moe_offline/results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD

GROUPS = ["pick", "partner", "bury", "play_t02", "play_t3plus"]

# ------------------------------------------------------------------------- #
# Row labeling (mirrors critic_stratified_ev)
# ------------------------------------------------------------------------- #


def _group_of(action_name: str, trick: int) -> int:
    if action_name in ("PICK", "PASS"):
        return 0
    if action_name.startswith("BURY"):
        return 2
    if action_name.startswith("PLAY"):
        return 3 if trick <= 2 else 4
    return 1  # partner call (called-ace/under/alone declarations)


def _strata_of(row: dict) -> list[str]:
    name = ACTIONS[row["action"]]
    out = ["all"]
    if row["is_leaster"]:
        out.append("leaster")
        return out
    if name in ("PICK", "PASS"):
        out.append("pick")
    elif name.startswith("BURY"):
        out.append("bury")
    elif name.startswith("PLAY"):
        early = row["trick"] <= 2
        if row["lead"] and early:
            out.append("play_lead_t02")
            out.append(f"play_lead_t02_{row['role']}")
        elif early:
            out.append("play_follow_t02")
        else:
            out.append("play_t3plus")
    else:
        out.append("partner_call")
    return out


# ------------------------------------------------------------------------- #
# generate: parallel self-play dataset with full event streams
# ------------------------------------------------------------------------- #

_W: dict = {}


def _worker_init(ckpt: str):
    from sheepshead.agent.ppo import load_agent

    torch.set_num_threads(1)
    _W["agent"] = load_agent(ckpt)
    _W["frozen"] = load_agent(ckpt)


def _worker_episodes(task) -> list[dict]:
    from sheepshead.training.pfsp_runtime import play_population_game

    start, count, seed_base = task
    agent, frozen = _W["agent"], _W["frozen"]
    opponents = [SimpleNamespace(agent=frozen)] * 4
    episodes: list[dict] = []
    for e in range(start, start + count):
        random.seed(seed_base + e)
        np.random.seed((seed_base + e) % (2**32))
        torch.manual_seed(seed_base + e)
        mode = PARTNER_BY_JD if e % 2 == 0 else PARTNER_BY_CALLED_ACE
        agent.events = []
        _, events, _, _, _ = play_population_game(
            agent,
            opponents,
            mode,
            training_agent_position=(e % 5) + 1,
            reward_mode="terminal",
            collect_oracle=True,
        )
        agent.store_episode_events(events)
        kinds = [ev["kind"] for ev in agent.events]
        for s, t_end in agent._segments_from_events(kinds):
            steps = agent.events[s : t_end + 1]
            acts = [ev for ev in steps if ev["kind"] == "action"]
            if not acts:
                continue
            g = np.zeros(len(acts))
            acc = 0.0
            for t in range(len(acts) - 1, -1, -1):
                acc = acts[t]["reward"] + agent.gamma * acc
                g[t] = acc
            obs, is_action, g_full, group_full, strata_full = [], [], [], [], []
            ai = 0
            for ev in steps:
                obs.append(ev["oracle_state"])
                if ev["kind"] != "action":
                    is_action.append(False)
                    g_full.append(0.0)
                    group_full.append(-1)
                    strata_full.append([])
                    continue
                st = ev["state"]
                trick_ids = np.asarray(st["trick_card_ids"]).ravel()
                picker_rel = int(st["picker_rel"])
                partner_rel = int(st["partner_rel"])
                if picker_rel == 0:
                    role = "picker"
                elif float(ev.get("secret_partner", 0.0)) > 0.5:
                    role = "secret_partner"
                elif partner_rel == 0:
                    role = "partner"
                else:
                    role = "defender"
                row = {
                    "action": ev["action"],
                    "trick": int(st["current_trick"]),
                    "lead": bool((trick_ids == 0).all()) and bool(st["play_started"]),
                    "role": role,
                    "is_leaster": bool(st["is_leaster"]),
                }
                is_action.append(True)
                g_full.append(float(g[ai]))
                group_full.append(_group_of(ACTIONS[row["action"]], row["trick"]))
                strata_full.append(_strata_of(row))
                ai += 1
            episodes.append(
                {
                    "obs": obs,
                    "is_action": is_action,
                    "g": g_full,
                    "group": group_full,
                    "strata": strata_full,
                }
            )
        agent.events = []
    return episodes


def cmd_generate(args) -> int:
    n_tasks = args.workers * 8
    per = args.episodes // n_tasks
    tasks, start = [], 0
    for i in range(n_tasks):
        count = per + (1 if i < args.episodes - per * n_tasks else 0)
        if count:
            tasks.append((start, count, args.seed * 1_000_003))
            start += count
    t0 = time.time()
    episodes: list[dict] = []
    with Pool(args.workers, initializer=_worker_init, initargs=(args.ckpt,)) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_worker_episodes, tasks)):
            episodes.extend(chunk)
            print(
                f"  task {i + 1}/{len(tasks)}: {len(episodes)} episodes "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    n_rows = sum(sum(ep["is_action"]) for ep in episodes)
    print(f"{len(episodes)} episodes, {n_rows} action rows")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ckpt": args.ckpt,
            "episodes_requested": args.episodes,
            "seed": args.seed,
            "episodes": episodes,
        },
        out,
    )
    print(f"wrote {out}")
    return 0


# ------------------------------------------------------------------------- #
# train: shared vs moe vs ref on identical splits
# ------------------------------------------------------------------------- #


def _batches(indices: list[int], batch_size: int, rng: random.Random | None):
    idx = list(indices)
    if rng is not None:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield idx[i : i + batch_size]


def _forward_batch(net, eps: list[dict], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (values (B,T), target (B,T), mask (B,T)) for loss steps."""
    seqs = [ep["obs"] for ep in eps]
    vals = net.forward_sequences(seqs, device=device)
    B, T = vals.shape
    target = torch.zeros((B, T), device=vals.device)
    mask = torch.zeros((B, T), dtype=torch.bool, device=vals.device)
    for b, ep in enumerate(eps):
        loss_steps = ep.get("_loss_step")  # None = every action step
        for t, (is_a, g) in enumerate(zip(ep["is_action"], ep["g"])):
            if is_a and (loss_steps is None or loss_steps[t]):
                target[b, t] = g
                mask[b, t] = True
    return vals, target, mask


def _masked_mse(vals, target, mask) -> torch.Tensor:
    if not bool(mask.any()):
        return vals.new_zeros(())
    diff = (vals - target) ** 2
    return diff[mask].mean()


def _truncate_for_group(ep: dict, group: int) -> dict | None:
    """Prefix of the episode up to the last group-routed action step, with
    loss restricted to that group's steps. None if the group never occurs."""
    last = -1
    for t, (is_a, gr) in enumerate(zip(ep["is_action"], ep["group"])):
        if is_a and gr == group:
            last = t
    if last < 0:
        return None
    out = {
        "obs": ep["obs"][: last + 1],
        "is_action": ep["is_action"][: last + 1],
        "g": ep["g"][: last + 1],
        "group": ep["group"][: last + 1],
    }
    out["_loss_step"] = [
        bool(a and gr == group)
        for a, gr in zip(out["is_action"], out["group"])
    ]
    return out


def _train_net(
    net,
    train_eps: list[dict],
    val_eps: list[dict],
    device,
    label: str,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[dict, list[float]]:
    opt = torch.optim.Adam(net.param_groups(lr))
    rng = random.Random(seed)
    best_val, best_state, bad, curve = float("inf"), None, 0, []
    for epoch in range(max_epochs):
        t0 = time.time()
        net.train()
        for batch_idx in _batches(list(range(len(train_eps))), batch_size, rng):
            eps = [train_eps[i] for i in batch_idx]
            vals, target, mask = _forward_batch(net, eps, device)
            loss = _masked_mse(vals, target, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
        net.eval()
        with torch.no_grad():
            se, n = 0.0, 0
            for batch_idx in _batches(list(range(len(val_eps))), batch_size, None):
                eps = [val_eps[i] for i in batch_idx]
                vals, target, mask = _forward_batch(net, eps, device)
                if bool(mask.any()):
                    se += float(((vals - target) ** 2)[mask].sum())
                    n += int(mask.sum())
            val_mse = se / max(n, 1)
        curve.append(val_mse)
        print(
            f"    [{label}] epoch {epoch + 1}: val MSE {val_mse:.5f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if val_mse < best_val - 1e-6:
            best_val, bad = val_mse, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad > patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return {"best_val_mse": best_val, "epochs": len(curve)}, curve


def _eval_values(value_fn, eps: list[dict], batch_size: int) -> list[dict]:
    """value_fn(list-of-episodes) -> (B,T) values. Returns flat action rows
    with g, v, strata."""
    rows = []
    with torch.no_grad():
        for i in range(0, len(eps), batch_size):
            chunk = eps[i : i + batch_size]
            vals = value_fn(chunk)
            for b, ep in enumerate(chunk):
                for t, is_a in enumerate(ep["is_action"]):
                    if is_a:
                        rows.append(
                            {
                                "g": ep["g"][t],
                                "v": float(vals[b, t]),
                                "strata": ep["strata"][t],
                                "group": ep["group"][t],
                            }
                        )
    return rows


def _stratum_report(rows: list[dict]) -> dict:
    strata: dict[str, list[dict]] = {}
    for r in rows:
        for s in r["strata"]:
            strata.setdefault(s, []).append(r)
    out = {}
    for s, rs in strata.items():
        g = np.array([r["g"] for r in rs])
        v = np.array([r["v"] for r in rs])
        if len(g) < 20 or float(np.var(g)) < 1e-9:
            out[s] = {"n": len(rs), "ev": None}
            continue
        out[s] = {
            "n": len(rs),
            "sd_g": float(np.std(g)),
            "ev": float(1.0 - np.var(g - v) / np.var(g)),
        }
    return out


ORDER = [
    "all",
    "pick",
    "partner_call",
    "bury",
    "play_lead_t02",
    "play_lead_t02_secret_partner",
    "play_lead_t02_partner",
    "play_lead_t02_defender",
    "play_lead_t02_picker",
    "play_follow_t02",
    "play_t3plus",
    "leaster",
]


def cmd_train(args) -> int:
    from sheepshead.agent.oracle import OracleValueNetwork
    from sheepshead.agent.ppo import device, load_agent

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = torch.load(args.dataset, weights_only=False)
    episodes = data["episodes"]
    test = [ep for i, ep in enumerate(episodes) if i % 10 == 0]
    val = [ep for i, ep in enumerate(episodes) if i % 10 == 1]
    train = [ep for i, ep in enumerate(episodes) if i % 10 >= 2]
    print(
        f"{len(episodes)} episodes: train {len(train)}, val {len(val)}, "
        f"test {len(test)}"
    )

    results: dict = {
        "dataset": args.dataset,
        "source_ckpt": data.get("ckpt"),
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "arms": {},
    }

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    # --- ref: the checkpoint's online-trained oracle, eval only ---------- #
    if "ref" in arms:
        agent = load_agent(data["ckpt"] if args.ckpt is None else args.ckpt)
        ref_net = agent.oracle_critic
        ref_net.eval()
        rows = _eval_values(
            lambda eps: ref_net.forward_sequences(
                [ep["obs"] for ep in eps], device=device
            ),
            test,
            args.batch_size,
        )
        results["arms"]["ref"] = {"test": _stratum_report(rows)}
        print("\n[ref] per-stratum EV (checkpoint oracle, no training):")
        _print_report(results["arms"]["ref"]["test"])

    # --- shared: one production-shape oracle trained from scratch ------- #
    if "shared" in arms:
        print("\n[shared] training…", flush=True)
        net = OracleValueNetwork().to(device)
        fit, curve = _train_net(
            net, train, val, device, "shared", args.lr, args.batch_size,
            args.max_epochs, args.patience, args.seed,
        )
        rows = _eval_values(
            lambda eps: net.forward_sequences(
                [ep["obs"] for ep in eps], device=device
            ),
            test,
            args.batch_size,
        )
        results["arms"]["shared"] = {
            "fit": fit, "val_curve": curve, "test": _stratum_report(rows)
        }
        print("\n[shared] per-stratum EV:")
        _print_report(results["arms"]["shared"]["test"])
        if args.save_nets:
            torch.save(net.state_dict(), Path(args.out).parent / "shared.pt")

    # --- moe: five per-phase experts, hard-routed ------------------------ #
    if "moe" in arms:
        experts: list = []
        moe_fit = {}
        for k, gname in enumerate(GROUPS):
            tr = [e for e in (_truncate_for_group(ep, k) for ep in train) if e]
            va = [e for e in (_truncate_for_group(ep, k) for ep in val) if e]
            n_rows = sum(sum(e["_loss_step"]) for e in tr)
            print(
                f"\n[moe:{gname}] training on {len(tr)} episodes "
                f"({n_rows} routed rows)…",
                flush=True,
            )
            net_k = OracleValueNetwork().to(device)
            fit, curve = _train_net(
                net_k, tr, va, device, f"moe:{gname}", args.lr,
                args.batch_size, args.max_epochs, args.patience,
                args.seed + 1 + k,
            )
            moe_fit[gname] = {"fit": fit, "val_curve": curve, "n_rows": n_rows}
            experts.append(net_k)

        def moe_values(eps):
            vals_out = None
            for k, net_k in enumerate(experts):
                v = net_k.forward_sequences(
                    [ep["obs"] for ep in eps], device=device
                )
                if vals_out is None:
                    vals_out = torch.zeros_like(v)
                for b, ep in enumerate(eps):
                    for t, (is_a, gr) in enumerate(
                        zip(ep["is_action"], ep["group"])
                    ):
                        if is_a and gr == k:
                            vals_out[b, t] = v[b, t]
            return vals_out

        rows = _eval_values(moe_values, test, args.batch_size)
        results["arms"]["moe"] = {"experts": moe_fit, "test": _stratum_report(rows)}
        print("\n[moe] per-stratum EV (routed):")
        _print_report(results["arms"]["moe"]["test"])
        if args.save_nets:
            for gname, net_k in zip(GROUPS, experts):
                torch.save(
                    net_k.state_dict(), Path(args.out).parent / f"moe_{gname}.pt"
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


def _print_report(rep: dict):
    print(f"  {'stratum':<28}{'n':>7}{'EV':>8}")
    for s in ORDER:
        if s in rep:
            ev = rep[s].get("ev")
            print(
                f"  {s:<28}{rep[s]['n']:>7}"
                f"{ev if ev is not None else float('nan'):>8.3f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="parallel self-play dataset")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--episodes", type=int, default=36000)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--seed", type=int, default=20260721)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("train", help="train + evaluate the arms")
    t.add_argument("--dataset", required=True)
    t.add_argument("--ckpt", default=None, help="override ref checkpoint")
    t.add_argument("--arms", default="ref,shared,moe")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--batch-size", type=int, default=48)
    t.add_argument("--max-epochs", type=int, default=15)
    t.add_argument("--patience", type=int, default=2)
    t.add_argument("--seed", type=int, default=20260721)
    t.add_argument("--save-nets", action="store_true")
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
