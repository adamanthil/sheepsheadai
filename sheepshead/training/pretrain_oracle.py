#!/usr/bin/env python3
"""Phase 1 of the training program: supervised pretraining of the privileged
(oracle) critic on a frozen policy's self-play (Training_Program_Redesign
§4.2; Learning_System_Redesign §7.1-§7.4).

Why: the league phase uses the oracle's value as its GAE baseline from the
first update. A freshly initialized oracle spends ~20 updates learning the
return scale, and the retention study found that burn-in window is exactly
where earlier league arms lost their conventions. Fitting the oracle offline
on the bootstrap policy's own games (held-out EV 0.51 in the July run, above
the online oracle's plateau) removes the window: ev_oracle read 0.52 from
update 1 of the validated run.

Two commands:

  generate   play N terminal-reward self-play episodes of the frozen
             checkpoint (hero rotating through the seats against frozen
             copies of itself) and store each hero stream's full-information
             observations with gamma-discounted returns and a stratum label.
  pretrain   fit the production ``OracleValueNetwork`` (with its two
             validated aux heads: per-seat team membership, team points with
             bury) to those returns; early stop on validation value-MSE;
             per-stratum explained variance on a test split for the record.

Usage:
  uv run python -m sheepshead.training.pretrain_oracle generate \\
      --ckpt runs/rc/bootstrap/final.pt --episodes 40000 --workers 8 \\
      --gamma 1.0 --out runs/rc/oracle/dataset.pt
  uv run python -m sheepshead.training.pretrain_oracle pretrain \\
      --dataset runs/rc/oracle/dataset.pt --out runs/rc/oracle/oracle_init.pt

Literature: asymmetric actor-critic (Pinto et al. 2017), centralized
critics (Yu et al. 2021), the history-state value the recurrent oracle
estimates (Baisero & Amato 2022); critic pre-fitting before policy
optimization as in Ziegler et al. 2019.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD

STRATA_ORDER = [
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


def strata_of(action: int, trick: int, lead: bool, role: str, is_leaster: bool):
    """Stratum labels of one action row (the critic_stratified_ev cells)."""
    name = ACTIONS[action]
    out = ["all"]
    if is_leaster:
        out.append("leaster")
        return out
    if name in ("PICK", "PASS"):
        out.append("pick")
    elif name.startswith("BURY"):
        out.append("bury")
    elif name.startswith("PLAY"):
        early = trick <= 2
        if lead and early:
            out.append("play_lead_t02")
            out.append(f"play_lead_t02_{role}")
        elif early:
            out.append("play_follow_t02")
        else:
            out.append("play_t3plus")
    else:
        out.append("partner_call")
    return out


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
_W: dict = {}


def _worker_init(init_args):
    from sheepshead.agent.ppo import load_agent

    ckpt, gamma = init_args
    torch.set_num_threads(1)
    _W["agent"] = load_agent(ckpt)
    _W["frozen"] = load_agent(ckpt)
    _W["agent"].gamma = float(gamma)


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
            obs, is_action, g_full, strata_full = [], [], [], []
            ai = 0
            for ev in steps:
                obs.append(ev["oracle_state"])
                if ev["kind"] != "action":
                    is_action.append(False)
                    g_full.append(0.0)
                    strata_full.append([])
                    continue
                st = ev["state"]
                trick_ids = np.asarray(st["trick_card_ids"]).ravel()
                # rel-seat convention: 0 = none/unknown, 1 = SELF.
                if int(st["picker_rel"]) == 1:
                    role = "picker"
                elif float(ev.get("secret_partner", 0.0)) > 0.5:
                    role = "secret_partner"
                elif int(st["partner_rel"]) == 1:
                    role = "partner"
                else:
                    role = "defender"
                is_action.append(True)
                g_full.append(float(g[ai]))
                strata_full.append(
                    strata_of(
                        ev["action"],
                        int(st["current_trick"]),
                        bool((trick_ids == 0).all()) and bool(st["play_started"]),
                        role,
                        bool(st["is_leaster"]),
                    )
                )
                ai += 1
            episodes.append(
                {"obs": obs, "is_action": is_action, "g": g_full, "strata": strata_full}
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
    with get_context("spawn").Pool(
        args.workers, initializer=_worker_init, initargs=((args.ckpt, args.gamma),)
    ) as pool:
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
            "gamma": args.gamma,
            "episodes": episodes,
        },
        out,
    )
    print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
# pretrain
# --------------------------------------------------------------------------- #
def _batches(indices: list[int], batch_size: int, rng: random.Random | None):
    idx = list(indices)
    if rng is not None:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield idx[i : i + batch_size]


def _masked_mse(vals, target, mask) -> torch.Tensor:
    if not bool(mask.any()):
        return vals.new_zeros(())
    return ((vals - target) ** 2)[mask].mean()


def stratum_report(rows: list[dict]) -> dict:
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


def print_report(rep: dict) -> None:
    print(f"  {'stratum':<28}{'n':>7}{'EV':>8}")
    for s in STRATA_ORDER:
        if s in rep:
            ev = rep[s].get("ev")
            print(
                f"  {s:<28}{rep[s]['n']:>7}"
                f"{ev if ev is not None else float('nan'):>8.3f}"
            )


def cmd_pretrain(args) -> int:
    """Train the production headed OracleValueNetwork on a frozen-policy
    dataset and save its state_dict for train_ppo --oracle-init: value MSE
    plus the two aux losses at the limited critic's coefficients, model
    selection on validation value-MSE."""
    from sheepshead.agent.oracle import OracleValueNetwork, team_aux_labels
    from sheepshead.agent.ppo import device

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = torch.load(args.dataset, weights_only=False)
    episodes = data["episodes"]
    test = [ep for i, ep in enumerate(episodes) if i % 10 == 0]
    val = [ep for i, ep in enumerate(episodes) if i % 10 == 1]
    train = [ep for i, ep in enumerate(episodes) if i % 10 >= 2]
    print(
        f"{len(episodes)} episodes (gamma={data.get('gamma')}): "
        f"train {len(train)}, val {len(val)}, test {len(test)}"
    )

    net = OracleValueNetwork(use_aux_heads=True).to(device)
    opt = torch.optim.Adam(net.param_groups(args.lr))
    rng = random.Random(args.seed)

    def batch_forward(eps):
        seqs = [ep["obs"] for ep in eps]
        vals, trunk = net.forward_sequences_full(seqs, device=device)
        B, T = vals.shape
        target = torch.zeros((B, T), device=vals.device)
        mask = torch.zeros((B, T), dtype=torch.bool, device=vals.device)
        for b, ep in enumerate(eps):
            for t, (is_a, g) in enumerate(zip(ep["is_action"], ep["g"])):
                if is_a:
                    target[b, t] = g
                    mask[b, t] = True
        return seqs, vals, trunk, target, mask

    best_val, best_state, bad, curve = float("inf"), None, 0, []
    for epoch in range(args.max_epochs):
        t0 = time.time()
        net.train()
        for batch_idx in _batches(list(range(len(train))), args.batch_size, rng):
            eps = [train[i] for i in batch_idx]
            seqs, vals, trunk, target, mask = batch_forward(eps)
            m_loss, p_loss = net.aux_losses(trunk, seqs, mask)
            loss = (
                _masked_mse(vals, target, mask)
                + args.aux_partner_coeff * m_loss
                + args.aux_points_coeff * p_loss
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
        net.eval()
        with torch.no_grad():
            se, n = 0.0, 0
            for batch_idx in _batches(list(range(len(val))), args.batch_size, None):
                eps = [val[i] for i in batch_idx]
                _, vals, _, target, mask = batch_forward(eps)
                if bool(mask.any()):
                    se += float(((vals - target) ** 2)[mask].sum())
                    n += int(mask.sum())
            val_mse = se / max(n, 1)
        curve.append(val_mse)
        print(
            f"  [pretrain] epoch {epoch + 1}: val MSE {val_mse:.5f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if val_mse < best_val - 1e-6:
            best_val, bad = val_mse, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad > args.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    rows = []
    with torch.no_grad():
        for i in range(0, len(test), args.batch_size):
            chunk = test[i : i + args.batch_size]
            vals = net.forward_sequences([ep["obs"] for ep in chunk], device=device)
            for b, ep in enumerate(chunk):
                for t, is_a in enumerate(ep["is_action"]):
                    if is_a:
                        rows.append(
                            {
                                "g": ep["g"][t],
                                "v": float(vals[b, t]),
                                "strata": ep["strata"][t],
                            }
                        )
    report = stratum_report(rows)
    print("\n[pretrain] per-stratum EV (test):")
    print_report(report)

    n_ok = n_rows = 0
    mae_sum = mae_n = 0.0
    with torch.no_grad():
        for i in range(0, len(test), args.batch_size):
            eps = test[i : i + args.batch_size]
            seqs, vals, trunk, _, mask = batch_forward(eps)
            B, T = vals.shape
            member, team, team_mask = team_aux_labels(seqs, B, T, vals.device)
            tm = mask & team_mask
            if not bool(tm.any()):
                continue
            pred = net.team_membership(trunk).gt(0.0).float()
            n_ok += int(pred.eq(member).all(-1)[tm].sum())
            n_rows += int(tm.sum())
            diff = (net.team_points(trunk) - team).abs() * 120.0
            mae_sum += float(diff[tm].sum())
            mae_n += int(tm.sum()) * 2
    head_metrics = {
        "membership_acc_exact": n_ok / max(n_rows, 1),
        "team_points_mae": mae_sum / max(mae_n, 1),
    }
    print(f"  head metrics: {head_metrics}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out)
    report_path = out.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "gamma": data.get("gamma"),
                "seed": args.seed,
                "best_val_mse": best_val,
                "val_curve": curve,
                "test": report,
                "head_metrics": head_metrics,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out} and {report_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--episodes", type=int, default=40_000)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--seed", type=int, default=20260725)
    g.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="return discount; must match the league phase (1.0)",
    )
    g.add_argument("--out", required=True)
    p = sub.add_parser("pretrain")
    p.add_argument("--dataset", required=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--max-epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument("--aux-partner-coeff", type=float, default=0.1)
    p.add_argument("--aux-points-coeff", type=float, default=0.2)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return cmd_generate(args) if args.cmd == "generate" else cmd_pretrain(args)


if __name__ == "__main__":
    sys.exit(main())
