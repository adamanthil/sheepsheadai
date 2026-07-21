#!/usr/bin/env python3
"""Phase-stratified explained variance of both critics + limited-head sanity.

The trainer logs ONE pooled EV pair per update (``ev_oracle``/``ev_limited``,
ppo.py ``_compute_update_targets``). Pooled EV is dominated by late-play
states where a privileged critic is trivially accurate; the gradient-starved
decisions (pick, early leads, rare partner nodes) are a small minority of
steps. This probe answers: does the oracle critic's variance reduction reach
the decision types that matter?

Method: self-play episodes from one checkpoint (stochastic acting, terminal
reward, oracle observations collected — the trainer's own
``play_population_game`` machinery), ``_fill_oracle_values`` for the
privileged values, then per-stratum EV of each critic against the empirical
discounted return G (gamma = agent.gamma, lambda = 1, zero values — identical
to the trainer's diagnostic). Also fits a 2-fold linear
hand-strength -> G baseline at pick nodes: if a one-feature linear probe
beats the limited head there, the limited critic is undertrained, and part of
the pooled oracle-vs-limited EV gap is training quality, not information.

Usage (from repo root):

    uv run python -m sheepshead.analysis.diagnostics.critic_stratified_ev \
        --ckpt runs/league_arch_perceiver-shared-v2/checkpoints/pfsp_perceiver-shared-v2_checkpoint_2000000.pt \
        --episodes 3000 --out runs/convention_erosion_202607/critic_stratified_ev_2000k.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sheepshead import ACTIONS, DECK, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
from sheepshead.agent.ppo import load_agent
from sheepshead.training.pfsp_runtime import play_population_game
from sheepshead.training.reward_shaping import estimate_hand_strength_score


def _stratum(row: dict) -> list[str]:
    """All stratum labels a row belongs to (a row can be in several)."""
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


def _ev(g: np.ndarray, v: np.ndarray) -> float | None:
    if len(g) < 20:
        return None
    var_g = float(np.var(g))
    if var_g < 1e-9:
        return None
    return float(1.0 - np.var(g - v) / var_g)


def collect(agent, frozen, episodes: int, chunk: int) -> list[dict]:
    rows: list[dict] = []
    opponents = [SimpleNamespace(agent=frozen)] * 4
    done_eps = 0
    t0 = time.time()
    while done_eps < episodes:
        n = min(chunk, episodes - done_eps)
        for i in range(n):
            e = done_eps + i
            mode = PARTNER_BY_JD if e % 2 == 0 else PARTNER_BY_CALLED_ACE
            _, events, _, _, _ = play_population_game(
                agent,
                opponents,
                mode,
                training_agent_position=(e % 5) + 1,
                reward_mode="terminal",
                collect_oracle=True,
            )
            agent.store_episode_events(events)
        agent._fill_oracle_values()

        # Extract rows, then drop the buffer before the next chunk.
        ep_actions: list[dict] = []
        for ev in agent.events:
            if ev["kind"] != "action":
                continue
            ep_actions.append(ev)
            if not ev["done"]:
                continue
            rewards = np.array([a["reward"] for a in ep_actions])
            g = np.zeros(len(ep_actions))
            acc = 0.0
            for t in range(len(ep_actions) - 1, -1, -1):
                acc = rewards[t] + agent.gamma * acc
                g[t] = acc
            for t, a in enumerate(ep_actions):
                st = a["state"]
                trick_ids = np.asarray(st["trick_card_ids"]).ravel()
                picker_rel = int(st["picker_rel"])
                partner_rel = int(st["partner_rel"])
                if picker_rel == 0:
                    role = "picker"
                elif float(a.get("secret_partner", 0.0)) > 0.5:
                    role = "secret_partner"
                elif partner_rel == 0:
                    role = "partner"
                else:
                    role = "defender"
                hand = [DECK[i - 1] for i in np.asarray(st["hand_ids"]).ravel() if i > 0]
                rows.append(
                    {
                        "action": a["action"],
                        "g": float(g[t]),
                        "v_lim": float(a["value"]),
                        "v_ora": float(a["value_oracle"]),
                        "trick": int(st["current_trick"]),
                        "lead": bool((trick_ids == 0).all())
                        and bool(st["play_started"]),
                        "role": role,
                        "is_leaster": bool(st["is_leaster"]),
                        "hand_strength": float(estimate_hand_strength_score(hand)),
                    }
                )
            ep_actions = []
        agent.events = []
        done_eps += n
        print(
            f"  {done_eps}/{episodes} episodes, {len(rows)} action rows "
            f"({(time.time() - t0):.0f}s)",
            flush=True,
        )
    return rows


def report(rows: list[dict]) -> dict:
    strata: dict[str, list[dict]] = {}
    for r in rows:
        for s in _stratum(r):
            strata.setdefault(s, []).append(r)

    out = {}
    print(f"\n{'stratum':<28}{'n':>7}{'sd(G)':>8}{'EV_lim':>8}{'EV_ora':>8}{'gap':>7}")
    order = [
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
    for s in order:
        rs = strata.get(s, [])
        if not rs:
            continue
        g = np.array([r["g"] for r in rs])
        ev_l = _ev(g, np.array([r["v_lim"] for r in rs]))
        ev_o = _ev(g, np.array([r["v_ora"] for r in rs]))
        out[s] = {
            "n": len(rs),
            "sd_g": float(np.std(g)),
            "ev_limited": ev_l,
            "ev_oracle": ev_o,
        }
        gap = (ev_o - ev_l) if (ev_o is not None and ev_l is not None) else None
        print(
            f"{s:<28}{len(rs):>7}{np.std(g):>8.3f}"
            f"{ev_l if ev_l is not None else float('nan'):>8.3f}"
            f"{ev_o if ev_o is not None else float('nan'):>8.3f}"
            f"{gap if gap is not None else float('nan'):>7.3f}"
        )

    # Limited-head sanity: 2-fold OOF linear hand-strength baseline at pick.
    pick = strata.get("pick", [])
    if len(pick) >= 100:
        g = np.array([r["g"] for r in pick])
        x = np.array([r["hand_strength"] for r in pick])
        pred = np.zeros_like(g)
        idx = np.arange(len(g))
        for fold in (0, 1):
            tr, te = idx % 2 != fold, idx % 2 == fold
            b, a = np.polyfit(x[tr], g[tr], 1)
            pred[te] = b * x[te] + a
        ev_lin = float(1.0 - np.var(g - pred) / np.var(g))
        out["pick_linear_hand_strength"] = {"n": len(pick), "ev": ev_lin}
        print(
            f"\npick-node linear hand-strength baseline (2-fold OOF): "
            f"EV {ev_lin:.3f}  vs limited head {out['pick']['ev_limited']:.3f}  "
            f"vs oracle {out['pick']['ev_oracle']:.3f}"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = load_agent(args.ckpt)
    if getattr(agent, "oracle_critic", None) is None:
        ap.error(f"{args.ckpt} has no oracle critic head (need an oracle-mode ckpt)")
    frozen = load_agent(args.ckpt)

    print(f"collecting {args.episodes} self-play episodes from {args.ckpt}")
    rows = collect(agent, frozen, args.episodes, args.chunk)
    out = report(rows)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "ckpt": args.ckpt,
                    "episodes": args.episodes,
                    "seed": args.seed,
                    "gamma": agent.gamma,
                    "strata": out,
                },
                indent=2,
            )
        )
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
