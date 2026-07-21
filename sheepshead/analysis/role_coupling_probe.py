#!/usr/bin/env python3
"""Role-coupling logit probe: is "lead trump" one feature or two?

Convention_Erosion_202607 found partner trump-leading and defender
trump-leading CO-MOVING across the league run (behavior-rate r = +0.75, I1)
while their values cut in opposite directions (+0.24 vs −0.22). Behavioral
correlation can't distinguish (a) a genuinely shared/coupled "lead trump"
parameterization from (b) coincident-but-separate drift. This probe measures
it at the POLICY-DISTRIBUTION level on a FIXED node set:

  * Replay fully-scripted games (all 5 seats ScriptedAgent, fixed CRN seeds)
    — trajectories are checkpoint-independent, so every checkpoint is probed
    on the IDENTICAL states.
  * A shadow checkpoint agent never acts; it maintains per-seat recurrent
    memory via ``observe`` on trick completions and, at each eligible lead
    node (both trump and fail legal, tricks 0-2), reports its trump-lead
    probability mass via ``get_action_probs_with_logits`` under
    snapshot/restore (the DecisionProbe pattern; 2026-06-10 double-encode
    lesson).
  * Nodes are grouped secret-partner vs defender. Across the checkpoint
    ladder, the correlation of FIRST DIFFERENCES of the two groups' mean
    trump mass is the coupling statistic: parametric coupling predicts the
    two series move together step-by-step; independent drift predicts ≈ 0.

Usage (from repo root):

    uv run python -m sheepshead.analysis.role_coupling_probe \
        --ckpt-dir runs/league_arch_perceiver-shared-v2/checkpoints \
        --extra 0=runs/league_arch_perceiver-shared-v2/warmstart_perceiver-shared-v2_400k.pt \
        --deals 250 --out runs/convention_erosion_202607/role_coupling.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from sheepshead.analysis.trump_lead_probe import _is_secret_partner, _lead_options
from sheepshead.scripted_agent import ScriptedAgent
from sheepshead import ACTION_LOOKUP, PARTNER_BY_CALLED_ACE, TRUMP_SET, Game

PROBE_SEED = 20260719  # same CRN deal set as the decay curve / lead probes
MAX_TRICK = 2


def probe_checkpoint(shadow, n_deals: int, seed: int) -> dict:
    """Trump-lead prob mass for ``shadow`` at every eligible lead node of the
    scripted replay. Returns {node_key: (group, mass)}."""
    field = ScriptedAgent()
    nodes: dict[str, tuple[str, float]] = {}

    for d in range(n_deals):
        deal_seed = seed * 1_000_003 + d
        game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=deal_seed)
        shadow.reset_recurrent_state()
        field.reset_recurrent_state()
        while not game.is_done() and game.current_trick <= MAX_TRICK:
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    probe_group = None
                    if (
                        game.play_started
                        and not game.is_leaster
                        and game.current_trick <= MAX_TRICK
                        and game.leader == player.position
                        and game.cards_played == 0
                        and not player.is_picker
                        and game.partner != player.position
                    ):
                        trumps, fails = _lead_options(player)
                        if trumps and fails:
                            probe_group = (
                                "partner"
                                if _is_secret_partner(game, player)
                                else "defender"
                            )
                    if probe_group is not None:
                        saved = shadow.snapshot_player_memories()
                        probs, _ = shadow.get_action_probs_with_logits(
                            player.get_state_dict(),
                            valid,
                            player_id=player.position,
                        )
                        shadow.restore_player_memories(saved)
                        p = probs[0].detach().cpu().numpy()
                        mass = 0.0
                        for aid in valid:
                            name = ACTION_LOOKUP[aid]
                            if (
                                name.startswith("PLAY ")
                                and name.split(" ", 1)[1] in TRUMP_SET
                            ):
                                mass += float(p[aid - 1])
                        key = f"{d}:{player.position}:{game.current_trick}"
                        nodes[key] = (probe_group, mass)

                    a, _, _ = field.act(
                        player.get_state_dict(),
                        valid,
                        player.position,
                        deterministic=True,
                    )
                    player.act(a)
                    valid = player.get_valid_action_ids()
                    if game.was_trick_just_completed:
                        for pl in game.players:
                            shadow.observe(
                                pl.get_last_trick_state_dict(), player_id=pl.position
                            )
                            field.observe(
                                pl.get_last_trick_state_dict(), player_id=pl.position
                            )
                if game.is_done() or game.current_trick > MAX_TRICK:
                    break
    return nodes


def _ladder(args) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    if args.ckpt_dir:
        for p in sorted(Path(args.ckpt_dir).glob("*checkpoint_*.pt")):
            m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
            if m:
                entries.append((int(m.group(1)), str(p)))
    for extra in args.extra:
        ep, _, path = extra.partition("=")
        entries.append((int(ep), path))
    entries.sort(key=lambda t: t[0])
    return entries[:: args.stride] if args.stride > 1 else entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--extra", action="append", default=[], help="EPISODE=PATH")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--deals", type=int, default=250)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    entries = _ladder(args)
    if not entries:
        ap.error("no checkpoints found")
    print(f"{len(entries)} ladder entries, {args.deals} scripted deals each")

    from sheepshead.agent.ppo import load_agent

    results = []
    for ep, path in entries:
        t0 = time.time()
        shadow = load_agent(path)
        nodes = probe_checkpoint(shadow, args.deals, args.seed)
        masses = {"partner": [], "defender": []}
        for group, mass in nodes.values():
            masses[group].append(mass)
        row = {
            "episode": ep,
            "ckpt": Path(path).name,
            "partner_n": len(masses["partner"]),
            "partner_mass": float(np.mean(masses["partner"]))
            if masses["partner"]
            else None,
            "defender_n": len(masses["defender"]),
            "defender_mass": float(np.mean(masses["defender"]))
            if masses["defender"]
            else None,
            "nodes": {k: {"group": g, "mass": m} for k, (g, m) in nodes.items()},
        }
        results.append(row)
        print(
            f"[{ep:>9}] partner {row['partner_mass']:.3f} ({row['partner_n']})  "
            f"defender {row['defender_mass']:.3f} ({row['defender_n']})  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    # Coupling statistic: correlation of first differences of the two series.
    ps = np.array([r["partner_mass"] for r in results], dtype=float)
    ds = np.array([r["defender_mass"] for r in results], dtype=float)
    ok = ~(np.isnan(ps) | np.isnan(ds))
    dp, dd = np.diff(ps[ok]), np.diff(ds[ok])
    corr_diff = float(np.corrcoef(dp, dd)[0, 1]) if len(dp) > 2 else None
    corr_level = float(np.corrcoef(ps[ok], ds[ok])[0, 1]) if ok.sum() > 2 else None
    fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
    print(
        f"\ncoupling: corr(levels) = {fmt(corr_level)}  "
        f"corr(first differences) = {fmt(corr_diff)}  (n={ok.sum()} ckpts)"
    )
    print(
        "Reading: diff-corr near +1 = shared parameterization (role-decoupled "
        "credit needed); near 0 = independent drift (variance/SNR fixes can act "
        "per-role)."
    )

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "deals": args.deals,
                "corr_levels": corr_level,
                "corr_first_differences": corr_diff,
                "ladder": results,
            },
            indent=2,
        )
    )
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
