#!/usr/bin/env python3
"""Fail-lead logit-gap probe across a checkpoint ladder (E7).

Distinguishes "learned but entropy-suppressed" from "not learning" for the
low-vs-fat defender fail-lead preference (Convention_Optimality_202607.md E6:
early defender leads should probe with 0-point fail, not donate 10/11-point
fail). Behavior probes cannot separate the two: a formed logit ordering can be
masked by a high entropy target. This probe reads the ordering directly.

Design (frozen trajectories): ONE driver checkpoint plays every seed greedily
(argmax, RNG-free — the ``/analyze`` ``deterministic=True`` convention), and
the resulting state stream is cached. Every probe checkpoint then replays the
IDENTICAL stream through its own encoder (per-seat recurrent memory rebuilt
exactly as in training: actor encodes on act, all seats observe post-trick
frames) and reports its play-head logits at the frozen node set. Trends across
the ladder are therefore free of both sampling noise and state-distribution
drift; only the networks differ.

A NODE is a play-phase LEAD (current trick empty) at trick 0 or 1 by a
ground-truth defender (not picker, not revealed or secret partner) in a
standard called-ace game (no leaster, no alone), where the legal leads
include at least one ZERO-point fail (7/8/9) and at least one FAT fail
(A=11/10=10). Kings (4 pts) belong to neither class. Per node and checkpoint:

    gap = max logit over zero-point fail leads - max logit over fat fail leads

gap > 0 means the network orders the probe lead above the donation lead at
that state. Summary = median gap, frac(gap>0), and low-class argmax share,
with a cluster (by-seed) bootstrap CI; cells: trick 0/1, C2-context (holds a
legal called-suit fail while the suit is unled) vs not.

Usage (from repo root):

    uv run python -m sheepshead.analysis.fail_lead_logit_probe \\
        --driver runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_7000000.pt \\
        --ckpt 1M=runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_1000000.pt \\
        --ckpt 7M=runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_7000000.pt \\
        --num-seeds 400 --out runs/convention_optimality_202607/fail_lead_logit_ladder.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sheepshead import ACTION_LOOKUP, FAIL, PARTNER_BY_CALLED_ACE, TRUMP_SET, Game
from sheepshead.agent.ppo import load_agent
from sheepshead.game import CARD_POINTS

DEVICE = torch.device("cpu")
FAIL_SET = set(FAIL)
LOW_FAIL = {c for c in FAIL_SET if CARD_POINTS.get(c, 0) == 0}
FAT_FAIL = {c for c in FAIL_SET if CARD_POINTS.get(c, 0) >= 10}

# action id -> card for PLAY actions (1-based ids, logits are 0-indexed).
PLAY_CARD_BY_AID = {
    aid: name[5:] for aid, name in ACTION_LOOKUP.items() if name.startswith("PLAY ")
}


def _called_suit_fail(card: str, called: str) -> bool:
    return card in FAIL_SET and card[-1] == called[-1]


def _called_suit_already_led(game: Game) -> bool:
    """Mirrors scan_called_suit_leads: any completed trick led in the called
    suit (an UNDER lead counts as the called suit)."""
    if not game.called_card:
        return False
    for t in range(game.current_trick):
        leader = game.leaders[t]
        if not leader:
            continue
        lead = game.history[t][leader - 1]
        if not lead:
            continue
        if lead == "U" or _called_suit_fail(lead, game.called_card):
            return True
    return False


def build_trajectories(driver_path: str, seeds: list[int]) -> list[dict]:
    """Phase A: greedy driver plays each seed once; cache the full per-seat
    encode/observe stream plus the frozen node markers."""
    driver = load_agent(driver_path)
    games = []
    for seed in seeds:
        game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=seed)
        driver.reset_recurrent_state()
        stream = []  # ("act"|"obs", pos, state, valid_ids, node_meta|None)
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    state = player.get_state_dict()
                    node = None
                    action_kind = ACTION_LOOKUP.get(next(iter(valid)), "")
                    if (
                        action_kind.startswith("PLAY ")
                        and not game.is_leaster
                        and not game.alone_called
                        and game.called_card
                        and game.current_trick <= 1
                        and all(c == "" for c in game.history[game.current_trick])
                        and player.position != game.picker
                        and player.position != game.partner
                        and not player.is_secret_partner
                    ):
                        lead_cards = [
                            PLAY_CARD_BY_AID[a] for a in valid if a in PLAY_CARD_BY_AID
                        ]
                        low = sorted(c for c in lead_cards if c in LOW_FAIL)
                        fat = sorted(c for c in lead_cards if c in FAT_FAIL)
                        if low and fat:
                            called_opts = [
                                c
                                for c in lead_cards
                                if _called_suit_fail(c, game.called_card)
                            ]
                            node = {
                                "seed": seed,
                                "trickIndex": game.current_trick,
                                "seat": player.position,
                                "relPos": (player.position - game.picker) % 5,
                                "lowOptions": low,
                                "fatOptions": fat,
                                "c2Context": bool(called_opts)
                                and not _called_suit_already_led(game),
                                "underCall": game.is_called_under,
                                "handTrumpCount": sum(
                                    1 for c in player.hand if c in TRUMP_SET
                                ),
                            }
                    action = _greedy_action(driver, player.position, state, valid)
                    if node is not None:
                        node["driverLed"] = ACTION_LOOKUP[action][5:]
                    stream.append(("act", player.position, state, sorted(valid), node))
                    player.act(action)
                    if game.was_trick_just_completed and not game.is_done():
                        for seat in game.players:
                            stream.append(
                                (
                                    "obs",
                                    seat.position,
                                    seat.get_last_trick_state_dict(),
                                    None,
                                    None,
                                )
                            )
                    valid = player.get_valid_action_ids()
        games.append({"seed": seed, "stream": stream})
    return games


def _masked_logits(agent, pos: int, state: dict, valid: list[int]) -> torch.Tensor:
    """One encode + actor forward, updating the agent's per-seat memory the
    same way training and aux_audit do. Returns masked logits (1, A)."""
    memory_in = agent.get_recurrent_memory(pos, device=DEVICE)
    with torch.no_grad():
        enc = agent.encoder.encode_batch(
            [state], memory_in=memory_in.unsqueeze(0), device=DEVICE
        )
        agent.set_recurrent_memory(pos, enc["memory_out"][0])
        mask = (
            agent.get_action_mask(set(valid), agent.action_size).unsqueeze(0).to(DEVICE)
        )
        hand_ids = torch.as_tensor(
            state["hand_ids"], dtype=torch.long, device=DEVICE
        ).view(1, -1)
        _, logits = agent.actor.forward_with_logits(
            enc, mask, hand_ids, agent.encoder.card
        )
    return logits


def _greedy_action(agent, pos: int, state: dict, valid) -> int:
    logits = _masked_logits(agent, pos, state, sorted(valid))
    aid = int(torch.argmax(logits.squeeze(0)).item()) + 1
    if aid not in valid:  # numerical guard, matches aux_audit
        aid = sorted(valid)[0]
    return aid


def probe_checkpoint(ckpt_path: str, games: list[dict]) -> list[dict]:
    """Phase B: replay the frozen streams through one checkpoint; emit a row
    per (node, checkpoint)."""
    agent = load_agent(ckpt_path)
    rows = []
    for g in games:
        agent.reset_recurrent_state()
        for kind, pos, state, valid, node in g["stream"]:
            if kind == "obs":
                agent.observe(state, player_id=pos)
                continue
            if node is None:
                # Memory must advance on every acting step regardless.
                _masked_logits(agent, pos, state, valid)
                continue
            logits = _masked_logits(agent, pos, state, valid).squeeze(0)
            probs = torch.softmax(logits, dim=-1)

            def best(cards):
                aids = [
                    a
                    for a in valid
                    if a in PLAY_CARD_BY_AID and PLAY_CARD_BY_AID[a] in cards
                ]
                lg = max(float(logits[a - 1]) for a in aids)
                pm = sum(float(probs[a - 1]) for a in aids)
                return lg, pm

            low_lg, low_pm = best(set(node["lowOptions"]))
            fat_lg, fat_pm = best(set(node["fatOptions"]))
            arg_aid = int(torch.argmax(logits).item()) + 1
            arg_card = PLAY_CARD_BY_AID.get(arg_aid, "")
            rows.append(
                {
                    **node,
                    "gapLogit": low_lg - fat_lg,
                    "pLowMass": low_pm,
                    "pFatMass": fat_pm,
                    "argmaxCard": arg_card,
                    "argmaxClass": (
                        "low"
                        if arg_card in LOW_FAIL
                        else "fat"
                        if arg_card in FAT_FAIL
                        else "other"
                    ),
                }
            )
    return rows


def _boot_ci(rows, stat, n_boot=2000, seed=20260810):
    """Cluster bootstrap by deal seed; returns (lo, hi) for ``stat(rows)``."""
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    seeds = sorted(by_seed)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(seeds), size=len(seeds), replace=True)
        sample = [r for i in pick for r in by_seed[seeds[i]]]
        vals.append(stat(sample))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(label: str, rows: list[dict]) -> dict:
    med = lambda rs: float(np.median([r["gapLogit"] for r in rs]))  # noqa: E731
    pos = lambda rs: float(np.mean([r["gapLogit"] > 0 for r in rs]))  # noqa: E731
    lo, hi = _boot_ci(rows, med)
    out = {
        "label": label,
        "n": len(rows),
        "medianGap": med(rows),
        "medianGapCI": [lo, hi],
        "fracGapPos": pos(rows),
        "fracArgmaxLow": float(np.mean([r["argmaxClass"] == "low" for r in rows])),
        "fracArgmaxFat": float(np.mean([r["argmaxClass"] == "fat" for r in rows])),
        "cells": {},
    }
    for cell, pred in (
        ("trick0", lambda r: r["trickIndex"] == 0),
        ("trick1", lambda r: r["trickIndex"] == 1),
        ("c2", lambda r: r["c2Context"]),
        ("nonC2", lambda r: not r["c2Context"]),
    ):
        sub = [r for r in rows if pred(r)]
        if sub:
            out["cells"][cell] = {
                "n": len(sub),
                "medianGap": med(sub),
                "fracGapPos": pos(sub),
            }
    print(
        f"{label}: n={out['n']}  median gap {out['medianGap']:+.3f} "
        f"[{lo:+.3f},{hi:+.3f}]  gap>0 {out['fracGapPos']:.1%}  "
        f"argmax low/fat/other {out['fracArgmaxLow']:.1%}/{out['fracArgmaxFat']:.1%}/"
        f"{1 - out['fracArgmaxLow'] - out['fracArgmaxFat']:.1%}",
        flush=True,
    )
    for cell, c in out["cells"].items():
        print(
            f"    {cell:>6}: n={c['n']:4d}  median {c['medianGap']:+.3f}  "
            f"gap>0 {c['fracGapPos']:.1%}",
            flush=True,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--driver",
        required=True,
        help="checkpoint that generates the frozen trajectories",
    )
    ap.add_argument(
        "--ckpt",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="probe checkpoint (repeatable; rows keep the given order)",
    )
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=400)
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    ckpts = []
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"--ckpt needs LABEL=PATH, got {spec!r}")
        ckpts.append((label, path))

    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    print(f"Phase A: driver {args.driver} over {len(seeds)} seeds ...", flush=True)
    games = build_trajectories(args.driver, seeds)
    n_nodes = sum(1 for g in games for e in g["stream"] if e[4] is not None)
    print(f"Frozen node set: {n_nodes} nodes across {len(games)} games", flush=True)

    summaries, all_rows = [], {}
    for label, path in ckpts:
        rows = probe_checkpoint(path, games)
        summaries.append(summarize(label, rows))
        all_rows[label] = rows

    if len(summaries) >= 2:
        first, last = summaries[0], summaries[-1]
        paired = list(zip(all_rows[first["label"]], all_rows[last["label"]]))
        deltas = [
            {"seed": a["seed"], "gapLogit": b["gapLogit"] - a["gapLogit"]}
            for a, b in paired
        ]
        dmed = float(np.median([d["gapLogit"] for d in deltas]))
        lo, hi = _boot_ci(
            deltas, lambda rs: float(np.median([r["gapLogit"] for r in rs]))
        )
        print(
            f"\nEndpoint drift {first['label']} -> {last['label']}: "
            f"median per-node Δgap {dmed:+.3f} [{lo:+.3f},{hi:+.3f}]",
            flush=True,
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "meta": {
                        "driver": args.driver,
                        "ckpts": dict(ckpts),
                        "startSeed": args.start_seed,
                        "numSeeds": args.num_seeds,
                        "lowClass": sorted(LOW_FAIL),
                        "fatClass": sorted(FAT_FAIL),
                    },
                    "summaries": summaries,
                    "rows": all_rows,
                },
                indent=2,
            )
        )
        print(f"Wrote probe ladder -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
