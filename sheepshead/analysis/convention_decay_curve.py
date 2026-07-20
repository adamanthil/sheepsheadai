#!/usr/bin/env python3
"""Convention behavior rates across a checkpoint ladder (erosion timeline).

One pass per (checkpoint, partner mode) — hero seated in all 5 seats per CRN
deal against a ScriptedAgent field (lineage-free context), deterministic play,
games abandoned after trick 2 — tallying three lead behaviors at once:

  * partner_trump   — secret partner leads trump when a fail lead is legal
                      (the partner convention; ``partner_trump_lead_probe``'s
                      statistic, same node definition and probe seed).
  * c2_called_suit  — a defender leads the called suit while it is unled and a
                      non-called-suit lead is legal (convention C2; called-ace
                      mode only).
  * defender_trump  — a defender leads trump with a fail legal (the C1 leak;
                      persistence CONTROL — expected low and stable while the
                      two conventions above erode or not).

Timestamps the shaped→terminal erosion described in
Convention_Erosion_202607.md: run over the league run's 50k checkpoint ladder
with the selfplay warmstart prepended at episode 0.

Usage (from repo root):

    uv run python -m sheepshead.analysis.convention_decay_curve \
        --ckpt-dir runs/league_arch_perceiver-shared-v2/checkpoints \
        --extra 0=runs/league_arch_perceiver-shared-v2/warmstart_perceiver-shared-v2_400k.pt \
        --deals 400 --out-csv runs/convention_erosion_202607/decay_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from sheepshead.analysis.trump_lead_probe import _is_secret_partner, _lead_options
from sheepshead.scripted_agent import ScriptedAgent
from sheepshead import (
    ACTION_LOOKUP,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    TRUMP_SET,
    Game,
)

PROBE_SEED = 20260719  # same CRN deal set as partner_trump_lead_probe
MAX_TRICK = 2
STATS = ("partner_trump", "c2_called_suit", "defender_trump")


def probe_checkpoint(hero, n_deals: int, partner_mode: int, seed: int) -> dict:
    """Tally all three lead stats for ``hero`` in one pass. Deterministic."""
    field = ScriptedAgent()
    stats = {k: [0, 0] for k in STATS}  # [opportunities, hits]

    for d in range(n_deals):
        deal_seed = seed * 1_000_003 + d
        for hero_seat in range(1, 6):
            game = Game(partner_selection_mode=partner_mode, seed=deal_seed)
            hero.reset_recurrent_state()
            field.reset_recurrent_state()
            while not game.is_done() and game.current_trick <= MAX_TRICK:
                for player in game.players:
                    valid = player.get_valid_action_ids()
                    while valid:
                        is_hero = player.position == hero_seat
                        ag = hero if is_hero else field
                        record: list[tuple[str, set]] = []
                        if (
                            is_hero
                            and game.play_started
                            and not game.is_leaster
                            and game.current_trick <= MAX_TRICK
                            and game.leader == player.position
                            and game.cards_played == 0
                            and not player.is_picker
                            and game.partner != player.position
                        ):
                            trumps, fails = _lead_options(player)
                            secret = _is_secret_partner(game, player)
                            if secret and trumps and fails:
                                record.append(("partner_trump", set(trumps)))
                            if not secret and trumps and fails:
                                record.append(("defender_trump", set(trumps)))
                            if (
                                not secret
                                and game.called_card
                                and not game.was_called_suit_played
                            ):
                                called_fails = {
                                    c for c in fails if c[-1] == game.called_card[-1]
                                }
                                others = (set(trumps) | set(fails)) - called_fails
                                if called_fails and others:
                                    record.append(("c2_called_suit", called_fails))
                        a, _, _ = ag.act(
                            player.get_state_dict(),
                            valid,
                            player.position,
                            deterministic=True,
                        )
                        if record:
                            name = ACTION_LOOKUP[a]
                            card = (
                                name.split(" ", 1)[1]
                                if name.startswith("PLAY ")
                                else None
                            )
                            for stat, hit_set in record:
                                stats[stat][0] += 1
                                if card in hit_set:
                                    stats[stat][1] += 1
                        player.act(a)
                        valid = player.get_valid_action_ids()
                        if game.was_trick_just_completed:
                            for p in game.players:
                                ctrl = hero if p.position == hero_seat else field
                                ctrl.observe(
                                    p.get_last_trick_state_dict(), player_id=p.position
                                )
                    if game.is_done() or game.current_trick > MAX_TRICK:
                        break

    out = {}
    for k, (opp, hit) in stats.items():
        out[k] = {"opportunities": opp, "hits": hit, "rate": hit / opp if opp else None}
    return out


def _ladder(args) -> list[tuple[int, str, object]]:
    """(episode, label, loader-arg) tuples sorted by episode. ``scripted`` rows
    use episode -1."""
    entries: list[tuple[int, str, object]] = []
    if args.ckpt_dir:
        for p in sorted(Path(args.ckpt_dir).glob("*checkpoint_*.pt")):
            m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
            if m:
                entries.append((int(m.group(1)), p.name, str(p)))
    for extra in args.extra:
        ep, _, path = extra.partition("=")
        entries.append((int(ep), Path(path).name, path))
    entries.sort(key=lambda t: t[0])
    if args.stride > 1:
        entries = entries[:: args.stride]
    if args.scripted_anchor:
        entries.insert(0, (-1, "ScriptedAgent", None))
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default=None, help="glob *checkpoint_*.pt here")
    ap.add_argument(
        "--extra",
        action="append",
        default=[],
        help="EPISODE=PATH, repeatable (e.g. the selfplay warmstart at 0)",
    )
    ap.add_argument("--stride", type=int, default=1, help="probe every Nth checkpoint")
    ap.add_argument(
        "--modes",
        default="called",
        help="comma list from {called,jd} (C2 stat is called-mode only)",
    )
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument(
        "--scripted-anchor",
        action="store_true",
        help="prepend a ScriptedAgent row (convention high-anchor self-check)",
    )
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    mode_map = {"called": PARTNER_BY_CALLED_ACE, "jd": PARTNER_BY_JD}
    modes = [(m, mode_map[m]) for m in args.modes.split(",")]
    entries = _ladder(args)
    if not entries:
        ap.error("no checkpoints found; pass --ckpt-dir and/or --extra")
    print(f"{len(entries)} ladder entries x {len(modes)} mode(s), {args.deals} deals")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["episode", "ckpt", "mode"]
    for k in STATS:
        fields += [f"{k}_opps", f"{k}_hits", f"{k}_rate"]
    fields.append("secs")

    from sheepshead.agent.ppo import load_agent

    rows = []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ep, label, path in entries:
            hero = ScriptedAgent() if path is None else load_agent(path)
            for mode_name, mode in modes:
                t0 = time.time()
                r = probe_checkpoint(hero, args.deals, mode, seed=args.seed)
                row = {"episode": ep, "ckpt": label, "mode": mode_name}
                for k in STATS:
                    row[f"{k}_opps"] = r[k]["opportunities"]
                    row[f"{k}_hits"] = r[k]["hits"]
                    row[f"{k}_rate"] = (
                        f"{r[k]['rate']:.4f}" if r[k]["rate"] is not None else ""
                    )
                row["secs"] = f"{time.time() - t0:.0f}"
                w.writerow(row)
                f.flush()
                rows.append(row)
                print(
                    f"[{ep:>9}] {mode_name}: "
                    + "  ".join(
                        f"{k}={row[f'{k}_rate'] or 'n/a'}({row[f'{k}_opps']})"
                        for k in STATS
                    )
                    + f"  ({row['secs']}s)",
                    flush=True,
                )
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump({"seed": args.seed, "deals": args.deals, "rows": rows}, f, indent=2)
    print(f"wrote {out_path} (+.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
