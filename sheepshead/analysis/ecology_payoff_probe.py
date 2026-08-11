#!/usr/bin/env python3
"""In-ecology payoff probe for low-vs-fat defender leads (E8, rung 3).

E7 showed league training BUILT the fat-lead preference against the search
verdict. Two candidate mechanisms: generalization bleed (schmear contexts
shaping shared features) vs IN-ECOLOGY PAYOFF (fat leads genuinely pay
against the actual roster the learner trains with — all existing search
instruments roll out with the training policy on all five seats, so they
would not see this). This probe measures the payoff hypothesis directly.

At every E7 frozen node (same driver, same greedy replay, identical node
set) two counterfactual arms are forced — the driver's best zero-point fail
lead vs its best fat (10/11-point) fail lead — and each arm is rolled to
terminal R times under two ecologies:

    self : the actor model plays all five seats (the ecology every existing
           search/counterfactual instrument assumes);
    pop  : the actor model keeps the acting seat; the other four seats are
           played by league roster members (sampled per node from the
           members directory, uniform, seeded). Roster seats' recurrent
           memories are reconstructed by replaying the prefix observation
           stream for their seat, as if they had sat at the table all game.

Estimands per node: d = mean(actor score | low) - mean(actor score | fat)
under each ecology; pooled with a by-deal cluster bootstrap. The contrast
d_pop - d_self < 0 (fat relatively better in the population) supports the
in-ecology-payoff mechanism; d_pop ~ d_self localizes E7's wrong-side
gradient in generalization bleed / credit assignment instead.

Usage (from repo root):

    uv run python -m sheepshead.analysis.ecology_payoff_probe \\
        --actor runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_7000000.pt \\
        --num-seeds 600 --rollouts 24 \\
        --out runs/convention_optimality_202607/ecology_payoff_e8.json
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from sheepshead import ACTION_IDS, ACTION_LOOKUP, PARTNER_BY_CALLED_ACE, TRUMP_SET, Game
from sheepshead.agent.ppo import load_agent
from sheepshead.analysis.counterfactual_trump_leads import (
    _restore_memory,
    _snapshot_memory,
)
from sheepshead.analysis.fail_lead_logit_probe import (
    FAT_FAIL,
    LOW_FAIL,
    PLAY_CARD_BY_AID,
    _called_suit_already_led,
    _masked_logits,
)

DEVICE = torch.device("cpu")
BASE_RNG_SEED = 20260810


@lru_cache(maxsize=16)
def _member(path: str):
    return load_agent(path)


def _play_out_multi(agents_by_pos: dict, game: Game) -> None:
    """Multi-agent play-out: each seat acts (sampled) with its own agent and
    observes its own post-trick frame — the league-episode convention."""
    while not game.is_done():
        actor = None
        for player in game.players:
            if player.get_valid_action_ids():
                actor = player
                break
        if actor is None:
            break
        pos = actor.position
        agent = agents_by_pos[pos]
        action, _, _ = agent.act(
            actor.get_state_dict(), actor.get_valid_action_ids(), pos
        )
        actor.act(action)
        if game.was_trick_just_completed:
            for seat in game.players:
                agents_by_pos[seat.position].observe(
                    seat.get_last_trick_state_dict(), player_id=seat.position
                )


def _run_arm(
    agents_by_pos: dict,
    snapshots: list,
    node_game: Game,
    seat: int,
    card: str,
    rollouts: int,
) -> list:
    scores = []
    for _ in range(rollouts):
        for agent, snap in snapshots:
            _restore_memory(agent, snap)
        g = copy.deepcopy(node_game)
        g.players[seat - 1].act(ACTION_IDS[f"PLAY {card}"])  # a lead never
        _play_out_multi(agents_by_pos, g)  # completes a trick
        scores.append(int(g.players[seat - 1].get_score()))
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--actor",
        required=True,
        help="training-policy checkpoint (drives replay + acting seat)",
    )
    ap.add_argument(
        "--roster-dir",
        default="runs/league_retention_pg/league/members",
        help="league members directory for population seats",
    )
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=600)
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--limit-nodes", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    roster_paths = sorted(glob.glob(str(Path(args.roster_dir) / "*.pt")))
    if len(roster_paths) < 4:
        ap.error(f"need >=4 roster members in {args.roster_dir}")
    print(f"Roster pool: {len(roster_paths)} members", flush=True)

    driver = load_agent(args.actor)
    rows = []
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        if args.limit_nodes is not None and len(rows) >= args.limit_nodes:
            break
        game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=seed)
        driver.reset_recurrent_state()
        seat_streams: dict[int, list] = {p.position: [] for p in game.players}
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    state = player.get_state_dict()
                    pos = player.position
                    valid_sorted = sorted(valid)
                    action_kind = ACTION_LOOKUP.get(valid_sorted[0], "")
                    node = None
                    if (
                        action_kind.startswith("PLAY ")
                        and not game.is_leaster
                        and not game.alone_called
                        and game.called_card
                        and game.current_trick <= 1
                        and all(c == "" for c in game.history[game.current_trick])
                        and pos != game.picker
                        and pos != game.partner
                        and not player.is_secret_partner
                    ):
                        lead_cards = [
                            PLAY_CARD_BY_AID[a]
                            for a in valid_sorted
                            if a in PLAY_CARD_BY_AID
                        ]
                        low = sorted(c for c in lead_cards if c in LOW_FAIL)
                        fat = sorted(c for c in lead_cards if c in FAT_FAIL)
                        if low and fat:
                            node = {
                                "low": low,
                                "fat": fat,
                                "c2Context": any(
                                    c[-1] == game.called_card[-1] and c not in TRUMP_SET
                                    for c in lead_cards
                                )
                                and not _called_suit_already_led(game),
                            }

                    logits = _masked_logits(driver, pos, state, valid_sorted).squeeze(0)
                    seat_streams[pos].append(state)
                    aid = int(torch.argmax(logits).item()) + 1
                    if aid not in valid:
                        aid = valid_sorted[0]

                    if node is not None and (
                        args.limit_nodes is None or len(rows) < args.limit_nodes
                    ):

                        def best(cards):
                            aids = [
                                a
                                for a in valid_sorted
                                if PLAY_CARD_BY_AID.get(a) in cards
                            ]
                            top = max(aids, key=lambda a: float(logits[a - 1]))
                            return PLAY_CARD_BY_AID[top]

                        low_card, fat_card = (
                            best(set(node["low"])),
                            best(set(node["fat"])),
                        )
                        node_game = copy.deepcopy(game)
                        node_mem = _snapshot_memory(driver)

                        rng = random.Random(BASE_RNG_SEED + seed * 100 + len(rows))
                        member_paths = rng.sample(roster_paths, 4)
                        members = [_member(p) for p in member_paths]
                        other_seats = [s for s in (1, 2, 3, 4, 5) if s != pos]
                        pop_agents = {pos: driver}
                        pop_snaps = [(driver, node_mem)]
                        for s, m in zip(other_seats, members):
                            m.reset_recurrent_state()
                            for st in seat_streams[s]:
                                m.observe(st, player_id=s)
                            pop_agents[s] = m
                            pop_snaps.append((m, _snapshot_memory(m)))

                        torch.manual_seed(BASE_RNG_SEED + seed * 100 + len(rows))
                        self_agents = {s: driver for s in (1, 2, 3, 4, 5)}
                        arms = {}
                        for ecol, agents, snaps in (
                            ("self", self_agents, [(driver, node_mem)]),
                            ("pop", pop_agents, pop_snaps),
                        ):
                            arms[ecol] = {
                                "low": _run_arm(
                                    agents,
                                    snaps,
                                    node_game,
                                    pos,
                                    low_card,
                                    args.rollouts,
                                ),
                                "fat": _run_arm(
                                    agents,
                                    snaps,
                                    node_game,
                                    pos,
                                    fat_card,
                                    args.rollouts,
                                ),
                            }
                        _restore_memory(driver, node_mem)

                        row = {
                            "seed": seed,
                            "trickIndex": game.current_trick,
                            "seat": pos,
                            "lowCard": low_card,
                            "fatCard": fat_card,
                            "c2Context": node["c2Context"],
                            "rosterMembers": [Path(p).name for p in member_paths],
                            "dSelf": float(
                                np.mean(arms["self"]["low"])
                                - np.mean(arms["self"]["fat"])
                            ),
                            "dPop": float(
                                np.mean(arms["pop"]["low"])
                                - np.mean(arms["pop"]["fat"])
                            ),
                            "arms": {
                                e: {
                                    k: [float(np.mean(v)), float(np.std(v, ddof=1))]
                                    for k, v in a.items()
                                }
                                for e, a in arms.items()
                            },
                        }
                        rows.append(row)
                        if len(rows) % 20 == 0:
                            ds = [r["dSelf"] for r in rows]
                            dp = [r["dPop"] for r in rows]
                            print(
                                f"  [{len(rows)}] seed={seed} "
                                f"dSelf mean {np.mean(ds):+.3f}  dPop mean {np.mean(dp):+.3f}",
                                flush=True,
                            )

                    player.act(aid)
                    if game.was_trick_just_completed and not game.is_done():
                        for seat_p in game.players:
                            st = seat_p.get_last_trick_state_dict()
                            seat_streams[seat_p.position].append(st)
                            driver.observe(st, player_id=seat_p.position)
                    valid = player.get_valid_action_ids()

    def _boot(vals_by_seed, stat, n_boot=2000):
        seeds_u = sorted(vals_by_seed)
        rng = np.random.default_rng(BASE_RNG_SEED)
        outs = []
        for _ in range(n_boot):
            pick = rng.choice(len(seeds_u), size=len(seeds_u), replace=True)
            sample = [v for i in pick for v in vals_by_seed[seeds_u[i]]]
            outs.append(stat(sample))
        return float(np.percentile(outs, 2.5)), float(np.percentile(outs, 97.5))

    def summarize(key):
        by_seed: dict[int, list] = {}
        for r in rows:
            by_seed.setdefault(r["seed"], []).append(r[key])
        vals = [v for vs in by_seed.values() for v in vs]
        lo, hi = _boot(by_seed, np.mean)
        print(
            f"{key}: mean {np.mean(vals):+.4f} [{lo:+.4f},{hi:+.4f}]  "
            f"median {np.median(vals):+.4f}  frac>0 {np.mean(np.array(vals) > 0):.1%}",
            flush=True,
        )
        return {
            "mean": float(np.mean(vals)),
            "ci": [lo, hi],
            "median": float(np.median(vals)),
        }

    print(f"\nNodes: {len(rows)}  (rollouts/arm {args.rollouts})", flush=True)
    s_self = summarize("dSelf")
    s_pop = summarize("dPop")
    by_seed_c: dict[int, list] = {}
    for r in rows:
        by_seed_c.setdefault(r["seed"], []).append(r["dPop"] - r["dSelf"])
    cvals = [v for vs in by_seed_c.values() for v in vs]
    lo, hi = _boot(by_seed_c, np.mean)
    print(
        f"contrast dPop-dSelf: mean {np.mean(cvals):+.4f} [{lo:+.4f},{hi:+.4f}]",
        flush=True,
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "meta": {
                        "actor": args.actor,
                        "rosterDir": args.roster_dir,
                        "rosterSize": len(roster_paths),
                        "startSeed": args.start_seed,
                        "numSeeds": args.num_seeds,
                        "rollouts": args.rollouts,
                    },
                    "summary": {
                        "dSelf": s_self,
                        "dPop": s_pop,
                        "contrast": {"mean": float(np.mean(cvals)), "ci": [lo, hi]},
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(f"Wrote -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
