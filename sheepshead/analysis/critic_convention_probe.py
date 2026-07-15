#!/usr/bin/env python3
"""Critic-gap probe at convention nodes (E5 part 2).

Question: does the critic even REPRESENT the value gap between the convention
lead and its alternative? PPO's early credit signal for a lead is the GAE
bootstrap  Q̂(s,a) ≈ r_trick + γ·V(s_next-decision); if Q̂(conv) − Q̂(alt) ≈ 0
while the realized rollout gap is positive, the critic is blind at these nodes
and learning must come from full-return noise (the documented
small-early-gap mechanism).

Per collected node (self-play, stochastic, engine-state eligibility):

  * V(s) at the node (limited critic; oracle critic too when the checkpoint
    was trained with --critic-mode oracle — Stage-1 arms).
  * For each branch (convention lead vs best alternative), R forced rollouts.
    Each rollout records the realized discounted return from the node (critic
    reward units, as in critic_calibration) AND the bootstrap readout at the
    leader's next decision: r_node_trick + γ·V(next). The oracle bootstrap
    re-encodes the leader's full oracle event stream (action events at own
    decisions, observation events at trick completions — the exact
    _fill_oracle_values protocol) with fresh zero memory.

Aggregates: realized gap, limited-critic bootstrap gap, oracle bootstrap gap
(all conv − alt, mean ± SE), per-node sign agreement with the realized gap,
and Pearson r(critic gap, realized gap).

Pre-registered hypothesis (Convention_Optimality_202607.md): oracle-critic
arms show a larger critic gap at C2 nodes than the limited critic. NOTE for
Stage-1 intermediate checkpoints: the oracle head fresh-initialized at the
400k warmstart, so its training age is (checkpoint episodes − 400k).

Usage:

  uv run python -m sheepshead.analysis.critic_convention_probe \
      --ckpt final_pfsp_swish_ppo.pt --convention c2 --nodes 80 --rollouts 25 \
      --out runs/convention_optimality_202607/critic_gap_final_c2.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from sheepshead.agent import ppo
from sheepshead.agent.ppo import load_agent
from sheepshead.analysis.called_suit_probe import _called_suit_fail
from sheepshead.analysis.critic_calibration import (
    GAMMA,
    policy_and_value,
    restore_memory,
    snapshot_memory,
)
from sheepshead.training.training_utils import RETURN_SCALE, TRICK_POINT_RATIO
from sheepshead import (
    ACTION_IDS,
    ACTIONS,
    FAIL,
    PARTNER_BY_CALLED_ACE,
    TRUMP,
    Game,
)

DEV = ppo.device
_TRUMP_SET = set(TRUMP)
_FAIL_SET = set(FAIL)


def _lead_cards(player) -> list[str]:
    return [
        ACTIONS[a - 1][5:]
        for a in player.get_valid_action_ids()
        if ACTIONS[a - 1].startswith("PLAY ")
        and ACTIONS[a - 1][5:] in _TRUMP_SET | _FAIL_SET
    ]


def _return_from(game, seat_pos: int, from_trick: int) -> float:
    """Node-relative discounted return for a DEFENDER seat, critic units."""
    g = 0.0
    for t in range(from_trick, 6):
        pts = game.trick_points[t]
        winner = game.trick_winners[t]
        sign = -1.0 if winner in (game.picker, game.partner) else 1.0
        r = (pts / TRICK_POINT_RATIO) * sign
        if t == 5:
            r += game.players[seat_pos - 1].get_score() / RETURN_SCALE
        g += GAMMA ** (t - from_trick) * r
    return g


def _trick_reward(game, node_trick: int) -> float:
    pts = game.trick_points[node_trick]
    winner = game.trick_winners[node_trick]
    sign = -1.0 if winner in (game.picker, game.partner) else 1.0
    return (pts / TRICK_POINT_RATIO) * sign


def _oracle_value(agent, events: list[dict]) -> float:
    """Value of the LAST event in a per-seat oracle event stream (fresh zero
    memory, the _fill_oracle_values protocol)."""
    with torch.no_grad():
        vals = agent.oracle_critic.forward_sequences([events], device=DEV)
    return float(vals[0, -1].item())


def _sample_action(probs: np.ndarray, rng: random.Random) -> int:
    ids = np.nonzero(probs > 0)[0]
    w = probs[ids]
    return int(rng.choices(list(ids), weights=list(w))[0]) + 1


class _EligibleNode:
    """Snapshot of one convention decision plus everything both critics need."""

    def __init__(self, game, mem, leader_pos, trick, conv_card, alt_card, group,
                 oracle_prefix):
        self.game = game
        self.mem = mem  # post-node-forward (cf._replay_to_node protocol)
        self.leader = leader_pos
        self.trick = trick
        self.conv_card = conv_card
        self.alt_card = alt_card
        self.group = group  # "agree" | "disagree" (argmax vs convention)
        self.oracle_prefix = oracle_prefix  # leader's event stream incl. node


def _c2_eligibility(game, leader) -> tuple[list[str], list[str]] | None:
    """Convention pool = called-suit fails, alternative pool = other leads;
    None when the node is not C2-eligible. Best-by-policy-prob picking within
    each pool happens at the caller."""
    if (
        game.is_leaster
        or game.alone_called
        or not game.called_card
        or game.is_called_under
        or game.was_called_suit_played
        or leader.is_picker
        or leader.is_partner
        or game.partner == leader.position
        or leader.is_secret_partner
    ):
        return None
    cards = _lead_cards(leader)
    conv = [c for c in cards if _called_suit_fail(c, game.called_card)]
    alt = [c for c in cards if not _called_suit_fail(c, game.called_card)]
    if not conv or not alt:
        return None
    return conv, alt


def _c1_eligibility(game, leader) -> tuple[list[str], list[str]] | None:
    """Convention pool = fail leads, alternative pool = trump leads."""
    if (
        game.is_leaster
        or leader.is_picker
        or leader.is_partner
        or game.partner == leader.position
        or leader.is_secret_partner
        or game.current_trick > 1
    ):
        return None
    cards = _lead_cards(leader)
    conv = [c for c in cards if c in _FAIL_SET]
    alt = [c for c in cards if c in _TRUMP_SET]
    if not conv or not alt:
        return None
    return conv, alt


def collect_nodes(agent, convention: str, n_nodes: int, max_games: int,
                  seed: int, want_oracle: bool) -> list[_EligibleNode]:
    """Stochastic self-play; capture eligible lead nodes with post-forward
    memory snapshots and (if needed) per-seat oracle event streams."""
    eligibility = _c2_eligibility if convention == "c2" else _c1_eligibility
    rng = random.Random(seed)
    nodes: list[_EligibleNode] = []
    for g_idx in range(max_games):
        if len(nodes) >= n_nodes:
            break
        torch.manual_seed(seed * 100003 + g_idx)
        game = Game(
            partner_selection_mode=PARTNER_BY_CALLED_ACE,
            seed=seed * 1_000_003 + g_idx,
        )
        agent.reset_recurrent_state()
        oracle_events: dict[int, list[dict]] = {p: [] for p in range(1, 6)}
        while not game.is_done():
            actor = next(
                (p for p in game.players if p.get_valid_action_ids()), None
            )
            if actor is None:
                break
            pos = actor.position
            state = actor.get_state_dict()
            valid = actor.get_valid_action_ids()
            if want_oracle:
                oracle_events[pos].append(actor.get_oracle_state_dict())

            probs, _v = policy_and_value(agent, state, valid, pid=pos)

            # Node check: a lead by this seat while play is on.
            node_taken = False
            if (
                game.play_started
                and game.leader == pos
                and game.cards_played == 0
                and len(nodes) < n_nodes
            ):
                pools = eligibility(game, actor)
                if pools is not None:
                    conv_pool, alt_pool = pools
                    card_p = {
                        ACTIONS[a - 1][5:]: float(probs[a - 1]) for a in valid
                    }
                    conv_card = max(conv_pool, key=lambda c: card_p.get(c, 0.0))
                    alt_card = max(alt_pool, key=lambda c: card_p.get(c, 0.0))
                    argmax_card = max(card_p, key=card_p.get)
                    nodes.append(
                        _EligibleNode(
                            game=copy.deepcopy(game),
                            mem=snapshot_memory(agent),
                            leader_pos=pos,
                            trick=int(game.current_trick),
                            conv_card=conv_card,
                            alt_card=alt_card,
                            group="agree" if argmax_card in conv_pool else "disagree",
                            oracle_prefix=(
                                copy.deepcopy(oracle_events[pos])
                                if want_oracle
                                else None
                            ),
                        )
                    )
                    node_taken = True

            actor.act(_sample_action(probs, rng))
            if game.was_trick_just_completed:
                for seat in game.players:
                    agent.observe(
                        seat.get_last_trick_state_dict(), player_id=seat.position
                    )
                    if want_oracle:
                        oracle_events[seat.position].append(
                            seat.get_last_trick_oracle_state_dict()
                        )
            if node_taken and len(nodes) >= n_nodes:
                break
    return nodes


def probe_branch(agent, node: _EligibleNode, card: str, R: int, seed: int,
                 want_oracle: bool) -> dict:
    """R forced rollouts of one branch. Returns realized returns and bootstrap
    readouts (limited + oracle) per rollout."""
    realized, boot_lim, boot_orc = [], [], []
    for r_idx in range(R):
        torch.manual_seed(seed ^ (r_idx << 8))
        rng = random.Random(seed ^ (r_idx << 8) ^ 0x5EED)
        g = copy.deepcopy(node.game)
        restore_memory(agent, node.mem)
        orc_events = (
            copy.deepcopy(node.oracle_prefix) if want_oracle else None
        )
        g.players[node.leader - 1].act(ACTION_IDS[f"PLAY {card}"])

        v_next_lim = None
        v_next_orc = None
        while not g.is_done():
            actor = next((p for p in g.players if p.get_valid_action_ids()), None)
            if actor is None:
                break
            pos = actor.position
            state = actor.get_state_dict()
            valid = actor.get_valid_action_ids()
            if pos == node.leader and want_oracle and v_next_orc is None:
                orc_events.append(actor.get_oracle_state_dict())
            probs, v = policy_and_value(agent, state, valid, pid=pos)
            if pos == node.leader and v_next_lim is None:
                v_next_lim = v
                if want_oracle:
                    v_next_orc = _oracle_value(agent, orc_events)
            actor.act(_sample_action(probs, rng))
            if g.was_trick_just_completed:
                for seat in g.players:
                    agent.observe(
                        seat.get_last_trick_state_dict(), player_id=seat.position
                    )
                    if (
                        want_oracle
                        and seat.position == node.leader
                        and v_next_orc is None
                    ):
                        orc_events.append(seat.get_last_trick_oracle_state_dict())

        ret = _return_from(g, node.leader, node.trick)
        realized.append(ret)
        if v_next_lim is None:
            # Leader never acted again (node was their last decision): the
            # bootstrap IS the realized return.
            boot_lim.append(ret)
            if want_oracle:
                boot_orc.append(ret)
        else:
            r0 = _trick_reward(g, node.trick)
            boot_lim.append(r0 + GAMMA * v_next_lim)
            if want_oracle:
                boot_orc.append(r0 + GAMMA * v_next_orc)
    out = {
        "realized": float(np.mean(realized)),
        "realizedSd": float(np.std(realized, ddof=1)) if R > 1 else 0.0,
        "bootLimited": float(np.mean(boot_lim)),
    }
    if want_oracle:
        out["bootOracle"] = float(np.mean(boot_orc))
    return out


def _agg(pairs: list[tuple[float, float]], label: str) -> dict:
    """pairs = per-node (critic_gap, realized_gap)."""
    cg = np.array([p[0] for p in pairs])
    rg = np.array([p[1] for p in pairs])
    n = len(cg)
    se = float(cg.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    sign = float(np.mean(np.sign(cg) == np.sign(rg))) if n else 0.0
    pear = float(np.corrcoef(cg, rg)[0, 1]) if n > 2 else float("nan")
    print(
        f"  {label:<22}: gap {cg.mean():+.4f} (SE {se:.4f})  "
        f"sign-agree {sign:.0%}  r(critic,realized) {pear:+.3f}"
    )
    return {"gapMean": float(cg.mean()), "gapSe": se, "signAgree": sign, "pearson": pear}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--convention", choices=["c1", "c2"], default="c2")
    ap.add_argument("--nodes", type=int, default=80)
    ap.add_argument("--max-games", type=int, default=4000)
    ap.add_argument("--rollouts", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    agent = load_agent(args.ckpt)
    want_oracle = getattr(agent, "oracle_critic", None) is not None
    print(
        f"ckpt={args.ckpt}  critic_mode={'oracle+limited' if want_oracle else 'limited'}"
        f"  convention={args.convention}"
    )

    nodes = collect_nodes(
        agent, args.convention, args.nodes, args.max_games, args.seed, want_oracle
    )
    print(f"collected {len(nodes)} eligible nodes "
          f"({sum(1 for n in nodes if n.group == 'agree')} agree / "
          f"{sum(1 for n in nodes if n.group == 'disagree')} disagree)")
    if not nodes:
        return 1

    rows = []
    for i, node in enumerate(nodes):
        base = args.seed ^ (i << 16)
        conv = probe_branch(agent, node, node.conv_card, args.rollouts, base, want_oracle)
        alt = probe_branch(agent, node, node.alt_card, args.rollouts, base ^ 0xA17, want_oracle)
        row = {
            "trick": node.trick,
            "group": node.group,
            "convCard": node.conv_card,
            "altCard": node.alt_card,
            "realizedGap": conv["realized"] - alt["realized"],
            "bootLimitedGap": conv["bootLimited"] - alt["bootLimited"],
            "conv": conv,
            "alt": alt,
        }
        if want_oracle:
            row["bootOracleGap"] = conv["bootOracle"] - alt["bootOracle"]
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(nodes)}")

    rg = [r["realizedGap"] for r in rows]
    n = len(rg)
    rg_se = float(np.std(rg, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    print("\nGaps are (convention − alternative), critic reward units:")
    print(f"  {'realized (rollouts)':<22}: gap {np.mean(rg):+.4f} (SE {rg_se:.4f})")
    summary = {
        "realized": {"gapMean": float(np.mean(rg)), "gapSe": rg_se},
        "limited": _agg(
            [(r["bootLimitedGap"], r["realizedGap"]) for r in rows], "limited bootstrap"
        ),
    }
    if want_oracle:
        summary["oracle"] = _agg(
            [(r["bootOracleGap"], r["realizedGap"]) for r in rows], "ORACLE bootstrap"
        )
    print(
        "\nRead: a critic that 'sees' the convention shows a bootstrap gap with "
        "the realized gap's sign and a materially nonzero magnitude; ≈0 gap with "
        "a nonzero realized gap = critic-blind (learning rides on full-return noise)."
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "ckpt": args.ckpt,
                        "convention": args.convention,
                        "nodes": len(rows),
                        "rollouts": args.rollouts,
                        "seed": args.seed,
                        "oracle": want_oracle,
                        "gamma": GAMMA,
                    },
                    "summary": summary,
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
