"""Head-routed duplicate-bridge h2h (CE_Teacher_Design §17.10 lever 2).

Prices the bidding-drift contribution to a distilled checkpoint's h2h
deficit by seating a chimera in the rigorous_eval gauntlet: BIDDING
decisions (pick / partner-call / bury / alone — every non-play head)
come from one checkpoint, PLAY decisions from another. Both underlying
agents advance their own recurrent streams on the identical realized
trajectory (PPO act() folds only the encoded state into memory, never
the chosen action, so the non-chosen agent cannot desync).

Reading, with B = bidding ckpt, P = play ckpt, vs anchor A == B:
  edge(route(B,P)) - edge(P)  ~= EV recovered by undoing P's bidding
  drift; edge(route(B,P)) ~= 0 means play distillation is EV-clean and
  the whole deficit was bidding.

Usage:
  uv run python -m sheepshead.analysis.head_routed_h2h \\
      --bid-ckpt <theta_k.pt> --play-ckpt <distilled.pt> \\
      [--anchor-ckpt <default: bid-ckpt>] [--deals-per-mode 2000]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
from sheepshead.analysis.rigorous_eval import (
    Model,
    ModelRegistry,
    _bootstrap_deal_indices,
    run_gauntlet,
)
from sheepshead.ismcts import infer_head


def is_lead_decision(state) -> bool:
    """A play decision where the acting seat leads the current trick: the
    observation's leader is the seat itself (relative seat 1), so no card
    of this trick has been played yet."""
    return int(state["leader_rel"]) == 1


class HeadRoutedAgent:
    """Routes act() by head: play -> play_agent, everything else ->
    bid_agent. With ``lead_agent`` set, play decisions split once more:
    LEADS (``is_lead_decision``) -> lead_agent, FOLLOWS -> play_agent.
    Every distinct underlying agent acts (and observes) exactly once per
    call, so their per-seat recurrent memories track the same realized
    trajectory; an agent used in two roles is not advanced twice."""

    def __init__(self, bid_agent, play_agent, lead_agent=None):
        self.bid_agent = bid_agent
        self.play_agent = play_agent
        self.lead_agent = lead_agent

    def _agents(self) -> list:
        unique = {}
        for agent in (self.bid_agent, self.play_agent, self.lead_agent):
            if agent is not None:
                unique.setdefault(id(agent), agent)
        return list(unique.values())

    def reset_recurrent_state(self):
        for agent in self._agents():
            agent.reset_recurrent_state()

    def observe(self, state, player_id=None):
        for agent in self._agents():
            agent.observe(state, player_id=player_id)

    def act(self, state, valid_actions, player_id, deterministic=True):
        outs = {
            id(agent): agent.act(
                state, valid_actions, player_id, deterministic=deterministic
            )
            for agent in self._agents()
        }
        if infer_head(valid_actions) != "play":
            return outs[id(self.bid_agent)]
        if self.lead_agent is not None and is_lead_decision(state):
            return outs[id(self.lead_agent)]
        return outs[id(self.play_agent)]


def routed_h2h(
    bid_ckpt: str,
    play_ckpt: str,
    anchor_ckpt: str,
    n_deals_per_mode: int = 2000,
    seed: int = 42,
    n_boot: int = 5000,
    lead_ckpt: str | None = None,
) -> dict:
    """Duplicate-bridge edge of route(bid, play) vs an all-anchor field.
    Mirrors league_progress_eval.h2h_duplicate (same seed pipeline, so
    results are comparable row-for-row with the §17.9 cert numbers).
    With ``lead_ckpt``, play decisions split into leads (lead_ckpt) and
    follows (play_ckpt): the lead-vs-follow EV attribution of a distilled
    checkpoint's edge (CE_Teacher_Design §20.12 stall diagnosis)."""
    registry = ModelRegistry()
    bid = registry.get(Path(bid_ckpt))
    play = registry.get(Path(play_ckpt))
    anchor = registry.get(Path(anchor_ckpt))
    lead = registry.get(Path(lead_ckpt)) if lead_ckpt else None
    label = f"{lead.model_id}>{play.model_id}" if lead else play.model_id
    cand = Model(
        model_id=f"route[{bid.model_id}|{label}]",
        filepath=Path(play_ckpt),
        episodes=None,
        agent=HeadRoutedAgent(bid.agent, play.agent, lead.agent if lead else None),
    )

    seed_rng = random.Random(seed)
    deal_seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(n_deals_per_mode)]
    boot_idx = _bootstrap_deal_indices(
        n_deals_per_mode, n_boot, np.random.default_rng(seed)
    )

    mode_edges = {}
    deal_scores = []
    for mode, name in ((PARTNER_BY_CALLED_ACE, "called"), (PARTNER_BY_JD, "jd")):
        rep = run_gauntlet([cand], [anchor], deal_seeds, mode, boot_idx)[0]
        mode_edges[name] = {"edge": rep.score.mean, "se": rep.score.se}
        deal_scores.append(rep.deal_score)
    edge = (mode_edges["called"]["edge"] + mode_edges["jd"]["edge"]) / 2.0
    se = 0.5 * float(
        np.sqrt(mode_edges["called"]["se"] ** 2 + mode_edges["jd"]["se"] ** 2)
    )
    pooled = np.concatenate(deal_scores)
    return {
        "edge": edge,
        "se": se,
        "win_frac": float(((pooled > 0) + 0.5 * (pooled == 0)).mean()),
        "deviating_frac": float((pooled != 0).mean()),
        "n_deals": 2 * n_deals_per_mode,
        "instrument": "duplicate_bridge_head_routed",
        "bid_ckpt": bid_ckpt,
        "play_ckpt": play_ckpt,
        "lead_ckpt": lead_ckpt,
        "anchor_ckpt": anchor_ckpt,
        "modes": mode_edges,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Head-routed duplicate h2h")
    p.add_argument("--bid-ckpt", required=True)
    p.add_argument("--play-ckpt", required=True)
    p.add_argument("--anchor-ckpt", default=None, help="default: --bid-ckpt")
    p.add_argument(
        "--lead-ckpt",
        default=None,
        help="route LEAD play decisions here; --play-ckpt then serves follows only",
    )
    p.add_argument("--deals-per-mode", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    res = routed_h2h(
        args.bid_ckpt,
        args.play_ckpt,
        args.anchor_ckpt or args.bid_ckpt,
        n_deals_per_mode=args.deals_per_mode,
        seed=args.seed,
        lead_ckpt=args.lead_ckpt,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
