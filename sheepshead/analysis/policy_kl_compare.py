#!/usr/bin/env python3
"""
Policy KL / behavioral-divergence comparison for two Sheepshead PPO checkpoints.

Measures how differently two models *act*, not just how far apart their weights
are. Two checkpoints are run in lockstep over identical self-play trajectories:
one model ("driver") chooses every action so both models see the exact same game
history, and at every decision point both models' masked action distributions are
queried and compared. Recurrent memories are advanced identically for both, so the
comparison is apples-to-apples at each step.

Reported per decision point (summarized as mean / median / p99 / max):
  - KL(A||B) and KL(B||A) in nats over the valid-action distribution
  - total-variation distance  0.5 * sum|p_A - p_B|
  - top-1 probability gap
And aggregate greedy-action (argmax) agreement.

Interpretation: median ~0 with a heavy tail and <100% argmax agreement means the
models are near-identical on typical states but diverge on a minority of
near-decision-boundary states (e.g. the same training run a few PPO updates apart).

Note on --driver: only the driver model chooses actions, so the sampled
trajectories are biased toward the states *that* model visits. For near-identical
models this is negligible, but when comparing two genuinely different policies run
it once with --driver a and once with --driver b to see both regions of state space.

Usage (from repo root):
    PYTHONPATH=. .venv/bin/python analysis/policy_kl_compare.py \
        --model-a runs/.../pfsp_swish_checkpoint_30000000.pt \
        --model-b final_pfsp_swish_ppo.pt \
        --games 400

    # both partner modes (default), 1000 games, JSON summary to a file:
    PYTHONPATH=. .venv/bin/python analysis/policy_kl_compare.py \
        -a model_a.pt -b model_b.pt --games 1000 --json out.json
"""


# Repo-root imports work regardless of invocation directory.

import argparse
import json
import statistics as st

import torch

from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game

_MODE_BY_NAME = {"jd": PARTNER_BY_JD, "called-ace": PARTNER_BY_CALLED_ACE}


def _valid_dist(probs: torch.Tensor, action_ids: list[int]) -> torch.Tensor:
    """Renormalized distribution over the valid actions (action ids are 1-indexed)."""
    idx = torch.as_tensor([a - 1 for a in action_ids], dtype=torch.long)
    p = probs[0, idx].clamp_min(0)
    s = p.sum()
    return p / s if s > 0 else p


def _summary(xs: list[float]) -> dict:
    s = sorted(xs)
    n = len(s)
    return {
        "mean": st.mean(xs),
        "median": st.median(xs),
        "p99": s[min(n - 1, int(0.99 * n))],
        "max": s[-1],
    }


def _fmt(d: dict) -> str:
    return f"mean={d['mean']:.3e}  median={d['median']:.3e}  p99={d['p99']:.3e}  max={d['max']:.3e}"


def compare(
    agent_a: PPOAgent,
    agent_b: PPOAgent,
    *,
    num_games: int,
    modes: list[int],
    seed: int,
    driver: str = "a",
) -> dict:
    """Run lockstep self-play and collect per-decision divergence stats.

    The ``driver`` model (greedy/argmax) chooses every action so both models share
    one trajectory; both models' distributions are still recorded at each step.
    """
    kl_ab: list[float] = []
    kl_ba: list[float] = []
    tvd: list[float] = []
    top1: list[float] = []
    argmax_agree = 0
    n = 0
    eps = 1e-12

    for g in range(num_games):
        mode = modes[g % len(modes)]
        game = Game(partner_selection_mode=mode, seed=seed + g)
        agent_a.reset_recurrent_state()
        agent_b.reset_recurrent_state()

        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    vs = sorted(valid)
                    state = player.get_state_dict()
                    pa, _ = agent_a.get_action_probs_with_logits(
                        state, valid, player.position
                    )
                    pb, _ = agent_b.get_action_probs_with_logits(
                        state, valid, player.position
                    )
                    qa = _valid_dist(pa, vs)
                    qb = _valid_dist(pb, vs)

                    kl_ab.append(
                        float((qa * ((qa + eps).log() - (qb + eps).log())).sum())
                    )
                    kl_ba.append(
                        float((qb * ((qb + eps).log() - (qa + eps).log())).sum())
                    )
                    tvd.append(float(0.5 * (qa - qb).abs().sum()))
                    ia, ib = int(qa.argmax()), int(qb.argmax())
                    argmax_agree += ia == ib
                    top1.append(abs(float(qa.max()) - float(qb.max())))
                    n += 1

                    # Advance the shared trajectory with the driver's greedy action.
                    chosen = ia if driver == "a" else ib
                    player.act(vs[chosen])
                    valid = player.get_valid_action_ids()

                    # Keep both models' recurrent memories in sync at trick end.
                    if game.was_trick_just_completed:
                        for seat in game.players:
                            lt = seat.get_last_trick_state_dict()
                            agent_a.observe(lt, player_id=seat.position)
                            agent_b.observe(lt, player_id=seat.position)

    return {
        "decision_points": n,
        "games": num_games,
        "argmax_agreement": argmax_agree / n if n else float("nan"),
        "argmax_disagree_count": n - argmax_agree,
        "kl_a_given_b": _summary(kl_ab),
        "kl_b_given_a": _summary(kl_ba),
        "total_variation": _summary(tvd),
        "top1_prob_gap": _summary(top1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-a", "--model-a", required=True, help="path to checkpoint A")
    ap.add_argument("-b", "--model-b", required=True, help="path to checkpoint B")
    ap.add_argument(
        "--games", type=int, default=400, help="number of games (default 400)"
    )
    ap.add_argument(
        "--seed", type=int, default=1000, help="base RNG seed (default 1000)"
    )
    ap.add_argument(
        "--partner-mode",
        choices=["jd", "called-ace", "both"],
        default="both",
        help="partner-selection mode to deal (default both, alternating)",
    )
    ap.add_argument(
        "--driver",
        choices=["a", "b"],
        default="a",
        help="which model's greedy action drives the shared trajectory (default a)",
    )
    ap.add_argument("--json", help="optional path to write the summary as JSON")
    args = ap.parse_args()

    if args.partner_mode == "both":
        modes = [PARTNER_BY_CALLED_ACE, PARTNER_BY_JD]
    else:
        modes = [_MODE_BY_NAME[args.partner_mode]]

    agent_a = load_agent(args.model_a)
    agent_b = load_agent(args.model_b)

    result = compare(
        agent_a,
        agent_b,
        num_games=args.games,
        modes=modes,
        seed=args.seed,
        driver=args.driver,
    )

    print(f"A: {args.model_a}")
    print(f"B: {args.model_b}")
    print(f"decision points: {result['decision_points']} over {result['games']} games")
    print("KL(A||B) nats  :", _fmt(result["kl_a_given_b"]))
    print("KL(B||A) nats  :", _fmt(result["kl_b_given_a"]))
    print("total-var dist :", _fmt(result["total_variation"]))
    print("top1-prob |gap|:", _fmt(result["top1_prob_gap"]))
    agree = result["argmax_agreement"]
    print(
        f"argmax agreement: {result['decision_points'] - result['argmax_disagree_count']}"
        f"/{result['decision_points']} = {100 * agree:.4f}%"
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"model_a": args.model_a, "model_b": args.model_b, **result},
                f,
                indent=2,
            )
        print(f"\nwrote summary -> {args.json}")


if __name__ == "__main__":
    main()
