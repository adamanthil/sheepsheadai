#!/usr/bin/env python3
"""Diagnostic B — audit the ISMCTS teacher's PICK target by hand strength (run-1 post-mortem).

The PASS->leaster collapse is driven through the pick head, which is searched at
f=1.0 and fully distilled toward pi' ∝ N(a)^(1/tau). This probe asks the teacher
directly: at a PICK/PASS root, what is pi'(PICK) and the search value gap
Q(PICK)-Q(PASS), bucketed by the searcher's hand strength?

Crucially the teacher rolls out assuming ALL seats play the agent's own policy
(self-play), so the teacher built on a given checkpoint reflects THAT checkpoint's
play. We therefore run it on both the pristine 30M reference and the collapsed
30M+45k checkpoint:

  * If even the REFERENCE teacher favors PASS on strong hands (pi'(PICK) < 0.5,
    Q gap < 0): the teacher is miscalibrated against picking from episode 0 -> the
    distillation pull toward PASS was baked in -> the fix must zero/anchor the pick
    head (don't trust the teacher on bidding).
  * If the reference teacher favors PICK on strong hands but the COLLAPSED teacher
    no longer does: the teacher started fine and the collapse is a self-reinforcing
    feedback loop (policy drifted -> self-play rollouts weakened -> teacher followed)
    -> a floor/anchor that prevents the initial drift should suffice.

We walk the pick phase forcing PASS so several seats' pick decisions are sampled per
deal (at growing forced_public), maximizing hand-strength coverage. Search config and
the trick-0 rollout depth (d_rollout = 6 - 0) mirror training exactly.

Run from the repo root:
  PYTHONPATH=. .venv/bin/python validation/teacher_pick_audit.py \
      -m runs/pfsp_exit_warmstart/checkpoints/pfsp_swish_checkpoint_30045000.pt --games 300
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from ismcts import ISMCTSConfig, ISMCTSTeacher
from ppo import PPOAgent
from sheepshead import ACTIONS, Game
from training_utils import estimate_hand_strength_score, get_partner_selection_mode

PICK_ID = ACTIONS.index("PICK") + 1
PASS_ID = ACTIONS.index("PASS") + 1
PICK_ROLLOUT_DEPTH = 6  # trick 0 <= t_full=1 -> 6 - current_trick(0); matches training


def load(model):
    a = PPOAgent(len(ACTIONS))
    a.load(model, load_optimizers=False)
    return a


# Hand-strength bins (estimate_hand_strength_score on the 6-card pre-pick hand:
# Q=3, J=2, other trump=1). The shaped baseline treated >=7 as pickable, >=8 strong.
_BINS = [
    ("<=4", lambda s: s <= 4),
    ("5-6", lambda s: 5 <= s <= 6),
    ("7", lambda s: s == 7),
    ("8-9", lambda s: 8 <= s <= 9),
    (">=10", lambda s: s >= 10),
]


def _bin(s):
    for label, pred in _BINS:
        if pred(s):
            return label
    return "?"


def audit(agent, n_games, seed):
    """Walk each deal's pick phase (forcing PASS to expose subsequent seats); at every
    PICK/PASS root run the teacher and record pi'(PICK), Q(PICK)-Q(PASS), ESS, accept."""
    teacher = ISMCTSTeacher(agent, ISMCTSConfig())
    # rows keyed by bin label -> list of (pick_prob, q_gap, ess, ok)
    rows: dict[str, list[tuple[float, float, float, bool]]] = {
        lbl: [] for lbl, _ in _BINS
    }

    for g in range(n_games):
        gseed = (seed * 1_000_003 + g) % (2**32)
        random.seed(gseed)
        np.random.seed(gseed)
        torch.manual_seed(gseed)
        rng = random.Random(gseed ^ 0x9E3779B9)

        mode = get_partner_selection_mode(g)
        game = Game(partner_selection_mode=mode, seed=gseed)
        agent.reset_recurrent_state()
        forced_public: list[tuple[int, int]] = []

        while not game.is_done():
            # Find the seat currently facing a PICK/PASS choice.
            actor = None
            for player in game.players:
                valid = player.get_valid_action_ids()
                if not valid:
                    continue
                names = [ACTIONS[a - 1] for a in valid]
                if "PICK" in names and "PASS" in names:
                    actor = player
                break  # only the current to-move seat has valid actions
            if actor is None:
                break  # pick phase over (someone picked, leaster, or forced pick)

            res = teacher.search(
                game,
                actor.position,
                list(forced_public),
                rng,
                d_rollout=PICK_ROLLOUT_DEPTH,
            )
            pick_p = float(res["pi"][PICK_ID - 1])
            q_gap = float(
                res["root_q"].get(PICK_ID, 0.0) - res["root_q"].get(PASS_ID, 0.0)
            )
            strength = estimate_hand_strength_score(actor.hand)
            rows[_bin(strength)].append(
                (pick_p, q_gap, float(res["ess"]), bool(res["ok"]))
            )

            # Force PASS so the next seat's pick decision is also sampled.
            actor.act(PASS_ID)
            forced_public.append((actor.position, PASS_ID))

    return rows


def summarize(rows):
    out = {}
    for lbl, _ in _BINS:
        data = rows[lbl]
        if not data:
            out[lbl] = None
            continue
        pp = np.array([d[0] for d in data])
        qg = np.array([d[1] for d in data])
        ess = np.array([d[2] for d in data])
        ok = np.array([d[3] for d in data], dtype=float)
        out[lbl] = {
            "n": len(data),
            "pick_prob": float(pp.mean()),
            "q_gap": float(qg.mean()),
            "ess": float(ess.mean()),
            "accept": float(ok.mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-m", "--model", required=True, help="collapsed ExIt ckpt (challenger)"
    )
    ap.add_argument(
        "-r",
        "--reference",
        default="final_pfsp_swish_ppo.pt",
        help="pristine 30M seed (the warm-start initial state)",
    )
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("DIAGNOSTIC B — teacher pi'(PICK) and value gap by hand strength")
    print(f"  games/model: {args.games}  | pick rollout depth: {PICK_ROLLOUT_DEPTH}")
    print("  pi'(PICK) > 0.5 and Q gap > 0  => teacher wants to PICK")
    print("=" * 78)

    results = {}
    for tag, path in (("reference", args.reference), ("challenger", args.model)):
        print(f"\n{tag} ({path}) — searching ...")
        results[tag] = summarize(audit(load(path), args.games, args.seed))

    hdr = f"  {'hand str':<9}" + "".join(f"{lbl:>22}" for lbl, _ in _BINS)
    for tag in ("reference", "challenger"):
        print("\n" + "-" * 78)
        print(
            f"  {tag.upper()} [{(args.reference if tag == 'reference' else args.model).split('/')[-1]}]"
        )
        print(hdr)
        r = results[tag]

        def fmt(field, pct=False):
            cells = []
            for lbl, _ in _BINS:
                v = r[lbl]
                if v is None:
                    cells.append(f"{'-':>22}")
                elif field == "n":
                    cells.append(f"{v['n']:>22}")
                elif pct:
                    cells.append(f"{v[field] * 100:>21.1f}%")
                else:
                    cells.append(f"{v[field]:>+22.3f}")
            return "".join(cells)

        print(f"  {'n':<9}" + fmt("n"))
        print(f"  {'pi(PICK)':<9}" + fmt("pick_prob", pct=True))
        print(f"  {'Q gap':<9}" + fmt("q_gap"))
        print(f"  {'ESS':<9}" + fmt("ess"))
        print(f"  {'accept':<9}" + fmt("accept", pct=True))

    print("\n" + "=" * 78)
    print(
        "READ: on strong hands (8-9, >=10) does pi'(PICK) exceed 0.5 / Q gap exceed 0?"
    )
    print("  reference favors PICK but challenger doesn't  -> self-reinforcing drift")
    print(
        "  NEITHER favors PICK on strong hands           -> teacher miscalibrated from t0"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
