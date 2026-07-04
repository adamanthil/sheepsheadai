#!/usr/bin/env python3
"""Teacher trump-lead falsifier (Arm-B gate).

Question: does the ISMCTS soft teacher — whose determinized rollouts model ALL
seats as pi_theta — endorse or suppress trick-0 defender trump leads relative to
the policy prior? Run on the PRISTINE 30M PPO model, where the prior is healthy
(greedy trump-lead 4.8% / conditional mass 0.3%, the exit_validation baseline).

Why this matters: the run-2 ExIt warm-start (play distill f=0.30 the only
distillation running) made the trick-0 leak 10x WORSE (48.6% greedy). The live
hypothesis is that self-modeled rollout opponents cannot punish an
information-revealing lead — the real cost of leading trump is what sharp
opponents infer and exploit, and pi_theta rollouts don't do that — so the
outcome-grounded search target pi' systematically over-weights trump leads. If
even the HEALTHY teacher's pi' puts substantially more mass on trump leads than
the healthy prior, the current (self-play-rollout) play teacher is biased and
play distillation is expected to hurt until opponents are population-modeled.

Method: stochastic self-play with the model in all five seats (the training
data distribution); at every trick-0 defender lead holding both trump and fail,
record the policy prior's trump mass, then run the production teacher
(default ISMCTSConfig, d_rollout=6 i.e. roll to terminal, as in training at
t_full=1) and record pi''s trump mass. Report the paired delta.

Usage: PYTHONPATH=. .venv/bin/python validation/teacher_trump_lead_audit.py \
           [-m final_pfsp_swish_ppo.pt] [--nodes 150] [--max-games 2000]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch

from ismcts import ISMCTSConfig, ISMCTSTeacher
from ppo import PPOAgent
from sheepshead import ACTIONS, TRUMP, Game
from training_utils import get_partner_selection_mode


def _is_private(valid) -> bool:
    return any(
        ACTIONS[a - 1].startswith("BURY ") or ACTIONS[a - 1].startswith("UNDER ")
        for a in valid
    )


def _trump_mass(p, valid) -> float:
    """Probability mass on trump PLAY actions, over the full action distribution
    restricted to the valid set (p is a length-action_size vector)."""
    return sum(
        float(p[a - 1])
        for a in valid
        if ACTIONS[a - 1].startswith("PLAY ") and ACTIONS[a - 1][5:] in TRUMP
    )


def _load_population_opponents(pop_dir: str, k: int = 4):
    """Load the ``k`` most-trained members of a PFSP population directory
    (jd_agents / called_ace_agents subdirs, <id>.pt + <id>_metadata.json)."""
    candidates = []
    for sub in ("jd_agents", "called_ace_agents"):
        for mf in glob.glob(os.path.join(pop_dir, sub, "*_metadata.json")):
            with open(mf) as f:
                meta = json.load(f)
            candidates.append(
                (
                    int(meta.get("training_episodes", 0)),
                    mf[: -len("_metadata.json")] + ".pt",
                )
            )
    candidates.sort(reverse=True)
    agents = []
    for eps, path in candidates[:k]:
        a = PPOAgent(len(ACTIONS))
        a.load(path, load_optimizers=False)
        agents.append(a)
        print(f"  opponent: {os.path.basename(path)} (eps={eps:,})")
    if len(agents) < k:
        raise SystemExit(f"only {len(agents)} population members in {pop_dir}")
    return agents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="final_pfsp_swish_ppo.pt")
    ap.add_argument("--nodes", type=int, default=150, help="target audit nodes")
    ap.add_argument("--max-games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--opponents-dir",
        default=None,
        help="PFSP population dir; loads the 4 most-trained members and "
        "population-grounds the teacher's non-observer seats (the acceptance "
        "test for notebooks/Population_Grounded_Teacher_Plan.md). Default: pure "
        "self-play teacher (the dirty baseline).",
    )
    ap.add_argument(
        "--iters-play",
        type=int,
        default=None,
        help="override play-head search iterations (default: production 96; "
        "384 = the measured deploy-search operating point)",
    )
    ap.add_argument(
        "--only-leak-nodes",
        action="store_true",
        help="audit ONLY nodes where the policy's greedy argmax is a trump "
        "lead (Gate-0-style targeted sampling): measures the deploy-relevant "
        "FLIP RATE — how often search overrides a bad lead — with full power, "
        "instead of diluting leak nodes across all defender leads",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.model} ...")
    agent = PPOAgent(len(ACTIONS))
    agent.load(args.model, load_optimizers=False)

    opponents = None
    if args.opponents_dir:
        print(f"Population-grounding teacher from {args.opponents_dir}:")
        opponents = _load_population_opponents(args.opponents_dir, k=4)

    cfg = ISMCTSConfig()
    if args.iters_play:
        cfg.iters = dict(cfg.iters, play=args.iters_play)
        print(f"Play-head search iterations: {args.iters_play}")
    teacher = ISMCTSTeacher(agent, cfg)
    det_rng = random.Random(args.seed + 1)

    rows = []  # (prior_mass, pi_mass, prior_argmax_trump, pi_argmax_trump, ess)
    q_gaps = []  # max root-Q over trump leads minus max root-Q over fail leads
    q_best_trump = []  # 1 if the best-Q root action is a trump lead
    # Trump mass of the visit-count target re-sharpened at lower temperatures
    # (pi_tau(a) ∝ N(a)^(1/tau)): quantifies how much of the injected mass the
    # cheap fix (sharpen the distill target) removes, with no extra search.
    taus = (1.0, 0.5, 0.25)
    tau_masses = {t: [] for t in taus}
    aborted = 0
    games = 0
    t0 = time.time()

    while len(rows) < args.nodes and games < args.max_games:
        game = Game(partner_selection_mode=get_partner_selection_mode(games))
        games += 1
        agent.reset_recurrent_state()
        forced_public = []
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    is_t0_def_lead = (
                        game.play_started
                        and not game.is_leaster
                        and game.cards_played == 0
                        and game.current_trick == 0
                        and game.leader == player.position
                        and not (
                            player.is_picker
                            or player.is_partner
                            or player.is_secret_partner
                        )
                        and any(c in TRUMP for c in player.hand)
                        and any(c not in TRUMP for c in player.hand)
                    )
                    if is_t0_def_lead and len(rows) < args.nodes:
                        saved_mem = {
                            pid: t.detach().clone()
                            for pid, t in agent._player_memories.items()
                        }
                        probs, _ = agent.get_action_probs_with_logits(
                            player.get_state_dict(), valid, player_id=player.position
                        )
                        agent._player_memories = saved_mem
                        prior = probs[0].detach().cpu().numpy()
                        prior_mass = _trump_mass(prior, valid)
                        if args.only_leak_nodes:
                            pa = max(valid, key=lambda a: prior[a - 1])
                            n_ = ACTIONS[pa - 1]
                            if not (n_.startswith("PLAY ") and n_[5:] in TRUMP):
                                is_t0_def_lead = False
                    if is_t0_def_lead and len(rows) < args.nodes:
                        # Production rollout depth at trick 0 (t_full=1 path):
                        # roll to terminal, as the training loop does.
                        seat_policies = (
                            {
                                s: opponents[i]
                                for i, s in enumerate(
                                    x for x in range(1, 6) if x != player.position
                                )
                            }
                            if opponents
                            else None
                        )
                        res = teacher.search(
                            game,
                            player.position,
                            list(forced_public),
                            det_rng,
                            d_rollout=6,
                            seat_policies=seat_policies,
                        )
                        if res["ok"]:
                            pi = np.asarray(res["pi"], dtype=np.float64)
                            pi_mass = _trump_mass(pi, valid)
                            valid_arr = list(valid)
                            prior_argmax = max(valid_arr, key=lambda a: prior[a - 1])
                            pi_argmax = max(valid_arr, key=lambda a: pi[a - 1])

                            def _is_trump_action(a):
                                n = ACTIONS[a - 1]
                                return n.startswith("PLAY ") and n[5:] in TRUMP

                            rows.append(
                                (
                                    prior_mass,
                                    pi_mass,
                                    _is_trump_action(prior_argmax),
                                    _is_trump_action(pi_argmax),
                                    float(res["ess"]),
                                )
                            )
                            root_n = res.get("root_n") or {}
                            if root_n:
                                for t in taus:
                                    w = np.array(
                                        [
                                            max(root_n.get(a, 0.0), 0.0) ** (1.0 / t)
                                            for a in valid_arr
                                        ]
                                    )
                                    if w.sum() > 0:
                                        w = w / w.sum()
                                        tau_masses[t].append(
                                            sum(
                                                float(w[i])
                                                for i, a in enumerate(valid_arr)
                                                if _is_trump_action(a)
                                            )
                                        )
                            # Root-Q comparison separates "pi' has an
                            # exploration floor" (FPU + root uniform mix) from
                            # "the teacher genuinely values trump leads".
                            root_q = res.get("root_q") or {}
                            tq = [
                                root_q[a]
                                for a in valid_arr
                                if a in root_q and _is_trump_action(a)
                            ]
                            fq = [
                                root_q[a]
                                for a in valid_arr
                                if a in root_q and not _is_trump_action(a)
                            ]
                            if tq and fq:
                                q_gaps.append(max(tq) - max(fq))
                                q_best_trump.append(1.0 if max(tq) > max(fq) else 0.0)
                            done = len(rows)
                            if done % 25 == 0:
                                print(
                                    f"  {done}/{args.nodes} nodes "
                                    f"({games} games, {time.time() - t0:.0f}s)"
                                )
                        else:
                            aborted += 1
                    a, _, _ = agent.act(
                        player.get_state_dict(),
                        valid,
                        player.position,
                        deterministic=False,
                    )
                    if not _is_private(valid):
                        forced_public.append((player.position, a))
                    player.act(a)
                    valid = player.get_valid_action_ids()
                    if game.was_trick_just_completed:
                        for seat in game.players:
                            agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )

    if not rows:
        print("No audit nodes collected.")
        return

    prior_m = np.array([r[0] for r in rows])
    pi_m = np.array([r[1] for r in rows])
    delta = pi_m - prior_m
    prior_argmax_trump = np.array([r[2] for r in rows])
    pi_argmax_trump = np.array([r[3] for r in rows])
    ess = np.array([r[4] for r in rows])

    n = len(rows)
    print()
    print("=" * 72)
    print(
        f"TEACHER TRUMP-LEAD AUDIT  ({n} trick-0 defender-lead nodes, "
        f"{games} games, ESS-aborts {aborted})"
    )
    print("=" * 72)
    print(
        f"  policy prior trump mass:  {prior_m.mean():.4f} +/- {prior_m.std(ddof=1) / np.sqrt(n):.4f}"
    )
    print(
        f"  teacher pi'  trump mass:  {pi_m.mean():.4f} +/- {pi_m.std(ddof=1) / np.sqrt(n):.4f}"
    )
    print(
        f"  paired delta (pi'-prior): {delta.mean():+.4f} +/- {delta.std(ddof=1) / np.sqrt(n):.4f}"
        f"  ({delta.mean() / (delta.std(ddof=1) / np.sqrt(n)):+.1f} SE)"
    )
    print(
        f"  argmax leads trump:       prior {100 * prior_argmax_trump.mean():.1f}%  ->  pi' {100 * pi_argmax_trump.mean():.1f}%"
    )
    print(f"  mean root ESS:            {ess.mean():.1f}")
    if q_gaps:
        qg = np.array(q_gaps)
        qb = np.array(q_best_trump)
        print(
            f"  root Q gap (best trump - best fail): {qg.mean():+.4f} "
            f"+/- {qg.std(ddof=1) / np.sqrt(len(qg)):.4f}  (n={len(qg)})"
        )
        print(f"  best-Q action is a trump lead:       {100 * qb.mean():.1f}%")
    for t in taus:
        if tau_masses[t]:
            tm = np.array(tau_masses[t])
            print(
                f"  pi'(tau={t}) trump mass:             {tm.mean():.4f} "
                f"+/- {tm.std(ddof=1) / np.sqrt(len(tm)):.4f}"
            )
    print()
    print(
        "  Interpretation: pi' trump mass >> prior trump mass on the HEALTHY"
        " model\n  => the self-play-rollout teacher endorses the leak it was"
        " built to fix\n  (rollout opponents don't punish information-revealing"
        " leads); play\n  distillation from this teacher is expected to recreate"
        " the run-2 leak\n  regression until opponent seats are"
        " population-modeled."
    )


if __name__ == "__main__":
    main()
