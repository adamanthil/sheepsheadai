#!/usr/bin/env python3
"""Stage B validation for the SO-ISMCTS soft-teacher (`ismcts.ISMCTSTeacher`).

Per the plan (§4 Stage B), the PRIMARY metric is STRENGTH, not point-EV
calibration: the single-deal oracle is too noisy mid-game for a tight per-state
verdict, but the teacher only needs to *rank* actions for the soft target, and a
search-augmented agent that follows that ranking should beat the raw policy.

Two modes (one-off analysis script -- NOT slated for commit):

  trick0  Reproduce the Gate-0 leak correction. At trick-0 DEFENDER lead states
          (seat 1, both a trump and a fail option) compare the teacher's
          conditional trump-lead mass pi'(trump)/(pi'(trump)+pi'(fail)) against
          the raw policy's. The teacher should LOWER it (the leak is EV-negative).
          Reports per-search ESS health.

  h2h     Head-to-head strength. For each deal, play it twice: once with a focal
          seat acting by the ISMCTS teacher on its PLAY decisions (raw policy
          elsewhere and for all other seats), once fully raw. Paired on the deal.
          Reports mean focal-seat score delta (teacher - raw) with SE, plus an
          oracle check: at each searched node, paired true-deal rollouts of the
          teacher's argmax action vs the policy's argmax action.
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import deque

import numpy as np
import torch

import ppo
from ppo import PPOAgent
from ismcts import ISMCTSTeacher, ISMCTSConfig
from sheepshead import ACTION_IDS, ACTIONS, TRUMP, Game
from training_utils import get_partner_selection_mode

DEV = ppo.device


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def snapshot_memory(agent):
    return {pid: t.detach().clone() for pid, t in agent._player_memories.items()}


def restore_memory(agent, snap):
    agent._player_memories = {pid: t.detach().clone() for pid, t in snap.items()}


def _is_private(valid):
    return any(
        ACTIONS[a - 1].startswith("BURY ") or ACTIONS[a - 1].startswith("UNDER ")
        for a in valid
    )


def best_in_class(probs, valid, want_trump):
    best_card, best_p, mass = None, -1.0, 0.0
    for a in valid:
        name = ACTIONS[a - 1]
        if not name.startswith("PLAY "):
            continue
        card = name[5:]
        if (card in TRUMP) != want_trump:
            continue
        p = probs[a - 1]
        mass += p
        if p > best_p:
            best_p, best_card = p, card
    return best_card, mass


def play_out(game, agent):
    """Sampled continuation to terminal with end-of-trick memory propagation."""
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                a, _, _ = agent.act(player.get_state_dict(), valid, player.position)
                player.act(a)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for seat in game.players:
                        agent.observe(
                            seat.get_last_trick_state_dict(), player_id=seat.position
                        )


# ---------------------------------------------------------------------------
# trick0 mode -- Gate-0 leak correction
# ---------------------------------------------------------------------------
def _qualifies_seat1(game):
    p = game.players[0]
    if p.is_picker or p.is_partner or p.is_secret_partner:
        return False
    has_trump = any(c in TRUMP for c in p.hand)
    has_fail = any(c not in TRUMP for c in p.hand)
    return has_trump and has_fail


def collect_trick0(agent, max_games, target, seed, min_raw_trump):
    """Drive the bidding (recording public actions) to each trick-0 lead; keep
    seat-1 defender leads with both classes available whose raw conditional
    trump-lead mass is >= ``min_raw_trump`` (set ~0.5 to focus on the leak)."""
    states = []
    g = 0
    while len(states) < target and g < max_games:
        mode = get_partner_selection_mode(g)
        game = Game(partner_selection_mode=mode)
        agent.reset_recurrent_state()
        forced_public = []
        ok = True
        guard = 0
        while not game.play_started and guard < 200:
            guard += 1
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    a, _, _ = agent.act(player.get_state_dict(), valid, player.position)
                    if not _is_private(valid):
                        forced_public.append((player.position, a))
                    player.act(a)
                    if game.play_started:
                        break
                    valid = player.get_valid_action_ids()
                if game.play_started:
                    break
        g += 1
        if not game.play_started or game.is_leaster:
            continue
        if not _qualifies_seat1(game):
            continue
        leader = game.players[0]
        valid = leader.get_valid_action_ids()
        probs_t, _ = agent.get_action_probs_with_logits(
            leader.get_state_dict(), valid, player_id=1
        )
        probs = probs_t[0].detach().cpu().numpy()
        _, trump_mass = best_in_class(probs, valid, True)
        _, fail_mass = best_in_class(probs, valid, False)
        raw_cond = (
            trump_mass / (trump_mass + fail_mass)
            if (trump_mass + fail_mass) > 0
            else 0.5
        )
        if raw_cond < min_raw_trump:
            continue
        states.append(
            {
                "game": copy.deepcopy(game),
                "mem": snapshot_memory(agent),
                "forced_public": list(forced_public),
                "raw_cond": float(raw_cond),
                "n_trump": sum(1 for c in leader.hand if c in TRUMP),
                "hand": list(leader.hand),
            }
        )
    return states, g


def run_trick0(agent, teacher, args):
    print(
        f"Scanning up to {args.max_games} games for trick-0 seat-1 defender leads "
        f"(min raw trump-cond {args.min_raw_trump:.2f}) ..."
    )
    states, scanned = collect_trick0(
        agent, args.max_games, args.states, args.seed, args.min_raw_trump
    )
    print(f"  collected {len(states)} states (scanned {scanned} games).")
    rng = random.Random(args.seed + 1)
    rows = []
    for i, st in enumerate(states):
        restore_memory(agent, st["mem"])
        res = teacher.search(st["game"], 1, st["forced_public"], rng)
        pi = res["pi"]
        valid = st["game"].players[0].get_valid_action_ids()
        t_mass = sum(
            pi[a - 1]
            for a in valid
            if ACTIONS[a - 1].startswith("PLAY ") and ACTIONS[a - 1][5:] in TRUMP
        )
        f_mass = sum(
            pi[a - 1]
            for a in valid
            if ACTIONS[a - 1].startswith("PLAY ") and ACTIONS[a - 1][5:] not in TRUMP
        )
        teach_cond = t_mass / (t_mass + f_mass) if (t_mass + f_mass) > 0 else 0.5
        dp = teach_cond - st["raw_cond"]
        # Q diagnostics: visit-weighted mean root Q over trump vs fail leads.
        rn, rq = res["root_n"], res["root_q"]
        tn = {a: rn[a] for a in rn if ACTIONS[a - 1][5:] in TRUMP}
        fn = {a: rn[a] for a in rn if a not in tn}
        qt = (
            (sum(rq[a] * rn[a] for a in tn) / sum(tn.values()))
            if sum(tn.values()) > 0
            else float("nan")
        )
        qf = (
            (sum(rq[a] * rn[a] for a in fn) / sum(fn.values()))
            if sum(fn.values()) > 0
            else float("nan")
        )
        rows.append(
            {
                "dp": dp,
                "raw": st["raw_cond"],
                "teach": teach_cond,
                "ess": res["ess"],
                "ok": res["ok"],
                "n_trump": st["n_trump"],
                "qt": qt,
                "qf": qf,
            }
        )
        print(
            f"  [{i + 1}/{len(states)}] ESS={res['ess']:4.1f} ok={int(res['ok'])} "
            f"n_iter={res['n_iter']:3d}  raw_trump={st['raw_cond']:.2f} "
            f"teach_trump={teach_cond:.2f}  dp={dp:+.2f}  "
            f"Qtrump={qt:+.3f} Qfail={qf:+.3f} Ntrump={sum(tn.values()):.1f} "
            f"Nfail={sum(fn.values()):.1f}",
            flush=True,
        )
    _summarize_trick0(rows)


def _summarize_trick0(rows):
    print("\n" + "=" * 72)
    print(f"STAGE B trick-0 leak correction (n={len(rows)})")
    print("=" * 72)
    if not rows:
        print("  no states")
        return
    dp = np.array([r["dp"] for r in rows])
    ess = np.array([r["ess"] for r in rows])
    usable = [r for r in rows if r["ok"]]
    se = dp.std(ddof=1) / np.sqrt(len(dp)) if len(dp) > 1 else float("nan")
    print(f"  mean dp (teacher trump-cond - raw) = {dp.mean():+.3f}  (SE {se:.3f})")
    print(f"  teacher LOWERS trump-lead prob in {np.mean(dp < 0) * 100:.0f}% of states")
    print(
        f"  ESS: mean={ess.mean():.1f}  min={ess.min():.1f}  "
        f"(>= floor in {np.mean([r['ok'] for r in rows]) * 100:.0f}% of states, "
        f"{len(usable)} usable)"
    )
    print("\n-- Verdict --")
    lowered = np.mean(dp < 0)
    print(
        f"  Direction (lowers trump lead, Gate-0 expected ~90%): "
        f"{'PASS' if lowered >= 0.7 else 'WEAK' if lowered >= 0.55 else 'FAIL'} "
        f"({lowered * 100:.0f}%)"
    )


# ---------------------------------------------------------------------------
# h2h mode -- head-to-head strength
# ---------------------------------------------------------------------------
def play_game(
    game, agent, teacher, focal_seat, det_rng, oracle_rows=None, oracle_rollouts=0
):
    """Play a full game. If ``teacher`` is not None, the focal seat acts by the
    ISMCTS teacher's argmax pi' on its PLAY decisions (raw policy for its
    bidding/bury and for every other seat). Returns the focal seat's score."""
    forced_public = []
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                private = _is_private(valid)
                use_search = (
                    teacher is not None
                    and player.position == focal_seat
                    and game.play_started
                    and not game.is_leaster
                    and any(ACTIONS[a - 1].startswith("PLAY ") for a in valid)
                )
                aid = None
                if use_search:
                    # search() is memory-neutral (snapshots/restores internally).
                    res = teacher.search(game, focal_seat, list(forced_public), det_rng)
                    if res["ok"] and res["pi"].sum() > 0:
                        aid = int(np.argmax(res["pi"])) + 1
                        if oracle_rows is not None and oracle_rollouts > 0:
                            _oracle_compare(
                                game,
                                agent,
                                focal_seat,
                                aid,
                                valid,
                                oracle_rollouts,
                                oracle_rows,
                            )
                        # Advance focal memory through this decision (a normal
                        # act would encode the same state).
                        agent.observe(player.get_state_dict(), player_id=focal_seat)
                if aid is None:
                    aid, _, _ = agent.act(
                        player.get_state_dict(), valid, player.position
                    )
                if not private:
                    forced_public.append((player.position, aid))
                player.act(aid)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for seat in game.players:
                        agent.observe(
                            seat.get_last_trick_state_dict(), player_id=seat.position
                        )
    return game.players[focal_seat - 1].get_score()


def _oracle_compare(game, agent, focal, teach_aid, valid, rollouts, rows):
    """Paired TRUE-deal rollouts: teacher argmax action vs policy argmax action.
    Fully memory-neutral (snapshots the agent's memory first, restores last)."""
    mem = snapshot_memory(agent)
    probs_t, _ = agent.get_action_probs_with_logits(
        game.players[focal - 1].get_state_dict(), valid, player_id=focal
    )
    probs = probs_t[0].detach().cpu().numpy()
    pol_aid = max(valid, key=lambda a: probs[a - 1])
    if teach_aid == pol_aid:
        restore_memory(agent, mem)
        return

    def ev(aid):
        sc = []
        for _ in range(rollouts):
            g = copy.deepcopy(game)
            restore_memory(agent, mem)
            g.players[focal - 1].act(aid)
            play_out(g, agent)
            sc.append(g.players[focal - 1].get_score())
        return float(np.mean(sc))

    teach_ev = ev(teach_aid)
    pol_ev = ev(pol_aid)
    restore_memory(agent, mem)
    rows.append({"d": teach_ev - pol_ev, "teach_ev": teach_ev, "pol_ev": pol_ev})


def run_h2h(agent, teacher, args):
    print(
        f"Head-to-head: {args.games} deals, focal seats {args.focal_seats}, "
        f"play-head search (M={teacher.config.iters['play']}) ..."
    )
    det_rng = random.Random(args.seed + 7)
    deltas = []
    oracle_rows = []
    for gi in range(args.games):
        mode = get_partner_selection_mode(gi)
        seed = args.seed * 100003 + gi
        for focal in args.focal_seats:
            # Search game.
            game_s = Game(partner_selection_mode=mode, seed=seed)
            agent.reset_recurrent_state()
            torch.manual_seed(seed)
            s_score = play_game(
                game_s,
                agent,
                teacher,
                focal,
                det_rng,
                oracle_rows if args.oracle_rollouts else None,
                args.oracle_rollouts,
            )
            # Raw game on the same deal.
            game_r = Game(partner_selection_mode=mode, seed=seed)
            agent.reset_recurrent_state()
            torch.manual_seed(seed)
            r_score = play_game(game_r, agent, None, focal, det_rng)
            deltas.append(s_score - r_score)
        if (gi + 1) % max(1, args.games // 20) == 0:
            d = np.array(deltas)
            print(
                f"  [{gi + 1}/{args.games}] mean delta={d.mean():+.3f} "
                f"(n={len(d)})  failures={dict(teacher.fail)}",
                flush=True,
            )
    _summarize_h2h(deltas, oracle_rows)


def _summarize_h2h(deltas, oracle_rows):
    d = np.array(deltas, dtype=np.float64)
    print("\n" + "=" * 72)
    print(f"STAGE B head-to-head strength (n={len(d)} focal-seat games)")
    print("=" * 72)
    if len(d):
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(
            f"  focal-seat score delta (teacher - raw) = {d.mean():+.3f}  (SE {se:.3f})"
        )
        print(f"  teacher >= raw in {np.mean(d >= 0) * 100:.0f}% of paired games")
    if oracle_rows:
        od = np.array([r["d"] for r in oracle_rows])
        ose = od.std(ddof=1) / np.sqrt(len(od)) if len(od) > 1 else float("nan")
        print(
            f"\n  oracle (true-deal) EV of teacher-argmax minus policy-argmax, "
            f"on the {len(od)} nodes where they differ:"
        )
        print(
            f"    mean delta = {od.mean():+.3f}  (SE {ose:.3f})  "
            f"teacher better in {np.mean(od > 0) * 100:.0f}%"
        )
    print("\n-- Verdict --")
    if len(d) > 1:
        se = d.std(ddof=1) / np.sqrt(len(d))
        z = d.mean() / se if se > 0 else 0.0
        print(
            f"  Strength (teacher beats raw): "
            f"{'PASS' if z > 1.0 else 'WEAK' if d.mean() >= 0 else 'FAIL'} "
            f"(z={z:+.2f})"
        )


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model",
        default="pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt",
    )
    ap.add_argument("--mode", choices=["trick0", "h2h"], default="trick0")
    ap.add_argument("--seed", type=int, default=0)
    # trick0
    ap.add_argument("--max-games", type=int, default=20000)
    ap.add_argument("--states", type=int, default=40)
    ap.add_argument(
        "--min-raw-trump",
        type=float,
        default=0.5,
        help="Keep only trick-0 leads whose raw conditional trump-lead "
        "mass is >= this (0.5 focuses on the leak).",
    )
    # h2h
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--focal-seats", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument(
        "--oracle-rollouts",
        type=int,
        default=0,
        help="If >0, paired true-deal rollouts comparing teacher vs "
        "policy argmax at each searched node.",
    )
    # engine knobs
    ap.add_argument(
        "--play-iters",
        type=int,
        default=None,
        help="Override play-head iterations M (default 96).",
    )
    ap.add_argument("--ess-floor", type=float, default=4.0)
    ap.add_argument("--d-rollout", type=int, default=2)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument(
        "--root-explore",
        type=float,
        default=0.25,
        help="Uniform mix into the root prior (counteracts a collapsed "
        "policy starving the better action under PUCT).",
    )
    ap.add_argument(
        "--fpu",
        type=float,
        default=1.0,
        help="First-play-urgency value (normalized Q space) for not-yet-tried actions.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Loading {args.model} (device={DEV}) ...")
    agent = PPOAgent(len(ACTIONS), activation="swish")
    agent.load(args.model, load_optimizers=False)

    cfg = ISMCTSConfig(
        ess_floor=args.ess_floor,
        d_rollout=args.d_rollout,
        tau_target=args.tau,
        root_explore_frac=args.root_explore,
        fpu=args.fpu,
    )
    if args.play_iters is not None:
        cfg.iters["play"] = args.play_iters
    teacher = ISMCTSTeacher(agent, cfg)

    if args.mode == "trick0":
        run_trick0(agent, teacher, args)
    else:
        run_h2h(agent, teacher, args)


if __name__ == "__main__":
    main()
