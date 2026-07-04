#!/usr/bin/env python3
"""One-off P4 validation (NOT committed): bidding-head ISMCTS search.

Confirms the pre-pick determinizer extension (Game._sample_prepick_deal) lets the
ISMCTS teacher search the PICK head, and that PARTNER / BURY heads (picker already
set, existing post-pick determinizer) also search cleanly.

For each head we drive real games (all seats sampled by the network), capture the
FIRST decision of that head — recording forced_public exactly as pfsp_runtime does
(every non-private public action, all seats) — then:

  PICK:    (a) assert every sampled pre-pick determinization is legal — full-deck
               partition, per-seat counts, observer's own hand preserved, empty
               bury/under; (b) run teacher.search and confirm ok + a valid pi'
               over exactly the root's legal action ids.
  PARTNER, BURY:
           run teacher.search and confirm it returns a valid pi' (no KeyError /
           determinizer failure), i.e. the post-pick path already covers them.

A "valid pi'" = nonnegative, sums to ~1, supported only on the root legal set.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from ppo import PPOAgent
from sheepshead import ACTIONS, DECK, Game
from ismcts import ISMCTSTeacher, ISMCTSConfig
from training_utils import get_partner_selection_mode

CKPT = "final_pfsp_swish_ppo.pt"


def _is_private(valid):
    return any(
        ACTIONS[a - 1].startswith("BURY ") or ACTIONS[a - 1].startswith("UNDER ")
        for a in valid
    )


def _head(valid):
    names = [ACTIONS[a - 1] for a in valid]
    if any(n in ("PICK", "PASS") for n in names):
        return "pick"
    if any(n == "ALONE" or n == "JD PARTNER" or n.startswith("CALL ") for n in names):
        return "partner"
    if any(n.startswith("BURY ") or n.startswith("UNDER ") for n in names):
        return "bury"
    return "play"


def collect_node(agent, game, want_head):
    """Drive a game until the FIRST decision of want_head, returning
    (observer, forced_public, snapshot_game) positioned at (not past) that node.
    forced_public mirrors pfsp_runtime: append (seat, action) for every
    non-private public action by any seat, in order. Returns None if the game
    finishes without that head (e.g. leaster has no pick-by-the-eventual-picker)."""
    forced_public = []
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                if not game.is_leaster and _head(valid) == want_head:
                    # Node reached; do NOT act. forced_public holds everything
                    # public before this decision.
                    return player.position, list(forced_public), game

                state = player.get_state_dict()
                action, _, _ = agent.act(
                    state, valid, player.position, deterministic=False
                )
                if not _is_private(valid):
                    forced_public.append((player.position, action))
                player.act(action)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for seat in game.players:
                        agent.observe(
                            seat.get_last_trick_state_dict(), player_id=seat.position
                        )
    return None


def check_prepick_legality(game, deal, observer):
    """Violations (empty list = legal) for a pre-pick redeal."""
    bad = []
    ih = deal["initial_hands"]
    # Counts: every seat 6 pre-pick.
    for s in range(1, 6):
        if len(ih[s]) != len(game.players[s - 1].hand):
            bad.append(
                f"seat {s} dealt {len(ih[s])} != {len(game.players[s - 1].hand)}"
            )
    # Partition: hands + blind == full deck, no dupes.
    allcards = [c for s in range(1, 6) for c in ih[s]] + list(deal["blind"])
    if sorted(allcards) != sorted(DECK):
        bad.append("initial_hands + blind != full deck (or duplicates)")
    if len(deal["blind"]) != 2:
        bad.append(f"blind has {len(deal['blind'])} cards != 2")
    # Observer's own hand preserved exactly.
    if sorted(ih[observer]) != sorted(game.players[observer - 1].initial_hand):
        bad.append("observer dealt hand altered")
    # No picker yet -> no bury/under.
    if deal["bury"] or deal["under_card"] is not None:
        bad.append(f"pre-pick deal has bury={deal['bury']} under={deal['under_card']}")
    return bad


def valid_pi(res):
    """Whether res['pi'] is a proper distribution supported on res['valid']."""
    pi = np.asarray(res["pi"], dtype=np.float64)
    if (pi < -1e-9).any():
        return False, "negative mass"
    if abs(pi.sum() - 1.0) > 1e-4:
        return False, f"sums to {pi.sum():.5f}"
    support = {a for a in range(1, len(pi) + 1) if pi[a - 1] > 0}
    if not support.issubset(set(res["valid"])):
        return False, f"mass off the legal set: {support - set(res['valid'])}"
    return True, ""


def run_head(agent, teacher, head, n_games, seed):
    print(f"\n===== {head.upper()} head =====")
    rng = random.Random(seed)
    legal_checks = legal_fail = 0
    searched = ok = 0
    pi_fail = 0
    g = 0
    found = 0
    while found < n_games and g < n_games * 60:
        mode = get_partner_selection_mode(g)
        game = Game(partner_selection_mode=mode)
        agent.reset_recurrent_state()
        out = collect_node(agent, game, head)
        g += 1
        if out is None:
            continue
        observer, forced_public, gnode = out
        found += 1

        # PICK: assert determinization legality directly.
        if head == "pick":
            for _ in range(8):
                try:
                    d = gnode.sample_determinization(observer, rng)
                except RuntimeError:
                    legal_fail += 1
                    legal_checks += 1
                    continue
                bad = check_prepick_legality(gnode, d, observer)
                legal_checks += 1
                if bad:
                    legal_fail += 1
                    if legal_fail <= 5:
                        print(f"  LEGALITY VIOLATION: {bad[:3]}", flush=True)

        # All heads: run the real teacher search.
        try:
            res = teacher.search(gnode, observer, forced_public, rng)
        except Exception as e:  # noqa: BLE001 - we want to surface ANY failure
            print(f"  SEARCH RAISED ({type(e).__name__}): {e}", flush=True)
            continue
        searched += 1
        good, why = valid_pi(res)
        if not good:
            pi_fail += 1
            print(
                f"  INVALID pi': {why}  (ess={res['ess']:.1f} ok={res['ok']})",
                flush=True,
            )
        if res["ok"]:
            ok += 1

    print(f"  nodes found: {found}/{g} games scanned")
    if head == "pick":
        print(
            f"  determinization legality: {legal_checks - legal_fail}/{legal_checks} legal "
            f"({legal_fail} violations)"
        )
    print(f"  search ran: {searched}/{found} (no exception)")
    print(f"  valid pi': {searched - pi_fail}/{searched}")
    print(f"  ESS>=floor (ok=True): {ok}/{searched}")
    return legal_fail, (searched - pi_fail), searched, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=CKPT)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Loading {args.model} ...")
    agent = PPOAgent(len(ACTIONS))
    agent.load(args.model, load_optimizers=False)

    # Small budgets so the check is fast but exercises every head.
    cfg = ISMCTSConfig(
        iters={"pick": 16, "partner": 16, "bury": 16, "play": 16},
        det_max_tries=400,
        ess_floor=1.0,
    )
    teacher = ISMCTSTeacher(agent, cfg)

    results = {}
    for head in ("pick", "partner", "bury"):
        results[head] = run_head(
            agent, teacher, head, args.games, args.seed + hash(head) % 1000
        )

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    all_ok = True
    for head, (legal_fail, valid_ok, searched, found) in results.items():
        ran = (
            found > 0
            and searched == found
            and (head != "pick" or legal_fail == 0)
            and valid_ok == searched
        )
        all_ok = all_ok and ran
        print(
            f"  {head:8s}: {'PASS' if ran else 'FAIL'}  "
            f"(found={found} searched={searched} valid_pi={valid_ok}"
            + (f" legal_fail={legal_fail}" if head == "pick" else "")
            + ")"
        )
    print(f"\n  P4 bidding-head search: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
