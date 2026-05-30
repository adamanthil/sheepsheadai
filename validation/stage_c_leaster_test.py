"""One-off leaster determinization + search validation (NOT committed).

Leasters (everyone passes; no picker) must be determinizable and searchable: with
the per-trick reward + LEASTER_FINAL_REWARD_BONUS removed, the pass->leaster branch
the bidding EV rides on is only well-valued if leaster PLAY decisions get a teacher
signal. This checks Game._sample_leaster_deal:

  1. determinization legality on a leaster play node (full-deck partition, per-seat
     counts, observer hand preserved, play-revealed voids, empty bury/under, 2-card
     blind never played);
  2. forced-replay reproduces the exact public history (so a fresh Game can rebuild
     the node from the redeal);
  3. teacher.search returns a valid pi' over the leaster play decision (no
     exception, proper distribution on the legal set).
"""

import random

import numpy as np

from ppo import PPOAgent
from sheepshead import Game, ACTIONS, ACTION_IDS, DECK, UNDER_TOKEN, PARTNER_BY_JD, get_card_suit
from ismcts import ISMCTSTeacher, ISMCTSConfig

CKPT = "final_pfsp_swish_ppo.pt"
PASS_ID = ACTIONS.index("PASS") + 1


def make_agent():
    a = PPOAgent(len(ACTIONS), activation="swish")
    a.load(CKPT, load_optimizers=False)
    return a


def drive_to_leaster_play(game, agent, observer, plies_into_play):
    """Force everyone to PASS (-> leaster), then play `plies_into_play` actions
    with the network, recording forced_public, and stop at the observer's next
    play decision. Returns forced_public or None."""
    forced_public = []
    # Bidding: force all passes to enter leaster.
    while not game.play_started and not game.is_leaster:
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                a = PASS_ID if PASS_ID in valid else min(valid)
                forced_public.append((player.position, a))
                player.act(a)
                if game.play_started or game.is_leaster:
                    break
                valid = player.get_valid_action_ids()
            if game.play_started or game.is_leaster:
                break
    if not game.is_leaster:
        return None

    played = 0
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                if player.position == observer and played >= plies_into_play:
                    return forced_public
                a, _, _ = agent.act(player.get_state_dict(), valid, player.position)
                forced_public.append((player.position, a))
                player.act(a)
                played += 1
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for seat in game.players:
                        agent.observe(seat.get_last_trick_state_dict(), player_id=seat.position)
    return None


def check_leaster_legality(game, deal, observer):
    bad = []
    ih = deal["initial_hands"]
    blind = deal["blind"]
    if deal["bury"] or deal["under_card"] is not None:
        bad.append(f"leaster deal has bury={deal['bury']} under={deal['under_card']}")
    if len(blind) != 2:
        bad.append(f"blind has {len(blind)} cards != 2")
    for s in range(1, 6):
        if len(ih[s]) != 6:
            bad.append(f"seat {s} dealt {len(ih[s])} != 6")
    allcards = [c for s in range(1, 6) for c in ih[s]] + list(blind)
    if sorted(allcards) != sorted(DECK):
        bad.append("initial_hands + blind != full deck (or duplicates)")
    if sorted(ih[observer]) != sorted(game.players[observer - 1].initial_hand):
        bad.append("observer dealt hand altered")

    played_by = {s: [] for s in range(1, 6)}
    for t in range(len(game.history)):
        for s in range(1, 6):
            c = game.history[t][s - 1]
            if c and c != UNDER_TOKEN:
                played_by[s].append(c)
    for s in range(1, 6):
        for c in played_by[s]:
            if c not in ih[s]:
                bad.append(f"seat {s} played {c} not in its dealt hand")
    # Blind cards were never played.
    all_played = {c for cards in played_by.values() for c in cards}
    for c in blind:
        if c in all_played:
            bad.append(f"blind card {c} was also played")

    voids = game._play_revealed_voids()
    for s in range(1, 6):
        cur = set(ih[s]) - set(played_by[s])
        for c in cur:
            if get_card_suit(c) in voids[s]:
                bad.append(f"seat {s} holds {c} but is void in {get_card_suit(c)}")
        if len(cur) != len(game.players[s - 1].hand):
            bad.append(f"seat {s} current size {len(cur)} != {len(game.players[s - 1].hand)}")
    return bad


def valid_pi(res):
    pi = np.asarray(res["pi"], dtype=np.float64)
    if (pi < -1e-9).any():
        return False, "negative mass"
    if abs(pi.sum() - 1.0) > 1e-4:
        return False, f"sums to {pi.sum():.5f}"
    support = {a for a in range(1, len(pi) + 1) if pi[a - 1] > 0}
    if not support.issubset(set(res["valid"])):
        return False, "mass off legal set"
    return True, ""


def main():
    random.seed(1)
    agent = make_agent()
    teacher = ISMCTSTeacher(
        agent, ISMCTSConfig(iters={"pick": 8, "partner": 8, "bury": 8, "play": 8},
                            det_max_tries=400, ess_floor=1.0)
    )
    rng = random.Random(7)

    legal_checks = legal_fail = 0
    searched = pi_fail = ok = 0
    found = 0
    g = 0
    observer = 1
    while found < 12 and g < 400:
        game = Game(partner_selection_mode=PARTNER_BY_JD)
        agent.reset_recurrent_state()
        # Vary how deep into the leaster the searched node sits.
        fp = drive_to_leaster_play(game, agent, observer, plies_into_play=(found % 4) + 1)
        g += 1
        if fp is None:
            continue
        found += 1

        for _ in range(8):
            try:
                d = game.sample_determinization(observer, rng)
            except RuntimeError:
                legal_fail += 1
                legal_checks += 1
                continue
            bad = check_leaster_legality(game, d, observer)
            legal_checks += 1
            if bad:
                legal_fail += 1
                if legal_fail <= 5:
                    print(f"  LEGALITY VIOLATION: {bad[:3]}", flush=True)

        try:
            res = teacher.search(game, observer, fp, rng, d_rollout=6)
        except Exception as e:  # noqa: BLE001
            print(f"  SEARCH RAISED ({type(e).__name__}): {e}", flush=True)
            continue
        searched += 1
        good, why = valid_pi(res)
        if not good:
            pi_fail += 1
            print(f"  INVALID pi': {why}", flush=True)
        if res["ok"]:
            ok += 1
        # n_iter > 0 confirms the determinizer actually built leaster worlds.
        if res["n_iter"] == 0:
            print(f"  WARNING: 0 worlds built (determinizer produced nothing)", flush=True)

    print(f"\nleaster play nodes found: {found}")
    print(f"determinization legality: {legal_checks - legal_fail}/{legal_checks} legal "
          f"({legal_fail} violations)")
    print(f"search ran: {searched}/{found}   valid pi': {searched - pi_fail}/{searched}   "
          f"ESS>=floor: {ok}/{searched}")
    ok_all = (found > 0 and legal_fail == 0 and searched == found and pi_fail == 0)
    print(f"\nLEASTER DETERMINIZATION + SEARCH {'PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    main()
