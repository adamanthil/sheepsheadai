"""One-off throughput profiler (NOT committed).

Two workloads, profiled separately so we can see where time actually goes:

  A. PURE GAME: play full games choosing a random legal action each decision via
     the production methods (get_valid_action_ids + act), no encoder/network. This
     isolates Game-logic cost (the part that also runs on every ISMCTS rollout ply
     and every PPO step's game advancement).

  B. ISMCTS SEARCH: profile a handful of real teacher.search() calls (Game logic +
     encoder + network + deepcopy), so we can see the Game-vs-encode-vs-copy split.

Run: .venv/bin/python profile_throughput.py [--games N] [--searches N]
"""

import argparse
import cProfile
import pstats
import io
import random
import time

from sheepshead import Game, ACTIONS, PARTNER_BY_JD, PARTNER_BY_CALLED_ACE


def pure_game_workload(n_games, seed=0):
    rng = random.Random(seed)
    decisions = 0
    for g in range(n_games):
        mode = PARTNER_BY_CALLED_ACE if g % 2 else PARTNER_BY_JD
        game = Game(partner_selection_mode=mode)
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    a = rng.choice(tuple(valid))
                    player.act(a)
                    decisions += 1
                    valid = player.get_valid_action_ids()
    return decisions


def time_pure_game(n_games):
    t0 = time.perf_counter()
    decisions = pure_game_workload(n_games)
    dt = time.perf_counter() - t0
    print(
        f"\n[A] PURE GAME: {n_games} games, {decisions} decisions in {dt:.3f}s "
        f"=> {dt / n_games * 1000:.3f} ms/game, {dt / decisions * 1e6:.1f} us/decision"
    )
    pr = cProfile.Profile()
    pr.enable()
    pure_game_workload(n_games)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(25)
    print(s.getvalue())


def time_ismcts(n_searches):
    # Imports here so pure-game timing isn't polluted by torch import cost.
    from ppo import PPOAgent
    from ismcts import ISMCTSTeacher, ISMCTSConfig

    agent = PPOAgent(len(ACTIONS), activation="swish")
    agent.load("final_pfsp_swish_ppo.pt", load_optimizers=False)
    cfg = ISMCTSConfig(
        iters={"pick": 48, "partner": 64, "bury": 96, "play": 96},
        det_max_tries=2000,
        ess_floor=4.0,
    )
    teacher = ISMCTSTeacher(agent, cfg)
    rng = random.Random(0)

    # Collect a mix of decision nodes (mostly play, where the deep tree lives).
    nodes = []
    g = 0
    while len(nodes) < n_searches and g < n_searches * 40:
        game = Game(
            partner_selection_mode=(PARTNER_BY_CALLED_ACE if g % 2 else PARTNER_BY_JD)
        )
        agent.reset_recurrent_state()
        fp = []
        observer = random.randint(1, 5)
        captured = False
        while not game.is_done() and not captured:
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    # Capture a mid-game PLAY decision for the observer.
                    from sheepshead import ACTIONS as A

                    is_play = any(A[a - 1].startswith("PLAY ") for a in valid)
                    if (
                        player.position == observer
                        and is_play
                        and game.current_trick >= 1
                        and not game.is_leaster
                    ):
                        nodes.append((game, observer, list(fp)))
                        captured = True
                        break
                    a, _, _ = agent.act(player.get_state_dict(), valid, player.position)
                    name = A[a - 1]
                    if not (name.startswith("BURY ") or name.startswith("UNDER ")):
                        fp.append((player.position, a))
                    player.act(a)
                    valid = player.get_valid_action_ids()
                    if game.was_trick_just_completed:
                        for seat in game.players:
                            agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
                if captured:
                    break
        g += 1

    print(f"\n[B] ISMCTS SEARCH: collected {len(nodes)} play nodes")
    # Warm timing.
    t0 = time.perf_counter()
    for game, observer, fp in nodes:
        teacher.search(game, observer, fp, rng, d_rollout=6)
    dt = time.perf_counter() - t0
    print(
        f"    {len(nodes)} searches in {dt:.3f}s => {dt / len(nodes) * 1000:.1f} ms/search"
    )

    pr = cProfile.Profile()
    pr.enable()
    for game, observer, fp in nodes:
        teacher.search(game, observer, fp, rng, d_rollout=6)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(30)
    print(s.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--searches", type=int, default=12)
    ap.add_argument("--skip-ismcts", action="store_true")
    args = ap.parse_args()

    time_pure_game(args.games)
    if not args.skip_ismcts:
        time_ismcts(args.searches)


if __name__ == "__main__":
    main()
