"""Parallel self-play (Lever 1) check — correctness + throughput of the worker pool.

Exercises ``run_pfsp_training``'s parallel game-generation path end-to-end (worker
spawn, versioned weight sync, lazy opponent loading, profile-event capture/replay):

  1. Correctness: run a short window with ``num_workers > 1`` and assert the
     authoritative population's strategic profiles + per-game stats actually updated
     (i.e. the workers' captured events were replayed onto the learner's population —
     not silently dropped).
  2. Throughput: time per-episode at ``num_workers = 1`` vs ``N`` in production ExIt
     search mode; report the speedup.

NOTE: must be run as a script with the ``__main__`` guard below — the spawn start
method re-imports the launching module in every worker, so an unguarded top-level
body would fork-bomb.

Run from the repo root:
  PYTHONPATH=. .venv/bin/python validation/parallel_selfplay_check.py            # correctness (shaped, fast)
  PYTHONPATH=. .venv/bin/python validation/parallel_selfplay_check.py --throughput  # + ExIt timing (slow)
"""

import argparse
import os
import shutil
import tempfile
import time


def _seed_population(popdir, n_per_mode=5):
    from pfsp import PFSPPopulation
    from ppo import PPOAgent
    from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD

    ckpt = os.path.abspath("final_pfsp_swish_ppo.pt")
    pop = PFSPPopulation(
        max_population_jd=75, max_population_called_ace=75, population_dir=popdir
    )
    for mode in (PARTNER_BY_JD, PARTNER_BY_CALLED_ACE):
        for _ in range(n_per_mode):
            agent = PPOAgent(len(ACTIONS), activation="swish")
            agent.load(ckpt, load_optimizers=False)
            pop.add_agent(
                agent=agent,
                partner_mode=mode,
                training_episodes=0,
                parent_id=None,
                activation="swish",
            )
    pop.save_population_state()


def _run(popdir, tmp, *, reward_mode, num_workers, episodes, run_tag):
    from config import PFSPHyperparams, SearchConfig
    from pfsp_runtime import run_pfsp_training

    if reward_mode == "terminal":
        search = SearchConfig(
            head_search_fractions={
                "pick": 1.0,
                "partner": 1.0,
                "bury": 1.0,
                "play": 0.10,
            },
            t_full=1,
            d_short=2,
        )
    else:
        search = None
    hp = PFSPHyperparams(
        reward_mode=reward_mode, search=search, num_workers=num_workers
    )
    big = 10**9
    t0 = time.time()
    run_pfsp_training(
        num_episodes=episodes,
        update_interval=20 if reward_mode == "shaped" else big,
        save_interval=big,
        strategic_eval_interval=big,
        population_add_interval=big,
        cross_eval_interval=big,
        hyperparams=hp,
        run_name=os.path.join(tmp, run_tag).lstrip("/"),
        population_dir=popdir,
    )
    return time.time() - t0


def _profiles_moved(popdir):
    from pfsp import PFSPPopulation

    pop = PFSPPopulation(
        max_population_jd=75, max_population_called_ace=75, population_dir=popdir
    )
    agents = pop.jd_population + pop.called_ace_population
    moved = sum(
        1
        for a in agents
        if abs(a.strategic_profile.trick_win_rate_picker - 0.5) > 1e-9
        or a.strategic_profile.lead_counts_defender > 0
        or a.metadata.games_played > 0
    )
    return moved, len(agents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throughput", action="store_true", help="also run ExIt timing")
    parser.add_argument("--workers", type=int, default=6, help="parallel worker count")
    parser.add_argument("--episodes", type=int, default=40, help="correctness episodes")
    parser.add_argument("--throughput-episodes", type=int, default=24)
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="parcheck_")
    try:
        popdir = os.path.abspath(os.path.join(tmp, "pop"))
        _seed_population(popdir)

        # 1. Correctness: parallel shaped window (fast — no ISMCTS), then verify the
        #    authoritative population was updated via replayed worker events.
        print(f"[correctness] {args.episodes} shaped episodes, {args.workers} workers")
        _run(
            popdir, tmp, reward_mode="shaped",
            num_workers=args.workers, episodes=args.episodes, run_tag="correctness",
        )
        moved, total = _profiles_moved(popdir)
        assert moved == total, f"only {moved}/{total} population agents updated — replay dropped events"
        print(f"[correctness] OK — {moved}/{total} population agents updated\n")

        # 2. Throughput (optional): production ExIt search, 1 vs N workers.
        if args.throughput:
            n = args.throughput_episodes
            print(f"[throughput] {n} ExIt episodes (production search)")
            seq = _run(popdir, tmp, reward_mode="terminal", num_workers=1, episodes=n, run_tag="thru1")
            par = _run(popdir, tmp, reward_mode="terminal", num_workers=args.workers, episodes=n, run_tag="thruN")
            print(f"[throughput] workers=1: {seq / n:.3f}s/ep | "
                  f"workers={args.workers}: {par / n:.3f}s/ep | speedup {seq / par:.2f}x "
                  f"(pool spawn amortizes away on real runs)")
        print("PARALLEL SELF-PLAY CHECK OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
