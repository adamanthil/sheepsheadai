"""One-off Tier-1 validation (NOT committed): batched pool build == sequential.

Confirms _build_worlds_batched reproduces the per-world _build_world reference on
the SAME determinized deals: identical reconstructed world states (hands/history),
and log_w / per-seat memory matching to float tolerance (batched vs batch-1 matmul
differs only at ~1e-5). Also times both to report the speedup.
"""

import random
import time
import copy

import torch

from sheepshead.agent.ppo import load_agent
from sheepshead import Game, ACTIONS, PARTNER_BY_JD, PARTNER_BY_CALLED_ACE
from sheepshead.ismcts import ISMCTSTeacher, ISMCTSConfig

CKPT = "final_pfsp_swish_ppo.pt"


def collect_play_node(agent, game, observer):
    fp = []
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                is_play = any(ACTIONS[a - 1].startswith("PLAY ") for a in valid)
                if (
                    player.position == observer
                    and is_play
                    and game.current_trick >= 1
                    and not game.is_leaster
                ):
                    return list(fp)
                a, _, _ = agent.act(player.get_state_dict(), valid, player.position)
                name = ACTIONS[a - 1]
                if not (name.startswith("BURY ") or name.startswith("UNDER ")):
                    fp.append((player.position, a))
                player.act(a)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for seat in game.players:
                        agent.observe(
                            seat.get_last_trick_state_dict(), player_id=seat.position
                        )
    return None


def sequential_pool(teacher, real_game, deals, fp, observer):
    """Reference: replay each deal separately via _build_world."""
    pool = []
    for deal in deals:
        world, log_w = teacher._build_world(real_game, deal, fp, observer)
        if world is None:
            pool.append(None)
            continue
        mem = {
            pid: t.detach().clone() for pid, t in teacher.agent._player_memories.items()
        }
        pool.append((world, mem, log_w))
    return pool


def main():
    random.seed(0)
    torch.manual_seed(0)
    agent = load_agent(CKPT)
    teacher = ISMCTSTeacher(agent, ISMCTSConfig(det_max_tries=2000))

    n_nodes = 6
    K = 48
    max_state_diff = 0.0
    max_logw_diff = 0.0
    state_mismatch = 0
    nodes = 0
    seq_time = batch_time = 0.0

    g = 0
    while nodes < n_nodes and g < n_nodes * 40:
        mode = PARTNER_BY_CALLED_ACE if g % 2 else PARTNER_BY_JD
        game = Game(partner_selection_mode=mode)
        agent.reset_recurrent_state()
        observer = random.randint(1, 5)
        fp = collect_play_node(agent, game, observer)
        g += 1
        if fp is None:
            continue
        nodes += 1

        rng = random.Random(123)
        deals = [game.sample_determinization(observer, rng) for _ in range(K)]

        # Sequential reference.
        t0 = time.perf_counter()
        seq = sequential_pool(
            teacher,
            copy.deepcopy(game),
            [copy.deepcopy(d) for d in deals],
            fp,
            observer,
        )
        seq_time += time.perf_counter() - t0

        # Batched.
        t0 = time.perf_counter()
        bat = teacher._build_worlds_batched(
            copy.deepcopy(game), [copy.deepcopy(d) for d in deals], fp, observer
        )
        batch_time += time.perf_counter() - t0

        # Both should keep all K worlds (consistent deals).
        seq = [p for p in seq if p is not None]
        assert len(seq) == len(bat) == K, f"pool sizes {len(seq)}/{len(bat)} != {K}"

        for (gw_s, mem_s, lw_s), (gw_b, mem_b, lw_b) in zip(seq, bat):
            # World states must be exactly equal.
            for s in range(1, 6):
                if (
                    sorted(gw_s.players[s - 1].initial_hand)
                    != sorted(gw_b.players[s - 1].initial_hand)
                    or gw_s.players[s - 1].hand != gw_b.players[s - 1].hand
                ):
                    state_mismatch += 1
            if gw_s.history != gw_b.history:
                state_mismatch += 1
            max_logw_diff = max(max_logw_diff, abs(lw_s - lw_b))
            for s in range(1, 6):
                d = (mem_s[s] - mem_b[s]).abs().max().item()
                max_state_diff = max(max_state_diff, d)

    print(f"nodes compared: {nodes}  (K={K} worlds each)")
    print(f"world-state mismatches (hands/history): {state_mismatch}  (must be 0)")
    print(f"max |log_w_seq - log_w_batched|: {max_logw_diff:.2e}")
    print(f"max |memory_seq - memory_batched|: {max_state_diff:.2e}")
    print(
        f"sequential: {seq_time:.3f}s   batched: {batch_time:.3f}s   "
        f"speedup: {seq_time / max(batch_time, 1e-9):.2f}x"
    )
    ok = state_mismatch == 0 and max_logw_diff < 1e-2 and max_state_diff < 1e-2
    print(f"\nBATCHED POOL == SEQUENTIAL: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
