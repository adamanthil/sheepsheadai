"""One-off P4 driver-wiring check (NOT committed).

Confirms play_population_game dispatches bidding-head ISMCTS search end-to-end
(not just the teacher in isolation): with SearchConfig.head_search_fractions = 1.0 on the bidding
heads, search targets must actually land on PICK/PARTNER/BURY transitions, the
update() must fire with distillation, and nothing may KeyError.
"""

import random
import time

from ppo import PPOAgent
from sheepshead import ACTIONS, PARTNER_BY_JD, PARTNER_BY_CALLED_ACE
from pfsp import PopulationAgent, AgentMetadata
from ismcts import ISMCTSTeacher, ISMCTSConfig
from pfsp_runtime import play_population_game, _search_head
from config import SearchConfig

CKPT = "final_pfsp_swish_ppo.pt"


def make_agent():
    a = PPOAgent(len(ACTIONS), activation="swish")
    a.load(CKPT, load_optimizers=False)
    return a


def make_opp(i, mode):
    meta = AgentMetadata(
        agent_id=f"drv_opp_{i}", creation_time=time.time(), parent_id=None,
        training_episodes=0, partner_mode=mode, activation="swish",
    )
    return PopulationAgent(make_agent(), meta)


def main():
    random.seed(1)
    agent = make_agent()
    cfg = ISMCTSConfig(
        iters={"pick": 8, "partner": 8, "bury": 8, "play": 8},
        det_max_tries=300, ess_floor=1.0,
    )
    teacher = ISMCTSTeacher(agent, cfg)
    determinization_rng = random.Random(7)
    # All bidding heads at 1.0 (the new default); modest play fraction.
    search_config = SearchConfig(
        head_search_fractions={"pick": 1.0, "partner": 1.0, "bury": 1.0, "play": 0.3}
    )

    by_head = {"pick": [0, 0], "partner": [0, 0], "bury": [0, 0], "play": [0, 0]}  # [searched, total]
    n_games = 16
    for g in range(n_games):
        mode = PARTNER_BY_CALLED_ACE if g % 2 else PARTNER_BY_JD
        opps = [make_opp(i, mode) for i in range(4)]
        game, events, final_scores, _, _ = play_population_game(
            training_agent=agent, opponents=opps, partner_mode=mode,
            training_agent_position=random.randint(1, 5),
            reward_mode="terminal", teacher=teacher,
            determinization_rng=determinization_rng,
            search_config=search_config,
        )
        for ev in events:
            if ev["kind"] != "action":
                continue
            head = _search_head(ev["valid_actions"])
            by_head[head][1] += 1
            if ev.get("has_search_target"):
                by_head[head][0] += 1
        agent.store_episode_events(events)

    print("head    searched/total")
    for h in ("pick", "partner", "bury", "play"):
        s, t = by_head[h]
        print(f"  {h:8s} {s}/{t}")

    stats = agent.update(epochs=2, batch_size=16)
    d = stats.get("distill", {})
    print("\nupdate: num_transitions=", stats.get("num_transitions"),
          " distill_loss=", round(d.get("loss", 0.0), 4),
          " pg_masked_fraction=", round(d.get("pg_masked_fraction", 0.0), 4))

    # Bidding heads must have been searched (the whole point of P4).
    ok = (by_head["pick"][0] > 0 and by_head["partner"][0] > 0 and by_head["bury"][0] > 0
          and d.get("pg_masked_fraction", 0.0) > 0.0)
    print("\nDRIVER BIDDING-SEARCH", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
