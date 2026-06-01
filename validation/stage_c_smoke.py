"""One-off Stage C smoke test (NOT committed).

Exercises the full Stage C path end-to-end on CPU with a tiny search budget:
  play_population_game (with ISMCTS teacher producing pi' targets, terminal-only
  reward) -> store_episode_events (search_target/has_search_target) -> update()
  (forward-KL distillation + PG-mask). Confirms it runs and the distill metrics
  are populated.
"""

import random
import time

from ppo import PPOAgent
from sheepshead import ACTIONS, PARTNER_BY_JD
from pfsp import PopulationAgent, AgentMetadata
from ismcts import ISMCTSTeacher, ISMCTSConfig
from pfsp_runtime import play_population_game
from config import SearchConfig

CKPT = "final_pfsp_swish_ppo.pt"


def make_agent():
    a = PPOAgent(len(ACTIONS), activation="swish")
    a.load(CKPT, load_optimizers=False)
    return a


def make_opponent(i):
    agent = make_agent()
    meta = AgentMetadata(
        agent_id=f"smoke_opp_{i}",
        creation_time=time.time(),
        parent_id=None,
        training_episodes=0,
        partner_mode=PARTNER_BY_JD,
        activation="swish",
    )
    return PopulationAgent(agent, meta)


def main():
    random.seed(0)
    training_agent = make_agent()
    opponents = [make_opponent(i) for i in range(4)]

    # Tiny search budget so the smoke is fast but still fires on every head.
    cfg = ISMCTSConfig(
        iters={"pick": 6, "partner": 6, "bury": 6, "play": 6},
        det_max_tries=200,
        ess_floor=1.0,  # low so targets actually pass and exercise distillation
    )
    teacher = ISMCTSTeacher(training_agent, cfg)
    determinization_rng = random.Random(123)
    # Play-head search only for this smoke. High play fraction so it reliably
    # exercises distillation + PG-mask.
    search_config = SearchConfig(
        head_search_fractions={"pick": 0.0, "partner": 0.0, "bury": 0.0, "play": 0.8}
    )

    n_games = 8
    searched = 0
    actions = 0
    for g in range(n_games):
        game, events, final_scores, _, _ = play_population_game(
            training_agent=training_agent,
            opponents=opponents,
            partner_mode=PARTNER_BY_JD,
            training_agent_position=random.randint(1, 5),
            reward_mode="terminal",
            teacher=teacher,
            determinization_rng=determinization_rng,
            search_config=search_config,
        )
        for ev in events:
            if ev["kind"] == "action":
                actions += 1
                if ev.get("has_search_target"):
                    searched += 1
        training_agent.store_episode_events(events)
        print(
            f"  game {g + 1}/{n_games}: {len(events)} events, leaster={game.is_leaster}"
        )

    print(f"\naction transitions={actions}, with search target={searched}")
    print("Running update() ...")
    stats = training_agent.update(epochs=2, batch_size=8)
    print("\n--- update stats ---")
    print("num_transitions:", stats.get("num_transitions"))
    print("approx_kl:", round(stats.get("approx_kl", 0.0), 5))
    print("distill:", {k: round(v, 5) for k, v in stats.get("distill", {}).items()})
    print("value loss:", round(stats["critic_losses"]["value"], 5))
    assert searched > 0, (
        "no search targets were produced — distillation never exercised"
    )
    assert stats.get("distill", {}).get("pg_masked_fraction", 0.0) > 0.0, (
        "PG-mask fraction is zero — mask never applied"
    )
    print("\nSMOKE PASS")


if __name__ == "__main__":
    main()
