"""Shared fixtures for the search-corpus and policy-iteration suites: a
seeded recall agent, a deterministic committee stand-in for the ISMCTS
teacher, and the worker-state install the corpus generator expects."""

from types import SimpleNamespace

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE
from sheepshead.agent.ppo import PPOAgent
from sheepshead.tests.ppo_test_helpers import seed_all
from sheepshead.training.distill_corpus import play_corpus_game, set_worker_state

ARCH = "perceiver-recall"


class ScriptedCommittee:
    """ISMCTSTeacher stand-in: deterministic committee replicates with a
    LARGE Q spread (always material under the real shrinkage) favoring the
    lowest-id valid action. Carries the engine config the target builder
    reads readout constants from."""

    def __init__(self, replicates=3):
        self.replicates = replicates
        self.config = SimpleNamespace(gumbel_c_visit=50.0, gumbel_c_scale=0.1)

    def search_committee(self, game, observer, forced_public, rngs, d_rollout=None):
        player = game.players[observer - 1]
        valid = sorted(player.get_valid_action_ids())
        out = []
        for rep in range(self.replicates):
            q = {a: 1.0 - 0.5 * i + 1e-4 * rep for i, a in enumerate(valid)}
            out.append(
                {
                    "ok": True,
                    "root_q": q,
                    "root_n": {a: 256.0 for a in valid},
                    "root_prior": {a: 1.0 / len(valid) for a in valid},
                }
            )
        return out


def worker_state(agent, **overrides):
    args = {
        "seed": 7,
        "collect_oracle": overrides.pop("collect_oracle", False),
        "iters": 8,
        "replicates": 3,
        "d_rollout": 1,
        "shrink_nu": 4.0,
        "shrink_s2_global": 6.95e-4,
        "p_base": overrides.pop("p_base", 1.0),
        "boost_lead": 1.0,
        "boost_cs": 1.0,
        "p_min": overrides.pop("p_min", 1.0),
        "p_max": overrides.pop("p_max", 1.0),
    }
    args.update(overrides)
    set_worker_state(agent, ScriptedCommittee(), args)
    return args


def fresh_agent():
    seed_all(0)
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    agent.stash_action_probs = True
    return agent


def generate_game(agent, game_idx=0, **overrides):
    worker_state(agent, **overrides)
    return play_corpus_game((game_idx, PARTNER_BY_CALLED_ACE))
