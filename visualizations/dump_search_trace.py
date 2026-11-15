"""Trace one real ISMCTS committee search and dump it to JSON for the
search walkthrough figure in the blog post.

Plays a called-ace hand with the league-final agent (the checkpoint the
first policy iteration searches with) up to the first-trick lead of a
defender who holds the called suit, then runs the corpus recipe on that
decision: three independent searches of 1,024 simulations, d_rollout 1,
oracle-critic leaves, undiscounted backups. Every simulation of the first
search is recorded.

    uv run python visualizations/dump_search_trace.py
    uv run python visualizations/dump_search_trace.py --scan 60   # list seeds

Output (search_trace.json), cards as strings like "8S":

  seat, picker, called   the decision: who is on lead, who picked, the call
  cards, prior           the legal cards and the network policy over them
  iters, pool, distinct  simulations per search, deals sampled, deals used
  p, k, v                per simulation of search 1: the observer's own cards
                         chosen in the tree (indices into cards), "e" if the
                         hand ran to the end or "c" for a critic estimate,
                         and the backed-up value x 10000
  detail                 a few simulations in full: the deal, the tricks
                         (seat-indexed), the outcome
  samples                a few sampled deals with how often each was drawn
  q, n                   root value and visit count per card, per search

Values are in the critic's units: game score / 12 from the observer's side.
"""

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
import torch

from sheepshead.agent.observation import observation_for
from sheepshead.agent.ppo import load_agent
from sheepshead.game import ACTIONS, PARTNER_BY_CALLED_ACE, Game
from sheepshead.search.ismcts import ISMCTSConfig, ISMCTSTeacher, is_private_action
from sheepshead.training.distill_corpus import lead_features

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = (
    ROOT
    / "runs"
    / "202609_recall_rc"
    / "league"
    / "checkpoints"
    / "checkpoint_8000000.pt"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "search_trace.json"
# A hand where the league-final policy most often leads trump, splitting the
# rest between two called-suit cards, and all three searches prefer the
# called suit. Found by scanning seeds 0-5999 for leads whose top-prior card
# is not in the called suit.
DEFAULT_SEED = 3152
ITERS = 1024  # the t0-lead budget of the corpus recipe
D_ROLLOUT = 1
REPLICATES = 3
# Simulations (0-based) exported in full, besides the first to reach the end
# of the hand. Simulation 0 expands the root and backs nothing up.
DETAIL = [1, 2, 3, 4, 5, 6, 7, 255, 640, 1023]
SAMPLE_WORLDS = 7


def card(action_id: int) -> str:
    return ACTIONS[action_id - 1][5:]


def play_to_node(agent, seed):
    """Play `seed` until the first trick's lead. Returns (game, player,
    forced_public) when the leader is a defender holding the called suit
    and an alternative, else None."""
    torch.manual_seed(seed)
    game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=seed)
    agent.reset_recurrent_state()
    forced = []
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                state = observation_for(player, agent)
                action, _, _ = agent.act(state, valid, player.position)
                if game.is_leaster:
                    return None
                if ACTIONS[action - 1].startswith("PLAY "):
                    if game.alone_called:
                        return None
                    _, eligible = lead_features(game, player)
                    return (game, player, forced) if eligible else None
                if not is_private_action(action):
                    forced.append((player.position, action))
                player.act(action)
                valid = player.get_valid_action_ids()
    return None


class TracingTeacher(ISMCTSTeacher):
    """Records the belief pool and every finished simulation of a search."""

    def _search_inner(self, real_game, observer, forced_public):
        self.traced_sims = []
        self.traced_pool = []
        self._pool_index = {}
        self._leaf_kind = None
        return super()._search_inner(real_game, observer, forced_public)

    def _build_pool(self, real_game, observer, forced_public, n_worlds):
        pool = super()._build_pool(real_game, observer, forced_public, n_worlds)
        self.traced_pool = pool
        return pool

    def _spawn_sims(self, chunk, pool, root):
        sims = super()._spawn_sims(chunk, pool, root)
        for sim, pool_idx in zip(sims, chunk):
            self._pool_index[id(sim)] = pool_idx
        return sims

    def _finish_terminal(self, sim, observer):
        self._leaf_kind = "end"
        super()._finish_terminal(sim, observer)

    def _finish_value(self, sim, value):
        kind, self._leaf_kind = self._leaf_kind or "critic", None
        world = sim.world
        self.traced_sims.append(
            {
                "world": self._pool_index[id(sim)],
                "path": [card(action_id) for _, action_id in sim.path],
                "kind": kind,
                "v": float(value),
                "history": [list(trick) for trick in world.history],
                "leaders": list(world.leaders),
                "winners": list(world.trick_winners),
                "points": list(world.trick_points),
            }
        )
        super()._finish_value(sim, value)


def export_world(pool, used, index):
    game = pool[index][0]
    partner = next((p.position for p in game.players if game.called_card in p.hand), 0)
    return {
        "h": [list(p.hand) for p in game.players],
        "bury": list(game.bury),
        "partner": partner,
        "used": used.get(index, 0),
    }


def export_detail(sim, index, pool, used):
    tricks = []
    for t, row in enumerate(sim["history"]):
        if not any(row):
            break
        complete = all(row)
        tricks.append(
            {
                "c": row,
                "lead": sim["leaders"][t],
                "win": sim["winners"][t] if complete else 0,
                "pts": sim["points"][t] if complete else None,
            }
        )
    return {
        "i": index,
        "world": export_world(pool, used, sim["world"]),
        "path": sim["path"],
        "kind": sim["kind"],
        "v": round(sim["v"], 4),
        "tricks": tricks,
    }


def trace(agent, seed):
    node = play_to_node(agent, seed)
    if node is None:
        raise SystemExit(
            f"Seed {seed} has no called-suit defender lead on the first trick; "
            "pick another --seed or list candidates with --scan."
        )
    game, player, forced = node
    teacher = TracingTeacher(
        agent,
        ISMCTSConfig(iters={h: ITERS for h in ("pick", "partner", "bury", "play")}),
    )
    base = random.Random(20260903 ^ seed)
    results = []
    sims: list = []
    pool: list = []
    for replicate in range(REPLICATES):
        rng = random.Random(base.getrandbits(64))
        result = teacher.search(
            game, player.position, list(forced), rng, d_rollout=D_ROLLOUT
        )
        results.append(result)
        if replicate == 0:
            sims, pool = teacher.traced_sims, teacher.traced_pool
    valid = results[0]["valid"]
    cards = [card(a) for a in valid]
    index = {c: i for i, c in enumerate(cards)}
    used = collections.Counter(s["world"] for s in sims)
    first_end = next(i for i, s in enumerate(sims) if s["kind"] == "end")
    detail = sorted({*DETAIL, first_end})
    # Sample deals for the figure: the one the search drew most, then deals
    # it never drew alternating with deals it did, in the order dealt.
    top = used.most_common(1)[0][0]
    drawn = [i for i in range(len(pool)) if used.get(i) and i != top]
    never = [i for i in range(len(pool)) if not used.get(i)]
    samples = [top]
    for pair in zip(never, drawn):
        samples.extend(pair)
    samples = samples[:SAMPLE_WORLDS]
    return {
        "seed": seed,
        "seat": player.position,
        "picker": game.picker,
        "called": game.called_card,
        "cards": cards,
        "prior": [
            round(float(np.mean([r["root_prior"][a] for r in results])), 4)
            for a in valid
        ],
        "iters": len(sims),
        "pool": len(pool),
        "distinct": len(used),
        "ess": round(results[0]["ess"]),
        "p": ["".join(str(index[c]) for c in s["path"]) for s in sims],
        "k": "".join("e" if s["kind"] == "end" else "c" for s in sims),
        "v": [round(s["v"] * 10000) for s in sims],
        "detail": [export_detail(sims[i], i, pool, used) for i in detail],
        "samples": [export_world(pool, used, i) for i in samples],
        "q": [[round(r["root_q"][a], 4) for a in valid] for r in results],
        "n": [[int(r["root_n"][a]) for a in valid] for r in results],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--scan",
        type=int,
        metavar="N",
        help="list the seeds below N that reach a qualifying decision, and exit",
    )
    args = ap.parse_args()

    agent = load_agent(str(args.checkpoint))
    agent.gamma = 1.0  # terminal-reward regime: undiscounted backups

    if args.scan:
        for seed in range(args.scan):
            node = play_to_node(agent, seed)
            if node is not None:
                game, player, _ = node
                print(seed, f"picker={game.picker}", game.called_card, player.hand)
        return

    data = trace(agent, args.seed)
    args.out.write_text(json.dumps(data, separators=(",", ":")))
    for c, q, n in zip(data["cards"], zip(*data["q"]), zip(*data["n"])):
        print(f"{c:>4}  q={[f'{x:+.3f}' for x in q]}  n={list(n)}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
