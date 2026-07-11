#!/usr/bin/env python3
"""Tune deploy-time ISMCTS search for realized play strength under a compute budget.

This is the RIGHT instrument for choosing a *deploy* search config -- distinct from
the single-node audit in ``analysis/counterfactual_trump_leads.py``. The audit asks
"is this one decision EV-wrong?" and answers it with an unbiased per-action Q at a
hand-picked node (so it flattens the prior, ``root_explore_frac=1.0``, and reads
top@Q). Deploy asks a different question: "which config wins the most games per unit
compute, across ALL decisions?" -- and that is a realized-strength question, not a
single-node-Q question. They optimize different things and their optima differ (the
audit wants high exploration at the leak node; deploy wants whatever maximizes
aggregate score under a finite per-move budget, which is usually a LOW frac because
the policy prior is mostly good and the leak nodes are rare).

Method -- paired A/B, search vs the identical raw policy
--------------------------------------------------------
For each deal (fixed ``seed``) and each hero seat, the deal is played twice:

  * **baseline**  -- the hero seat plays the raw policy (greedy argmax), and
  * **treatment** -- the hero seat plays via ISMCTS search at the config under test,

with the *other four seats playing the raw policy in both runs* and the *same deal*.
The paired difference ``Δ = score(search) - score(policy)`` is the value of search at
that config for that (seed, hero) pair. Averaged over many pairs it is the realized
strength gain, with a standard error; common-random-deals pairing cancels most of the
hidden-hand variance. This mirrors the project's established head-to-head metric
(search-augmented vs raw policy), NOT EV-vs-oracle.

Cost is measured alongside strength (mean wall-seconds per searched decision and per
game) so the output is a strength-vs-compute frontier, which is what a budgeted deploy
actually trades against.

Which decisions get searched is set by ``--head-sets`` (a ``;``-separated list of
head groups, each a comma list of ``pick,partner,bury,play``; run as a sweep
dimension so play-only / bid-only / all can be compared in one process, sharing the
raw-policy baselines). NOTE for bidding: a pick/pass (or bury/call) search is a
depth-1 tree whose rollout plays through the rest of bidding (blind, bury, call) and
then ``d_rollout`` observer *play* plies into the tricks before the critic bootstraps
-- so ``--depths`` is the lever that decides how deep into the hand a bid is evaluated.
``--max-depths`` only affects the play head and is inert for bid-only head sets.

Knobs swept (any subset; the cartesian product is run):
  * ``--head-sets`` searched heads     (e.g. ``play;pick,partner,bury;play,pick,partner,bury``)
  * ``--fracs``     root_explore_frac  (root-prior flattening; deploy wants this LOW)
  * ``--iters``     search iterations  (the budget knob; applied to every searched head)
  * ``--depths``    d_rollout          (observer play plies before the critic bootstraps;
                                        for BID heads this is how far into the tricks the
                                        bid is rolled; deep = trust rollout over the
                                        early-game-blind critic, slow; shallow = fast)
  * ``--cpucts``    c_puct             (global PUCT exploration; ~1.25 with the Q-norm)
  * ``--max-depths``play tree depth    (how deep the PUCT tree may grow; 6 = full hand;
                                        play head only)

Selection rule at the searched node is top@Q with a min-visit guard by default (our
audit finding: visit counts are prior-dominated; Q is the right deploy selector),
``--select visits`` switches to the most-visited action.

Usage (from repo root):

    uv run python analysis/tune_deploy_search.py \
        --num-seeds 40 --partner-mode 1 \
        --head-sets 'play;pick,partner,bury;play,pick,partner,bury' \
        --fracs 0.0,0.1,0.25,0.5 --iters 384 --depths 2 \
        --out runs/tune_deploy_search.json

Heavy: each searched decision is ~1-2s, so a game is ~6-12s per searched seat. Start
small and run in the background (PYTHONUNBUFFERED=1, no tight timeout).
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Importing the counterfactual module installs the cached load_agent patch (via the
# scanner it imports) and gives us the public/private + card helpers it already
# defines, so deploy play matches the audit's decision loop exactly.
import sheepshead.analysis.counterfactual_trump_leads as cf  # noqa: E402
import sheepshead.analysis.scan_defender_trump_leads as scan  # noqa: E402
from sheepshead import Game  # noqa: E402

DEFAULT_MODEL = scan.DEFAULT_MODEL


ALL_HEADS = ("pick", "partner", "bury", "play")


def _pick_from_search(
    res: dict, argmax_aid: int, min_visit_frac: float, by: str
) -> int:
    """The deploy action chosen from a search result. ``by='q'`` (default) returns the
    highest-Q action among those with enough visits for Q to be meaningful (guards a
    1-visit fluke); ``by='visits'`` returns the most-visited action."""
    root_n = res.get("root_n", {})
    root_q = res.get("root_q", {})
    valid = res.get("valid", [])
    if not valid:
        return argmax_aid
    if by == "visits":
        return max(valid, key=lambda a: root_n.get(a, 0.0))
    total_n = sum(root_n.values())
    guard = max(1.0, min_visit_frac * total_n)
    eligible = [a for a in valid if root_n.get(a, 0.0) >= guard]
    if not eligible:
        eligible = [a for a in valid if root_n.get(a, 0.0) > 0] or list(valid)
    return max(eligible, key=lambda a: root_q.get(a, float("-inf")))


# ---------------------------------------------------------------------------
# One game playout (baseline = raw policy; treatment = search at the hero seat)
# ---------------------------------------------------------------------------
def play_game(
    agent,
    teacher,
    hero_seat: int,
    seed: int,
    partner_mode: int,
    max_steps: int,
    *,
    search_heads: tuple = ("play",),
    select: str = "q",
    min_visit_frac: float = 0.01,
    d_rollout: Optional[int] = None,
    det_rng: Optional[random.Random] = None,
) -> tuple[float, int, float]:
    """Play one deterministic (greedy) game; the hero seat searches when ``teacher`` is
    not None and the decision's head is in ``search_heads``. Returns
    ``(hero_final_score, n_search_calls, search_seconds)``.

    The decision loop mirrors ``counterfactual_trump_leads._replay_to_node`` so the
    raw-policy branch reproduces the ``/analyze`` deterministic path exactly, and the
    search branch differs only where the hero's chosen action diverges from argmax.
    """
    agent.reset_recurrent_state()
    game = Game(partner_selection_mode=partner_mode, seed=seed)
    forced_public: List[tuple] = []
    n_search = 0
    search_secs = 0.0
    device = cf._device()

    step = 0
    while not game.is_done() and step < max_steps:
        actor = None
        for player in game.players:
            if player.get_valid_action_ids():
                actor = player
                break
        if actor is None:
            break

        pos = actor.position
        state = actor.get_state_dict()
        valid = actor.get_valid_action_ids()

        # Always run the forward: it advances this seat's recurrent memory (exactly as
        # the live game / analyze path does) and gives the greedy argmax fallback.
        memory_in = agent.get_recurrent_memory(pos, device=device)
        encoder_out = agent.encoder.encode_batch(
            [state], memory_in=memory_in.unsqueeze(0), device=device
        )
        agent.set_recurrent_memory(pos, encoder_out["memory_out"][0])
        with torch.no_grad():
            mask = (
                agent.get_action_mask(valid, agent.action_size).unsqueeze(0).to(device)
            )
            hand_ids = torch.as_tensor(
                state["hand_ids"], dtype=torch.long, device=device
            ).view(1, -1)
            action_probs, _ = agent.actor.forward_with_logits(
                encoder_out, mask, hand_ids, agent.encoder.card
            )
        chosen = int(torch.argmax(action_probs, dim=1).item()) + 1

        if (
            teacher is not None
            and pos == hero_seat
            and teacher._infer_head(sorted(valid)) in search_heads
        ):
            # teacher.search snapshots/restores its own copy of the recurrent memory,
            # so the live game's memory (just advanced above) is left intact.
            t0 = time.perf_counter()
            res = teacher.search(
                game,
                pos,
                list(forced_public),
                det_rng or random.Random(seed * 131 + step),
                d_rollout=d_rollout,
                seat_policies=None,  # pure self-play: all seats are this one model
            )
            search_secs += time.perf_counter() - t0
            n_search += 1
            chosen = _pick_from_search(res, chosen, min_visit_frac, select)

        if not cf._is_private_decision(valid):
            forced_public.append((pos, chosen))
        actor.act(chosen)
        if game.was_trick_just_completed:
            for seat in game.players:
                agent.observe(seat.get_last_trick_state_dict(), player_id=seat.position)
        step += 1

    return float(game.players[hero_seat - 1].get_score()), n_search, search_secs


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
@dataclass
class ConfigResult:
    headSet: str
    rootExploreFrac: float
    iters: int
    dRollout: int
    cPuct: float
    maxDepth: int
    select: str
    nPairs: int
    deltaScoreMean: float
    deltaScoreSE: float
    fracImproved: float
    fracChanged: float  # fraction of pairs where search changed the outcome at all
    baselineScoreMean: float
    searchScoreMean: float
    secPerDecision: float
    secPerGame: float
    decisionsPerGame: float


def _se(vals: np.ndarray) -> float:
    return float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0


def run_config(
    agent,
    teacher,
    *,
    head_set: tuple,
    frac: float,
    iters: int,
    d_rollout: int,
    c_puct: float,
    max_depth: int,
    seeds: List[int],
    hero_seats: List[int],
    partner_mode: int,
    max_steps: int,
    select: str,
    min_visit_frac: float,
    baseline_cache: dict,
) -> ConfigResult:
    """Run the paired A/B tournament for one search config; return aggregated stats.

    ``baseline_cache`` maps ``(seed, hero_seat) -> baseline_score`` and is shared across
    configs (the raw-policy baseline is config-independent, so it is computed once)."""
    cfg = teacher.config
    cfg.root_explore_frac = frac
    cfg.iters = {k: iters for k in cfg.iters}
    cfg.c_puct = c_puct
    cfg.max_depth = {**cfg.max_depth, "play": max_depth}

    deltas: List[float] = []
    base_scores: List[float] = []
    srch_scores: List[float] = []
    changed = 0
    tot_search = 0
    tot_secs = 0.0
    n_games = 0

    for seed in seeds:
        for hero in hero_seats:
            key = (seed, hero, partner_mode)
            if key not in baseline_cache:
                base, _, _ = play_game(agent, None, hero, seed, partner_mode, max_steps)
                baseline_cache[key] = base
            base = baseline_cache[key]

            srch, n_s, secs = play_game(
                agent,
                teacher,
                hero,
                seed,
                partner_mode,
                max_steps,
                search_heads=head_set,
                select=select,
                min_visit_frac=min_visit_frac,
                d_rollout=d_rollout,
                det_rng=random.Random(seed * 977 + hero),
            )
            deltas.append(srch - base)
            base_scores.append(base)
            srch_scores.append(srch)
            if srch != base:
                changed += 1
            tot_search += n_s
            tot_secs += secs
            n_games += 1

    d = np.array(deltas, dtype=float)
    return ConfigResult(
        headSet="+".join(head_set),
        rootExploreFrac=frac,
        iters=iters,
        dRollout=d_rollout,
        cPuct=c_puct,
        maxDepth=max_depth,
        select=select,
        nPairs=len(d),
        deltaScoreMean=float(d.mean()) if len(d) else 0.0,
        deltaScoreSE=_se(d),
        fracImproved=float(np.mean(d > 0)) if len(d) else 0.0,
        fracChanged=changed / n_games if n_games else 0.0,
        baselineScoreMean=float(np.mean(base_scores)) if base_scores else 0.0,
        searchScoreMean=float(np.mean(srch_scores)) if srch_scores else 0.0,
        secPerDecision=tot_secs / tot_search if tot_search else 0.0,
        secPerGame=tot_secs / n_games if n_games else 0.0,
        decisionsPerGame=tot_search / n_games if n_games else 0.0,
    )


def _fmt_row(r: ConfigResult) -> str:
    sig = r.deltaScoreMean / r.deltaScoreSE if r.deltaScoreSE > 0 else 0.0
    return (
        f"  [{r.headSet:<22s}] f={r.rootExploreFrac:<4g} it={r.iters:<5d} dR={r.dRollout:<2d} "
        f"c={r.cPuct:<4g} md={r.maxDepth:<2d} {r.select:>6s} | "
        f"Δscore {r.deltaScoreMean:+6.3f} ± {r.deltaScoreSE:.3f} ({sig:+.1f}σ)  "
        f"win {r.fracImproved * 100:4.0f}%  chg {r.fracChanged * 100:4.0f}%  | "
        f"{r.secPerDecision:5.2f}s/dec  {r.secPerGame:5.1f}s/game"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-seeds", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--partner-mode", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--hero-seats",
        default="1,2,3,4,5",
        help="Comma-separated seats the search controls (paired vs raw policy).",
    )
    parser.add_argument(
        "--head-sets",
        default="play",
        help="';'-separated head groups to compare, each a comma list of "
        "pick,partner,bury,play. E.g. 'play;pick,partner,bury;play,pick,partner,bury'.",
    )
    parser.add_argument("--fracs", default="0.0,0.1,0.25,0.5")
    parser.add_argument("--iters", default="384")
    parser.add_argument("--depths", default="2", help="d_rollout values to sweep")
    parser.add_argument("--cpucts", default="1.25")
    parser.add_argument("--max-depths", default="6", help="play tree max_depth values")
    parser.add_argument("--select", choices=["q", "visits"], default="q")
    parser.add_argument("--min-visit-frac", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    agent = scan._cached_load_agent(args.model)

    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

    cfg = ISMCTSConfig()
    cfg.batch_size = args.batch_size
    teacher = ISMCTSTeacher(agent, cfg)

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    hero_seats = [int(s) for s in args.hero_seats.split(",")]
    head_sets = []
    for grp in args.head_sets.split(";"):
        heads = tuple(h.strip() for h in grp.split(",") if h.strip())
        bad = [h for h in heads if h not in ALL_HEADS]
        if bad:
            parser.error(f"unknown head(s) {bad}; choose from {ALL_HEADS}")
        head_sets.append(heads)
    fracs = [float(x) for x in args.fracs.split(",")]
    iters_list = [int(x) for x in args.iters.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    cpucts = [float(x) for x in args.cpucts.split(",")]
    max_depths = [int(x) for x in args.max_depths.split(",")]

    grid = list(
        itertools.product(head_sets, fracs, iters_list, depths, cpucts, max_depths)
    )
    print(
        f"Deploy-search tuning: {len(seeds)} seeds x {len(hero_seats)} hero seats "
        f"= {len(seeds) * len(hero_seats)} paired games per config; {len(grid)} configs.\n"
        f"Paired Δ = score(search) - score(raw policy), same deal/opponents. "
        f"Selector: top@{args.select}.\n"
    )

    baseline_cache: dict = {}
    results: List[ConfigResult] = []
    for head_set, frac, iters, d_rollout, c_puct, max_depth in grid:
        t0 = time.perf_counter()
        r = run_config(
            agent,
            teacher,
            head_set=head_set,
            frac=frac,
            iters=iters,
            d_rollout=d_rollout,
            c_puct=c_puct,
            max_depth=max_depth,
            seeds=seeds,
            hero_seats=hero_seats,
            partner_mode=args.partner_mode,
            max_steps=args.max_steps,
            select=args.select,
            min_visit_frac=args.min_visit_frac,
            baseline_cache=baseline_cache,
        )
        results.append(r)
        print(_fmt_row(r) + f"   [{time.perf_counter() - t0:.0f}s]")

    # Frontier: best Δscore, and best Δscore-per-second.
    print("\n--- ranked by Δscore (realized strength) ---")
    for r in sorted(results, key=lambda r: -r.deltaScoreMean):
        print(_fmt_row(r))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "seeds": [seeds[0], seeds[-1]],
                    "heroSeats": hero_seats,
                    "partnerMode": args.partner_mode,
                    "select": args.select,
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
        )
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
