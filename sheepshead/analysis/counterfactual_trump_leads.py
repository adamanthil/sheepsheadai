#!/usr/bin/env python3
"""Counterfactual analysis of defender trump-leads on the first two tricks.

For each spot where a DEFENDER (not picker / partner / secret-partner, non-leaster)
*leads* on trick 0 or 1 with both a trump and a fail option available, we ask:
is leading trump better than leading the best fail? We answer it three ways, from
weakest to strongest, and report all three side by side:

1. **Single deterministic rollout** (point estimate, for continuity with the
   ``/analyze`` view). Replay the exact deterministic game to the decision node,
   then continue greedy self-play after forcing the best trump vs the best fail
   lead. One continuation each -- fast, reproducible, but high variance.

2. **Paired Monte-Carlo over the TRUE deal** (the statistical workhorse, ported
   from ``validation/counterfactual_trump_leads.py``). From the SAME snapshot, run
   R *stochastic* rollouts per branch (all five seats sampled from the policy). The
   deal is fixed; the only difference between branches is the first card. Report
   defender card points, the leader's RL game score, and defender win rate, as
   Δ = trump - fail with a standard error across cases.

2b. **Paired Monte-Carlo over the BELIEF pool**. Identical to (2), but each rollout
   is played from a determinized world sampled from the ISMCTS belief pool
   (``_build_pool`` / ``_pool_probs``) instead of the true deal. This is the middle
   rung of a 3-rung ladder: (2) true-deal-MC, (2b) belief-MC, (3) ISMCTS. The
   (2)->(2b) gap isolates hindsight (the true deal vs the agent's posterior); the
   (2b)->(3) gap isolates search-continuation optimism (the search plays a better
   rest-of-hand than the raw policy).

3. **ISMCTS search @ N iterations** (belief-averaged, search continuation). Run the
   SO-ISMCTS teacher at the node and report whether its most-visited action is the
   trump lead, a fail, or another trump.

Two groups are aggregated separately:
  * **TRUMP-PREF** -- the policy's argmax lead is a trump (the behavior under
    scrutiny). Δ>0 means the trump lead genuinely wins more.
  * **FAIL-PREF** -- control / falsification: argmax is a fail. We expect Δ<=0
    here; if the method instead shows trump winning, the measurement is suspect.

Fidelity
--------
* The deterministic replay + continuation mirrors ``server.services.analyze.
  simulate_game`` (encode -> argmax -> act with per-seat ``observe`` propagation),
  so the deterministic trump branch of a TRUMP-PREF case reproduces ``/analyze``
  for that seed.
* Both MC branches start from one snapshot (game + per-seat recurrent memory taken
  after the node's forward), exactly like the original validation script.
* All five seats are the single model under analysis (self-play), so the ISMCTS
  search uses ``seat_policies=None``.

Usage (from repo root):

    uv run python analysis/counterfactual_trump_leads.py \
        --num-seeds 3200 --partner-mode 1 --rollouts 50 --iters 384 \
        --out runs/counterfactual_trump_leads.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Importing the scanner installs the cached load_agent patch on the analyze
# service and gives us its case-detection helpers + simulate_game.
import sheepshead.analysis.scan_defender_trump_leads as scan  # noqa: E402
from server.api.schemas import AnalyzeSimulateRequest  # noqa: E402
from sheepshead import (  # noqa: E402
    ACTION_IDS,
    ACTION_LOOKUP,
    ACTIONS,
    FAIL,
    Game,
    TRUMP,
)

TRUMP_SET = set(TRUMP)
FAIL_SET = set(FAIL)

DEFAULT_MODEL = scan.DEFAULT_MODEL

# Defenders win the hand with 60+ card points (picker needs 61 to win).
DEFENDER_WIN_POINTS = 60


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_private_decision(valid_actions) -> bool:
    """True when the decision is a private bury/under (excluded from the public
    record fed to the ISMCTS teacher's forced replay) -- mirrors pfsp_runtime."""
    return any(
        ACTIONS[a - 1].startswith("BURY ") or ACTIONS[a - 1].startswith("UNDER ")
        for a in valid_actions
    )


def _card_of(action_id: int) -> Optional[str]:
    name = ACTION_LOOKUP.get(action_id, "")
    return name[5:] if name.startswith("PLAY ") else None


def _snapshot_memory(agent) -> dict:
    return {pid: t.detach().clone() for pid, t in agent._player_memories.items()}


def _restore_memory(agent, snap: dict) -> None:
    agent._player_memories = {pid: t.detach().clone() for pid, t in snap.items()}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class NodeInfo:
    seat: int
    trickIndex: int
    argmaxCard: str
    argmaxLogit: float
    bestTrumpCard: str
    bestTrumpLogit: float
    bestFailCard: str
    bestFailLogit: float
    trumpLeadOptions: List[str]
    failLeadOptions: List[str]
    handTrumpCount: int
    handFailCount: int
    hand: List[str]


@dataclass
class DetBranch:
    """One deterministic (greedy) continuation after a forced lead."""

    defenderPoints: int
    pickerPoints: int
    leaderScore: int
    win: int


@dataclass
class McBranch:
    """R stochastic continuations after a forced lead."""

    R: int
    defenderPointsMean: float
    leaderScoreMean: float
    winRate: float


@dataclass
class BeliefMcBranch:
    """R policy rollouts over the ISMCTS belief pool (determinized worlds sampled
    ~ exp(log_w)), after a forced lead. Same belief as the search; differs from it
    only in that the continuation is the raw policy, not tree search."""

    R: int
    poolSize: int
    ess: float
    defenderPointsMean: float
    leaderScoreMean: float
    winRate: float


@dataclass
class SearchOutcome:
    iters: int
    rootExploreFrac: float
    rolloutDepth: int
    ess: float
    ok: bool
    nIter: int
    # PRIMARY verdict: best action by Q among adequately-visited actions. With a
    # flattened root prior (frac~1) this is the value recommendation; visit counts
    # alone are prior-dominated and unreliable (see the explore-sweep diagnostic).
    topQActionId: int
    topQAction: str
    topQValue: float
    topQIsArgmax: bool
    topQIsTrump: bool
    topQIsFail: bool
    # Informational: best action by raw visit count.
    topActionId: int
    topAction: str
    topIsArgmax: bool
    topIsTrump: bool
    topIsFail: bool
    trumpVisits: float
    trumpQ: float
    failVisits: float
    failQ: float
    bestTrumpCard: str
    bestFailCard: str
    ranking: List[Dict]


@dataclass
class CaseResult:
    seed: int
    partnerMode: int
    stepIndex: int
    trickIndex: int
    seat: int
    seatName: str
    pickerSeat: int
    group: str  # "trump" (TRUMP-PREF) | "fail" (FAIL-PREF control)
    hand: List[str]
    node: NodeInfo
    # Single deterministic rollout (Delta = trump - fail).
    detTrump: DetBranch
    detFail: DetBranch
    detDeltaPoints: int
    detDeltaScore: int
    # Paired Monte-Carlo over the TRUE deal (Delta = trump - fail).
    mcTrump: McBranch
    mcFail: McBranch
    mcDeltaPoints: float
    mcDeltaScore: float
    mcDeltaWin: float
    # Paired Monte-Carlo over the BELIEF pool (determinized worlds), if available.
    beliefMcTrump: Optional[BeliefMcBranch] = None
    beliefMcFail: Optional[BeliefMcBranch] = None
    beliefMcDeltaPoints: Optional[float] = None
    beliefMcDeltaScore: Optional[float] = None
    beliefMcDeltaWin: Optional[float] = None
    search: Optional[SearchOutcome] = None


# ---------------------------------------------------------------------------
# Rollout primitives
# ---------------------------------------------------------------------------
def _play_out(agent, game, device, deterministic: bool) -> None:
    """Continue a positioned game to terminal, with per-seat observation
    propagation on trick completion (matches simulate_game / play.py)."""
    while not game.is_done():
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

        if deterministic:
            memory_in = agent.get_recurrent_memory(pos, device=device)
            encoder_out = agent.encoder.encode_batch(
                [state], memory_in=memory_in.unsqueeze(0), device=device
            )
            agent.set_recurrent_memory(pos, encoder_out["memory_out"][0])
            with torch.no_grad():
                mask = (
                    agent.get_action_mask(valid, agent.action_size)
                    .unsqueeze(0)
                    .to(device)
                )
                hand_ids = torch.as_tensor(
                    state["hand_ids"], dtype=torch.long, device=device
                ).view(1, -1)
                probs, _ = agent.actor.forward_with_logits(
                    encoder_out, mask, hand_ids, agent.encoder.card
                )
            action_id = int(torch.argmax(probs, dim=1).item()) + 1
        else:
            action_id, _, _ = agent.act(state, valid, pos, deterministic=False)

        actor.act(action_id)
        if game.was_trick_just_completed:
            for seat in game.players:
                agent.observe(seat.get_last_trick_state_dict(), player_id=seat.position)


def _branch_metrics(game, seat: int) -> DetBranch:
    dp = int(game.get_final_defender_points())
    return DetBranch(
        defenderPoints=dp,
        pickerPoints=int(game.get_final_picker_points()),
        leaderScore=int(game.players[seat - 1].get_score()),
        win=1 if dp >= DEFENDER_WIN_POINTS else 0,
    )


def _force_and_play(
    agent, node_game, node_mem, seat: int, card: str, device, deterministic: bool
) -> DetBranch:
    """Deepcopy the snapshot, restore memory, force ``seat`` to lead ``card``,
    then roll to terminal. Both branches start from the identical snapshot."""
    g = copy.deepcopy(node_game)
    _restore_memory(agent, node_mem)
    g.players[seat - 1].act(
        ACTION_IDS[f"PLAY {card}"]
    )  # a lead never completes a trick
    _play_out(agent, g, device, deterministic)
    return _branch_metrics(g, seat)


def _mc_branch(
    agent, node_game, node_mem, seat: int, card: str, R: int, device
) -> McBranch:
    dps, scores, wins = [], [], []
    for _ in range(R):
        m = _force_and_play(
            agent, node_game, node_mem, seat, card, device, deterministic=False
        )
        dps.append(m.defenderPoints)
        scores.append(m.leaderScore)
        wins.append(m.win)
    return McBranch(
        R=R,
        defenderPointsMean=float(np.mean(dps)),
        leaderScoreMean=float(np.mean(scores)),
        winRate=float(np.mean(wins)),
    )


def _replay_to_node(
    agent,
    seed: int,
    partner_mode: int,
    target_step: int,
    max_steps: int,
    device,
    *,
    teacher=None,
    det_rng: Optional[random.Random] = None,
    iters: int = 384,
    rollout_depth: Optional[int] = None,
    min_visit_frac: float = 0.01,
):
    """Deterministically replay the game to ``target_step`` (the defender lead),
    returning ``(node_game_copy, node_mem_snapshot, NodeInfo, SearchOutcome|None,
    forced_public_at_node)``.

    Mirrors simulate_game's decision loop so the snapshot is the exact ``/analyze``
    state at that node. Returns ``(None, ...)`` if the node isn't reached.
    """
    agent.reset_recurrent_state()
    game = Game(partner_selection_mode=partner_mode, seed=seed)
    forced_public: List[tuple] = []

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
        valid_sorted = sorted(valid)

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
            action_probs, logits = agent.actor.forward_with_logits(
                encoder_out, mask, hand_ids, agent.encoder.card
            )
        logits_np = logits.squeeze(0).cpu().numpy()
        argmax_action = int(torch.argmax(action_probs, dim=1).item()) + 1

        if step == target_step:
            # Capture the node: snapshot game + memory (post-forward, pre-act).
            node_game = copy.deepcopy(game)
            node_mem = _snapshot_memory(agent)

            trump_leads = [
                (aid, float(logits_np[aid - 1]))
                for aid in valid_sorted
                if _card_of(aid) in TRUMP_SET
            ]
            fail_leads = [
                (aid, float(logits_np[aid - 1]))
                for aid in valid_sorted
                if _card_of(aid) in FAIL_SET
            ]
            best_trump_aid, best_trump_logit = max(trump_leads, key=lambda x: x[1])
            best_fail_aid, best_fail_logit = max(fail_leads, key=lambda x: x[1])
            node = NodeInfo(
                seat=pos,
                trickIndex=int(game.current_trick),
                argmaxCard=_card_of(argmax_action) or ACTION_LOOKUP[argmax_action],
                argmaxLogit=float(logits_np[argmax_action - 1]),
                bestTrumpCard=_card_of(best_trump_aid),
                bestTrumpLogit=best_trump_logit,
                bestFailCard=_card_of(best_fail_aid),
                bestFailLogit=best_fail_logit,
                trumpLeadOptions=sorted(
                    (_card_of(a) for a, _ in trump_leads), key=TRUMP.index
                ),
                failLeadOptions=sorted(
                    (_card_of(a) for a, _ in fail_leads), key=FAIL.index
                ),
                handTrumpCount=sum(1 for c in actor.hand if c in TRUMP_SET),
                handFailCount=sum(1 for c in actor.hand if c in FAIL_SET),
                hand=sorted(actor.hand, key=lambda c: (c not in TRUMP_SET, c)),
            )

            search_outcome = None
            if teacher is not None:
                depth = (
                    rollout_depth
                    if rollout_depth is not None
                    else 6 - int(game.current_trick)  # roll to terminal early-game
                )
                res = teacher.search(
                    game,
                    pos,
                    list(forced_public),
                    det_rng,
                    d_rollout=depth,
                    seat_policies=None,  # pure self-play: all seats are this model
                )
                search_outcome = _summarize_search(
                    res,
                    argmax_action,
                    best_trump_aid,
                    best_fail_aid,
                    depth,
                    iters,
                    teacher.config.root_explore_frac,
                    min_visit_frac,
                )

            return node_game, node_mem, node, search_outcome, list(forced_public)

        if not _is_private_decision(valid):
            forced_public.append((pos, argmax_action))
        actor.act(argmax_action)
        if game.was_trick_just_completed:
            for seat in game.players:
                agent.observe(seat.get_last_trick_state_dict(), player_id=seat.position)
        step += 1

    return None, None, None, None, None


def _belief_mc(
    agent,
    teacher,
    node_game,
    observer: int,
    forced_public: List[tuple],
    trump_card: str,
    fail_card: str,
    R: int,
    pool_k: int,
    rng: random.Random,
    device,
):
    """Paired policy rollouts over the ISMCTS *belief* pool.

    Builds the exact determinized-world pool the search uses (``_build_pool``),
    resamples R worlds ~ exp(log_w) (``_pool_probs``), and -- on the SAME sampled
    worlds (common random worlds) -- forces the trump vs the fail lead and rolls
    the raw policy to terminal. This is ISMCTS's belief distribution evaluated by
    policy continuation instead of tree search.

    Returns ``(BeliefMcBranch_trump, BeliefMcBranch_fail)`` or ``(None, None)``.
    """
    teacher._rng = rng
    teacher._seat_policies = None
    saved = _snapshot_memory(agent)
    try:
        pool = teacher._build_pool(node_game, observer, list(forced_public), pool_k)
        if not pool:
            return None, None
        ess = teacher._pool_ess(pool)
        probs = teacher._pool_probs(pool)
        idxs = rng.choices(range(len(pool)), weights=probs, k=R)

        def _rollout_world(world_game, world_mem, card) -> DetBranch:
            g = copy.deepcopy(world_game)
            _restore_memory(agent, world_mem)
            g.players[observer - 1].act(ACTION_IDS[f"PLAY {card}"])
            _play_out(agent, g, device, deterministic=False)
            return _branch_metrics(g, observer)

        trump_m, fail_m = [], []
        for i in idxs:
            world_game, world_mem, _ = pool[i]
            trump_m.append(_rollout_world(world_game, world_mem, trump_card))
            fail_m.append(_rollout_world(world_game, world_mem, fail_card))
    finally:
        _restore_memory(agent, saved)

    def _branch(ms: List[DetBranch]) -> BeliefMcBranch:
        return BeliefMcBranch(
            R=R,
            poolSize=len(pool),
            ess=ess,
            defenderPointsMean=float(np.mean([m.defenderPoints for m in ms])),
            leaderScoreMean=float(np.mean([m.leaderScore for m in ms])),
            winRate=float(np.mean([m.win for m in ms])),
        )

    return _branch(trump_m), _branch(fail_m)


def _summarize_search(
    res: dict,
    argmax_aid: int,
    trump_aid: int,
    fail_aid: int,
    depth: int,
    iters: int,
    frac: float,
    min_visit_frac: float,
) -> SearchOutcome:
    root_n = res.get("root_n", {})
    root_q = res.get("root_q", {})
    valid = res.get("valid", [])
    total_n = sum(root_n.values())

    # Informational: most-visited action.
    top_n_aid = max(valid, key=lambda a: root_n.get(a, 0.0)) if valid else argmax_aid

    # Primary: best action by Q, restricted to actions with enough visits that the
    # Q estimate is meaningful (guards against a 1-visit fluke topping the ranking).
    guard = max(1.0, min_visit_frac * total_n)
    eligible = [a for a in valid if root_n.get(a, 0.0) >= guard]
    if not eligible:
        eligible = [a for a in valid if root_n.get(a, 0.0) > 0] or list(valid)
    top_q_aid = (
        max(eligible, key=lambda a: root_q.get(a, float("-inf")))
        if eligible
        else argmax_aid
    )

    ranking = sorted(
        (
            {
                "actionId": a,
                "action": ACTION_LOOKUP[a],
                "visits": round(float(root_n.get(a, 0.0)), 2),
                "q": round(float(root_q.get(a, 0.0)), 4),
            }
            for a in valid
        ),
        key=lambda d: d["q"],
        reverse=True,
    )[:4]
    top_n_card = _card_of(top_n_aid)
    top_q_card = _card_of(top_q_aid)
    return SearchOutcome(
        iters=iters,
        rootExploreFrac=frac,
        rolloutDepth=depth,
        ess=float(res.get("ess", 0.0)),
        ok=bool(res.get("ok", False)),
        nIter=int(res.get("n_iter", 0)),
        topQActionId=top_q_aid,
        topQAction=ACTION_LOOKUP[top_q_aid],
        topQValue=round(float(root_q.get(top_q_aid, 0.0)), 4),
        topQIsArgmax=(top_q_aid == argmax_aid),
        topQIsTrump=(top_q_card in TRUMP_SET),
        topQIsFail=(top_q_card in FAIL_SET),
        topActionId=top_n_aid,
        topAction=ACTION_LOOKUP[top_n_aid],
        topIsArgmax=(top_n_aid == argmax_aid),
        topIsTrump=(top_n_card in TRUMP_SET),
        topIsFail=(top_n_card in FAIL_SET),
        trumpVisits=round(float(root_n.get(trump_aid, 0.0)), 2),
        trumpQ=round(float(root_q.get(trump_aid, 0.0)), 4),
        failVisits=round(float(root_n.get(fail_aid, 0.0)), 2),
        failQ=round(float(root_q.get(fail_aid, 0.0)), 4),
        bestTrumpCard=_card_of(trump_aid),
        bestFailCard=_card_of(fail_aid),
        ranking=ranking,
    )


# ---------------------------------------------------------------------------
# Case detection
# ---------------------------------------------------------------------------
def _classify_spots(resp, seed: int, partner_mode: int, max_trick: int) -> List[dict]:
    """Every defender lead spot on tricks 0..max_trick with both a trump and a
    fail lead available, labelled by which class the policy's argmax chose."""
    spots = []
    for ad in resp.trace:
        if not ad.action.startswith("PLAY "):
            continue
        card = ad.action[5:]
        view = ad.view
        if not all(c == "" for c in (view.get("current_trick") or [])):
            continue  # leads only
        if view.get("is_leaster"):
            continue
        ti = int(view.get("current_trick_index", 0))
        if ti > max_trick:
            continue
        seat = ad.seat
        if (
            seat == (view.get("picker") or 0)
            or seat == (view.get("partner") or 0)
            or scan._is_secret_partner(view, partner_mode)
        ):
            continue
        has_trump = any(_card_of(v) in TRUMP_SET for v in ad.validActionIds)
        has_fail = any(_card_of(v) in FAIL_SET for v in ad.validActionIds)
        if not (has_trump and has_fail):
            continue  # need both classes for a trump-vs-fail comparison
        group = "trump" if card in TRUMP_SET else "fail" if card in FAIL_SET else None
        if group is None:
            continue
        spots.append(
            {
                "seed": seed,
                "partnerMode": partner_mode,
                "stepIndex": ad.stepIndex,
                "trickIndex": ti,
                "seat": seat,
                "seatName": ad.seatName,
                "pickerSeat": view.get("picker") or 0,
                "cardLed": card,
                "group": group,
            }
        )
    return spots


def _find_cases(args) -> tuple[List[dict], List[dict]]:
    """Scan the seed range; return (trump_pref, fail_pref_control) spot lists.

    FAIL-PREF controls are randomly subsampled to ``len(trump) * control_ratio``
    (seeded), mirroring the original validation script's balanced control set.
    """
    trump_spots: List[dict] = []
    fail_spots: List[dict] = []
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        req = AnalyzeSimulateRequest(
            seed=seed,
            partnerMode=args.partner_mode,
            deterministic=True,
            modelPath=args.model,
            maxSteps=args.max_steps,
        )
        resp = scan.simulate_game(req)
        for spot in _classify_spots(resp, seed, args.partner_mode, args.max_trick):
            (trump_spots if spot["group"] == "trump" else fail_spots).append(spot)

    n_control = int(round(len(trump_spots) * args.control_ratio))
    rng = random.Random(args.control_seed)
    rng.shuffle(fail_spots)
    fail_spots = fail_spots[:n_control]
    print(
        f"Scanned {args.num_seeds} seeds (from {args.start_seed}) -> "
        f"{len(trump_spots)} TRUMP-PREF, {len(fail_spots)} FAIL-PREF control "
        f"(of {n_control} target) defender leads on tricks 0-{args.max_trick}"
    )
    return trump_spots, fail_spots


# ---------------------------------------------------------------------------
# root_explore_frac / iters sweep diagnostic
# ---------------------------------------------------------------------------
def _explore_sweep(agent, teacher, spot, args, device, fracs, iters_list) -> None:
    """Re-run ISMCTS at one node across a root_explore_frac x iters grid.

    Tells us whether the search's trump preference is robust or an
    under-exploration artifact: if a higher frac gives the fail action enough
    visits and its Q rises toward / above trump (and the visit-recommendation
    flips), the default 0.25 was starving the alternative; if trump's Q stays
    ahead with ample visits, the preference is genuine.
    """
    seed, partner_mode = spot["seed"], spot["partnerMode"]
    target_step, seat = spot["stepIndex"], spot["seat"]
    node_game, _, node, _, forced_public = _replay_to_node(
        agent, seed, partner_mode, target_step, args.max_steps, device, teacher=None
    )
    if node is None or node.argmaxCard != spot["cardLed"]:
        print(
            f"  ! seed={seed} step={target_step}: node not reached / non-reproducing; skip"
        )
        return
    trump_aid = ACTION_IDS[f"PLAY {node.bestTrumpCard}"]
    fail_aid = ACTION_IDS[f"PLAY {node.bestFailCard}"]
    observer = seat
    depth = (
        args.rollout_depth if args.rollout_depth is not None else 6 - node.trickIndex
    )

    print(
        f"\nSWEEP seed={seed} trick={node.trickIndex + 1} seat {seat}  "
        f"trump {node.bestTrumpCard} (logit {node.bestTrumpLogit:+.2f}) vs "
        f"fail {node.bestFailCard} (logit {node.bestFailLogit:+.2f})  depth {depth}  "
        f"{len(node.trumpLeadOptions) + len(node.failLeadOptions)} legal leads"
    )
    print(
        f"  {'frac':>5} {'iters':>6} | {'ess':>5} | "
        f"{'trump N (share)  Q':>22} | {'fail N (share)  Q':>22} | top@N top@Q"
    )
    saved_frac = teacher.config.root_explore_frac
    saved_iters = dict(teacher.config.iters)
    try:
        for frac in fracs:
            for iters in iters_list:
                teacher.config.root_explore_frac = frac
                teacher.config.iters = {k: iters for k in teacher.config.iters}
                rng = random.Random((seed * 1000003) ^ int(frac * 1000) ^ (iters << 3))
                g = copy.deepcopy(node_game)
                res = teacher.search(
                    g,
                    observer,
                    list(forced_public),
                    rng,
                    d_rollout=depth,
                    seat_policies=None,
                )
                root_n, root_q, valid = res["root_n"], res["root_q"], res["valid"]
                total = sum(root_n.values()) or 1.0
                tn, fn = root_n.get(trump_aid, 0.0), root_n.get(fail_aid, 0.0)
                tq, fq = root_q.get(trump_aid, 0.0), root_q.get(fail_aid, 0.0)
                top_n = max(valid, key=lambda a: root_n.get(a, 0.0))
                visited = [a for a in valid if root_n.get(a, 0.0) > 0]
                top_q = (
                    max(visited, key=lambda a: root_q.get(a, 0.0)) if visited else top_n
                )
                print(
                    f"  {frac:>5.2f} {iters:>6} | {res['ess']:>5.1f} | "
                    f"{tn:>6.0f} ({tn / total * 100:>3.0f}%) {tq:>+6.3f} | "
                    f"{fn:>6.0f} ({fn / total * 100:>3.0f}%) {fq:>+6.3f} | "
                    f"{(_card_of(top_n) or ACTION_LOOKUP[top_n]):>4} "
                    f"{(_card_of(top_q) or ACTION_LOOKUP[top_q]):>4}"
                )
    finally:
        teacher.config.root_explore_frac = saved_frac
        teacher.config.iters = saved_iters


# ---------------------------------------------------------------------------
# Per-case analysis
# ---------------------------------------------------------------------------
def analyze_case(agent, teacher, spot: dict, args, device) -> Optional[CaseResult]:
    seed, partner_mode = spot["seed"], spot["partnerMode"]
    target_step, seat = spot["stepIndex"], spot["seat"]
    det_rng = random.Random(0xC0FFEE ^ (seed << 8) ^ target_step)

    search_teacher = None if args.no_search else teacher
    node_game, node_mem, node, search, forced_public = _replay_to_node(
        agent,
        seed,
        partner_mode,
        target_step,
        args.max_steps,
        device,
        teacher=search_teacher,
        det_rng=det_rng,
        iters=args.iters,
        rollout_depth=args.rollout_depth,
        min_visit_frac=args.min_visit_frac,
    )
    if node is None:
        print(f"  ! seed={seed} step={target_step}: node not reached; skipping")
        return None
    if node.argmaxCard != spot["cardLed"]:
        print(
            f"  ! seed={seed} step={target_step}: argmax {node.argmaxCard} "
            f"!= scanned {spot['cardLed']}; skipping (non-reproducing)"
        )
        return None

    # Single deterministic rollout per branch.
    det_trump = _force_and_play(
        agent, node_game, node_mem, seat, node.bestTrumpCard, device, deterministic=True
    )
    det_fail = _force_and_play(
        agent, node_game, node_mem, seat, node.bestFailCard, device, deterministic=True
    )

    # Paired Monte-Carlo (seed torch for reproducibility of the sampled rollouts).
    torch.manual_seed(0xA11CE ^ (seed << 4) ^ target_step)
    mc_trump = _mc_branch(
        agent, node_game, node_mem, seat, node.bestTrumpCard, args.rollouts, device
    )
    mc_fail = _mc_branch(
        agent, node_game, node_mem, seat, node.bestFailCard, args.rollouts, device
    )

    # Paired belief-pool Monte-Carlo (the middle rung of the ladder).
    belief_trump = belief_fail = None
    if teacher is not None and not args.no_belief_mc:
        belief_rng = random.Random(0xBE11E ^ (seed << 6) ^ target_step)
        torch.manual_seed(0xBE11E ^ (seed << 4) ^ target_step)
        pool_k = args.belief_worlds if args.belief_worlds is not None else args.iters
        belief_trump, belief_fail = _belief_mc(
            agent,
            teacher,
            node_game,
            seat,
            forced_public,
            node.bestTrumpCard,
            node.bestFailCard,
            args.rollouts,
            pool_k,
            belief_rng,
            device,
        )

    return CaseResult(
        seed=seed,
        partnerMode=partner_mode,
        stepIndex=target_step,
        trickIndex=spot["trickIndex"],
        seat=seat,
        seatName=spot["seatName"],
        pickerSeat=spot["pickerSeat"],
        group=spot["group"],
        hand=node.hand,
        node=node,
        detTrump=det_trump,
        detFail=det_fail,
        detDeltaPoints=det_trump.defenderPoints - det_fail.defenderPoints,
        detDeltaScore=det_trump.leaderScore - det_fail.leaderScore,
        mcTrump=mc_trump,
        mcFail=mc_fail,
        mcDeltaPoints=mc_trump.defenderPointsMean - mc_fail.defenderPointsMean,
        mcDeltaScore=mc_trump.leaderScoreMean - mc_fail.leaderScoreMean,
        mcDeltaWin=mc_trump.winRate - mc_fail.winRate,
        beliefMcTrump=belief_trump,
        beliefMcFail=belief_fail,
        beliefMcDeltaPoints=(
            belief_trump.defenderPointsMean - belief_fail.defenderPointsMean
            if belief_trump
            else None
        ),
        beliefMcDeltaScore=(
            belief_trump.leaderScoreMean - belief_fail.leaderScoreMean
            if belief_trump
            else None
        ),
        beliefMcDeltaWin=(
            belief_trump.winRate - belief_fail.winRate if belief_trump else None
        ),
        search=search,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_case(r: CaseResult) -> str:
    lines = [
        f"[{r.group.upper()}] seed={r.seed} step={r.stepIndex} trick={r.trickIndex + 1} "
        f"{r.seatName}(seat {r.seat}) vs picker {r.pickerSeat}  "
        f"argmax {r.node.argmaxCard}  trump={r.node.bestTrumpCard} fail={r.node.bestFailCard} "
        f"({r.node.handTrumpCount}T/{r.node.handFailCount}F in hand)",
        f"    det  : trumpPts={r.detTrump.defenderPoints} failPts={r.detFail.defenderPoints} "
        f"=> dPts={r.detDeltaPoints:+d} dScore={r.detDeltaScore:+d}",
        f"    mc{r.mcTrump.R:<3} (true deal): trumpPts={r.mcTrump.defenderPointsMean:5.1f} "
        f"failPts={r.mcFail.defenderPointsMean:5.1f} => dPts={r.mcDeltaPoints:+5.1f} "
        f"dScore={r.mcDeltaScore:+5.2f} dWin={r.mcDeltaWin * 100:+.0f}%",
    ]
    if r.beliefMcTrump is not None:
        b = r.beliefMcTrump
        lines.append(
            f"    bmc{b.R:<3} (belief, K={b.poolSize} ess {b.ess:.0f}): "
            f"trumpPts={r.beliefMcTrump.defenderPointsMean:5.1f} "
            f"failPts={r.beliefMcFail.defenderPointsMean:5.1f} => "
            f"dPts={r.beliefMcDeltaPoints:+5.1f} dScore={r.beliefMcDeltaScore:+5.2f} "
            f"dWin={r.beliefMcDeltaWin * 100:+.0f}%"
        )
    if r.search is not None:
        s = r.search

        def _verdict(card: str, is_t: bool, is_f: bool) -> str:
            return "TRUMP " + card if is_t else (("FAIL " + card) if is_f else card)

        primary = _verdict(
            s.topQAction[5:] if s.topQAction.startswith("PLAY ") else s.topQAction,
            s.topQIsTrump,
            s.topQIsFail,
        )
        info = _verdict(
            s.topAction[5:] if s.topAction.startswith("PLAY ") else s.topAction,
            s.topIsTrump,
            s.topIsFail,
        )
        lines.append(
            f"    ismcts({s.iters}it f{s.rootExploreFrac:g} ess {s.ess:.0f}"
            f"{'' if s.ok else ' LOW'}): top@Q={primary} (Q={s.topQValue:+.3f})  "
            f"[top@N={info}]  trump N={s.trumpVisits} Q={s.trumpQ:+.3f} | "
            f"fail N={s.failVisits} Q={s.failQ:+.3f}"
        )
    return "\n".join(lines)


def _se(vals: np.ndarray) -> float:
    n = len(vals)
    return float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def summarize_group(label: str, results: List[CaseResult], iters: int) -> None:
    n = len(results)
    print("\n" + "=" * 72)
    print(f"{label}  (n = {n} states)")
    print("=" * 72)
    if not n:
        print("  (no states)")
        return

    # Paired Monte-Carlo, Delta = trump - fail.
    print(
        f"  Paired Monte-Carlo (Δ = trump − fail, {results[0].mcTrump.R} rollouts/branch):"
    )
    for key, name, scale, unit in [
        ("mcDeltaPoints", "Defender card points", 1, "pts"),
        ("mcDeltaScore", "Leader game score", 1, "score"),
        ("mcDeltaWin", "Defender win rate", 100, "%"),
    ]:
        vals = np.array([getattr(r, key) for r in results], dtype=float)
        frac_pos = float(np.mean(vals > 0))
        print(
            f"    {name:<22} {vals.mean() * scale:+7.2f} {unit:<5} "
            f"(SE {_se(vals) * scale:5.2f}, trump better in {frac_pos * 100:.0f}% of states)"
        )
    tp = np.mean([r.mcTrump.defenderPointsMean for r in results])
    fp = np.mean([r.mcFail.defenderPointsMean for r in results])
    print(
        f"    Absolute EV: trump-lead {tp:.1f} pts vs fail-lead {fp:.1f} pts (need 60 to win)"
    )

    # Belief-pool Monte-Carlo: same policy continuation, but over the search's
    # determinized belief worlds instead of the true deal (isolates hindsight).
    belief = [r for r in results if r.beliefMcTrump is not None]
    if belief:
        print(
            f"  Belief-pool Monte-Carlo (Δ = trump − fail, policy continuation over "
            f"determinized worlds; n={len(belief)}):"
        )
        for key, name, scale, unit in [
            ("beliefMcDeltaPoints", "Defender card points", 1, "pts"),
            ("beliefMcDeltaScore", "Leader game score", 1, "score"),
            ("beliefMcDeltaWin", "Defender win rate", 100, "%"),
        ]:
            vals = np.array([getattr(r, key) for r in belief], dtype=float)
            frac_pos = float(np.mean(vals > 0))
            print(
                f"    {name:<22} {vals.mean() * scale:+7.2f} {unit:<5} "
                f"(SE {_se(vals) * scale:5.2f}, trump better in {frac_pos * 100:.0f}% of states)"
            )
        btp = np.mean([r.beliefMcTrump.defenderPointsMean for r in belief])
        bfp = np.mean([r.beliefMcFail.defenderPointsMean for r in belief])
        mean_ess = np.mean([r.beliefMcTrump.ess for r in belief])
        print(
            f"    Absolute EV: trump-lead {btp:.1f} pts vs fail-lead {bfp:.1f} pts "
            f"(mean belief ESS {mean_ess:.1f})"
        )

    # Single deterministic rollout, for comparison with the MC estimate.
    dpts = np.array([r.detDeltaPoints for r in results], dtype=float)
    dsc = np.array([r.detDeltaScore for r in results], dtype=float)
    print(
        f"  Single deterministic rollout: Δpts {dpts.mean():+.2f} (SE {_se(dpts):.2f}), "
        f"Δscore {dsc.mean():+.3f} (SE {_se(dsc):.3f}), trump better in "
        f"{float(np.mean(dpts > 0)) * 100:.0f}% of states"
    )

    # By trump count in the leading hand (MC).
    print("  --- by trump count in hand (MC) ---")
    for nt in sorted({r.node.handTrumpCount for r in results}):
        sub = [r for r in results if r.node.handTrumpCount == nt]
        dp = np.mean([r.mcDeltaPoints for r in sub])
        ds = np.mean([r.mcDeltaScore for r in sub])
        print(f"    {nt} trump (n={len(sub):3d}): Δpts {dp:+6.2f}, Δscore {ds:+6.3f}")

    # ISMCTS tallies. PRIMARY verdict is by Q (visit counts are prior-dominated).
    searched = [r for r in results if r.search is not None]
    if searched:
        ok = [r for r in searched if r.search.ok]
        frac = searched[0].search.rootExploreFrac
        q_trump = sum(1 for r in ok if r.search.topQIsTrump)
        q_fail = sum(1 for r in ok if r.search.topQIsFail)
        q_other = len(ok) - q_trump - q_fail
        q_agree = sum(1 for r in ok if r.search.topQIsArgmax)
        n_trump = sum(1 for r in ok if r.search.topIsTrump)
        n_fail = sum(1 for r in ok if r.search.topIsFail)
        print(
            f"  ISMCTS @ {iters}it frac {frac:g} (ESS-valid {len(ok)}/{len(searched)}):"
        )
        print(
            f"    by Q (primary): top is trump {q_trump}, fail {q_fail}, other {q_other}; "
            f"agrees with policy argmax {q_agree}/{len(ok)}"
        )
        print(
            f"    by visits (info): top is trump {n_trump}, fail {n_fail}, "
            f"other {len(ok) - n_trump - n_fail}"
        )


def print_examples(results: List[CaseResult], label: str, n: int = 6) -> None:
    if not results:
        return
    print(f"\n--- {label}: example states (largest |MC Δscore|) ---")
    for r in sorted(results, key=lambda r: -abs(r.mcDeltaScore))[:n]:
        s = ""
        if r.search is not None:
            tag = "T" if r.search.topQIsTrump else ("F" if r.search.topQIsFail else "?")
            s = f" | ismcts top@Q={tag}"
        print(
            f"  seed={r.seed} trick={r.trickIndex + 1} {r.node.bestTrumpCard} vs {r.node.bestFailCard} "
            f"| MC Δpts {r.mcDeltaPoints:+5.1f} Δscore {r.mcDeltaScore:+5.2f} "
            f"(trump {r.mcTrump.defenderPointsMean:.0f}pts/{r.mcTrump.leaderScoreMean:+.2f} vs "
            f"fail {r.mcFail.defenderPointsMean:.0f}pts/{r.mcFail.leaderScoreMean:+.2f}){s}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--partner-mode", type=int, choices=[0, 1], default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--max-trick",
        type=int,
        default=1,
        help="Analyze defender leads at or below this 0-based trick (default 1 = tricks 0-1).",
    )
    parser.add_argument(
        "--rollouts", type=int, default=50, help="Monte-Carlo rollouts/branch."
    )
    parser.add_argument(
        "--iters", type=int, default=512, help="ISMCTS iterations (audit default 512)."
    )
    parser.add_argument(
        "--root-explore-frac",
        type=float,
        default=1.0,
        help="ISMCTS root prior uniform-mix fraction. 1.0 (audit default) flattens the "
        "root prior so visits track value, not the policy's (biased) confidence.",
    )
    parser.add_argument(
        "--min-visit-frac",
        type=float,
        default=0.01,
        help="Min share of visits an action needs to be eligible for the top@Q verdict.",
    )
    parser.add_argument(
        "--rollout-depth",
        type=int,
        default=None,
        help="Search rollout depth; default rolls to terminal (6 - trick).",
    )
    parser.add_argument(
        "--control-ratio",
        type=float,
        default=1.0,
        help="FAIL-PREF controls collected = TRUMP-PREF count * this (default 1.0).",
    )
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument(
        "--belief-worlds",
        type=int,
        default=None,
        help="Belief-pool size for belief-MC; default = --iters (matches the search pool).",
    )
    parser.add_argument("--no-search", action="store_true", help="Skip ISMCTS.")
    parser.add_argument(
        "--no-belief-mc",
        action="store_true",
        help="Skip the belief-pool Monte-Carlo rung.",
    )
    parser.add_argument(
        "--no-control", action="store_true", help="Skip FAIL-PREF control group."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap TRUMP-PREF cases analyzed."
    )
    parser.add_argument("--out", default=None)
    # root_explore_frac / iters sweep diagnostic mode.
    parser.add_argument(
        "--explore-sweep",
        action="store_true",
        help="Diagnostic: re-run ISMCTS at each target node across a frac x iters grid.",
    )
    parser.add_argument(
        "--sweep-seeds",
        default="266,679",
        help="Comma-separated seeds whose trump-pref node(s) to sweep (explore-sweep mode).",
    )
    parser.add_argument("--sweep-fracs", default="0.25,0.5,1.0")
    parser.add_argument("--sweep-iters", default="384,1536")
    args = parser.parse_args()

    device = _device()
    agent = scan._cached_load_agent(args.model)

    # The teacher is needed for ISMCTS search AND for the belief-pool MC (it owns
    # the determinizer / belief-weighting). Build it if either is enabled.
    teacher = None
    if args.explore_sweep or not (args.no_search and args.no_belief_mc):
        from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

        cfg = ISMCTSConfig()
        cfg.iters = {k: args.iters for k in cfg.iters}
        cfg.root_explore_frac = args.root_explore_frac
        teacher = ISMCTSTeacher(agent, cfg)

    # ---- Diagnostic: root_explore_frac / iters sweep, then exit ------------
    if args.explore_sweep:
        fracs = [float(x) for x in args.sweep_fracs.split(",")]
        iters_list = [int(x) for x in args.sweep_iters.split(",")]
        sweep_seeds = [int(s) for s in args.sweep_seeds.split(",")]
        print(
            f"Exploration sweep: fracs={fracs} iters={iters_list} on seeds {sweep_seeds}\n"
        )
        for seed in sweep_seeds:
            req = AnalyzeSimulateRequest(
                seed=seed,
                partnerMode=args.partner_mode,
                deterministic=True,
                modelPath=args.model,
                maxSteps=args.max_steps,
            )
            resp = scan.simulate_game(req)
            spots = [
                s
                for s in _classify_spots(resp, seed, args.partner_mode, args.max_trick)
                if s["group"] == "trump"
            ]
            if not spots:
                print(
                    f"seed={seed}: no trump-pref defender lead on tricks 0-{args.max_trick}"
                )
                continue
            for spot in spots:
                _explore_sweep(agent, teacher, spot, args, device, fracs, iters_list)
        print(
            "\nReading: if raising frac lifts fail's visit-share and Q toward/above "
            "trump (and top@N flips), the default 0.25 was under-exploring; if trump's "
            "Q stays ahead with ample fail visits, the preference is genuine."
        )
        return 0

    trump_spots, fail_spots = _find_cases(args)
    if args.limit is not None:
        trump_spots = trump_spots[: args.limit]
    if args.no_control:
        fail_spots = []

    def run(spots, tag):
        out = []
        print(f"\n>>> Analyzing {len(spots)} {tag} case(s)\n")
        for spot in spots:
            r = analyze_case(agent, teacher, spot, args, device)
            if r is not None:
                out.append(r)
                print(_fmt_case(r) + "\n")
        return out

    trump_results = run(trump_spots, "TRUMP-PREF")
    fail_results = run(fail_spots, "FAIL-PREF control")

    summarize_group(
        "TRUMP-PREF defender leads (behavior under scrutiny)", trump_results, args.iters
    )
    print_examples(trump_results, "TRUMP-PREF")
    summarize_group(
        "FAIL-PREF defender leads (control / method check)", fail_results, args.iters
    )

    print(
        "\nInterpretation (Δ = trump − fail). Three-rung ladder to localize any "
        "true-deal-MC vs ISMCTS disagreement:\n"
        "  1. true-deal MC  -> policy continuation, the ONE real deal (hindsight).\n"
        "  2. belief-pool MC -> policy continuation, determinized belief worlds.\n"
        "  3. ISMCTS         -> SEARCH continuation, determinized belief worlds.\n"
        "  (1)->(2) gap = hindsight / true-deal luck; (2)->(3) gap = search "
        "continuation optimism over the raw policy. TRUMP-PREF Δ>0 = trump genuinely "
        "wins; FAIL-PREF should be Δ≤0 if the measurement is sound.\n"
        "ISMCTS verdict is read by Q (top@Q), not visit counts: with a flattened "
        "root prior the visit counts are still concentrated by exploration noise, "
        "but Q is the value estimate we actually want."
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "model": args.model,
                        "partnerMode": args.partner_mode,
                        "numSeeds": args.num_seeds,
                        "startSeed": args.start_seed,
                        "maxTrick": args.max_trick,
                        "rollouts": args.rollouts,
                        "iters": args.iters,
                        "rootExploreFrac": args.root_explore_frac,
                        "minVisitFrac": args.min_visit_frac,
                        "beliefWorlds": args.belief_worlds
                        if args.belief_worlds is not None
                        else args.iters,
                        "controlRatio": args.control_ratio,
                    },
                    "trumpPref": [asdict(r) for r in trump_results],
                    "failPref": [asdict(r) for r in fail_results],
                },
                indent=2,
            )
        )
        print(f"\nWrote report -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
