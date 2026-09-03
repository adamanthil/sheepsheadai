#!/usr/bin/env python3
"""Shared game-generation primitives for population/league training.

This module holds the single-game playing primitive (``play_population_game``)
and the small pure helpers (head classification, schedule interpolation, public
post-game summary) that the league trainer (``train_league_ppo.py``), the
exploiter (``exploiter.py``) and the deploy/audit ISMCTS probes build on. The
old PFSP museum (dual-population management, strategic-profile clustering, the
``run_pfsp_training`` loop and its worker pool) was dissolved in the June 2026
consolidation; ``train_league_ppo.py`` owns the training loop and its own
worker pool now. Hyperparameters live in config.py; pure helpers in
training_utils.py.
"""

import random
from dataclasses import dataclass

import numpy as np

from sheepshead import (
    Game,
)
from sheepshead.agent.ppo import PPOAgent
from sheepshead.ismcts import _minmax_unit, infer_head
from sheepshead.training.training_utils import (
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
    handle_trick_completion,
    process_episode_rewards,
    process_terminal_rewards,
    update_intermediate_rewards_for_action,
)


def play_cell(game, player) -> str:
    """Node-class label for a PLAY decision: ``t{trick}-{role}-{lead|follow}``.

    The E9 stratification (Search_Teacher_Design §3): the three axes every
    prior study conditioned on, all computable from public state at decision
    time — which is what lets the gated teacher classify nodes online.
    """
    pos = player.position
    if pos == game.picker:
        role = "picker"
    elif pos == game.partner or player.is_secret_partner:
        role = "partner"
    else:
        role = "defender"
    kind = (
        "lead" if all(c == "" for c in game.history[game.current_trick]) else "follow"
    )
    return f"t{game.current_trick}-{role}-{kind}"


def _search_head(valid_actions) -> str:
    """Classify a decision into a search head (delegates to ismcts.infer_head)."""
    return infer_head(valid_actions)


def interpolated_weight(schedule: dict, progress_pct: float) -> float:
    """Linear interpolation of schedule weights by percent progress.

    schedule: mapping of percent (0-100) to weight.
    progress_pct: current percent progress in [0, 100].
    """
    if not schedule:
        return 1.0
    # Normalize and sort points by percent
    points = sorted((float(k), float(v)) for k, v in schedule.items())
    # Clamp to endpoints
    if progress_pct <= points[0][0]:
        return points[0][1]
    if progress_pct >= points[-1][0]:
        return points[-1][1]
    # Find segment and interpolate
    for (k0, v0), (k1, v1) in zip(points, points[1:]):
        if k0 <= progress_pct <= k1:
            if k1 == k0:
                return v1
            t = (progress_pct - k0) / (k1 - k0)
            return v0 + t * (v1 - v0)
    # Fallback (should not hit due to clamps)
    return points[-1][1]


def _setup_seats(
    training_agent: PPOAgent,
    opponents: list,
    training_agent_position: int,
) -> tuple[list, dict]:
    """Build the position-to-agent seat mapping (shuffling which opponent sits
    where) and derive the position->pop-agent map from it."""
    # Create position-to-agent mapping; all five seats must be populated by training + 4 opponents
    agents: list = [None] * 5
    agents[training_agent_position - 1] = training_agent

    # Randomize which opponent sits in which non-training seat to reduce seat-assignment bias
    opponent_seat_positions = [
        pos for pos in range(1, 6) if pos != training_agent_position
    ]
    random.shuffle(opponent_seat_positions)
    for opponent, seat_pos in zip(opponents[:4], opponent_seat_positions):
        agents[seat_pos - 1] = opponent.agent

    # Map positions to population opponents (returned for the caller's bookkeeping)
    pos_to_pop_agent = {}
    opp_positions = opponent_seat_positions.copy()
    for opp, seat_pos in zip(opponents[: len(opp_positions)], opp_positions):
        pos_to_pop_agent[seat_pos] = opp

    return agents, pos_to_pop_agent


@dataclass(frozen=True)
class CommitteeSummary:
    """Pooled root statistics of one committee search over a legal set.

    This is the committee's evidence in the form every downstream consumer
    reads: the CE target builder (``build_ce_search_target``), the corpus
    generator's schema-2 rows (``distill_corpus.py``) and the search-Q
    policy-iteration trainer (CE_Teacher_Design §20). All vectors are
    aligned to ``actions`` (= ``sorted(valid_actions)``), Q in reward
    units (score / 12).

    Attributes:
        actions: the legal action ids, sorted.
        q_mean: pooled completed-Q per action — the replicate mean over
            replicates that visited the action; UNVISITED actions are
            completed with the visit-weighted mean of the visited ones
            (Gumbel MuZero's completed-Q, Danihelka et al. 2022), so their
            centered deviation is exactly zero.
        q_var: sampling variance of each pooled mean (the §1.2 hierarchical
            blend of per-action replicate variance with the global
            calibration, divided by the number of pooled observations).
            Unvisited actions carry one replicate's worth of the global
            variance (no pooling happened) — ``n_mean`` = 0 marks them.
        n_mean: per-action mean root visit count across replicates.
        prior: pooled UNMIXED root prior (the label-time policy averaged
            over the replicates' determinized worlds).
        noise_var: mean of ``q_var`` over visited actions — the
            shrinkage's noise term.
        signal_var: variance of ``q_mean`` across the legal set.
        w: positive-part James-Stein factor max(0, 1 - noise_var /
            signal_var); 0 when the spread is within replicate noise.
        gap: top-2 separation of ``q_mean`` (0 for a single action).
        spread: max - min of ``q_mean``.
        max_visits: mean over replicates of the max root visit count (the
            pi_gumbel tilt-scale input).
    """

    actions: tuple[int, ...]
    q_mean: np.ndarray
    q_var: np.ndarray
    n_mean: np.ndarray
    prior: np.ndarray
    noise_var: float
    signal_var: float
    w: float
    gap: float
    spread: float
    max_visits: float

    def as_row_fields(self) -> dict:
        """The schema-2 corpus row fields (CE_Teacher_Design §20.3), plain
        Python lists/floats so shards stay torch.load-able without numpy
        pickles."""
        return {
            "search_q": [float(x) for x in self.q_mean],
            "search_q_var": [float(x) for x in self.q_var],
            "search_n": [float(x) for x in self.n_mean],
            "search_prior": [float(x) for x in self.prior],
            "search_noise_var": float(self.noise_var),
            "search_w": float(self.w),
            "search_gap": float(self.gap),
            "search_spread": float(self.spread),
            "search_stats_source": "committee",
        }


def summarize_committee(
    replicates: list,
    valid_actions,
    *,
    shrink_nu: float,
    shrink_s2_global: float,
) -> "CommitteeSummary | None":
    """Pool a committee's per-replicate root statistics into a
    ``CommitteeSummary`` (see its docstring for every field), or ``None``
    when fewer than two replicates produced usable root statistics
    (replicate variance needs at least two committee opinions) or no
    action was visited at all.

    Construction (CE_Teacher_Design §1.1-§1.2, numerically identical to
    the pre-§20 target builder):

    1. **Pool.** Per action, the replicate mean over replicates that visited
       it; unvisited actions are completed with the visit-weighted mean.
    2. **Noise.** Per action, the hierarchical blend
       (nu * s2_global + (n_obs - 1) * s2_node) / (nu + n_obs - 1), divided
       by n_obs — the sampling variance OF THE POOLED MEAN; the node's noise
       term is its mean over visited actions.
    3. **Shrink.** w = max(0, 1 - noise_var / signal_var).
    """
    acts = sorted(valid_actions)
    usable = [
        r
        for r in replicates
        if r["ok"] and r.get("root_q") is not None and r.get("root_prior") is not None
    ]
    if len(usable) < 2:
        return None

    q_obs = {
        a: [float(r["root_q"][a]) for r in usable if r["root_n"].get(a, 0.0) > 0.0]
        for a in acts
    }
    n_pool = {
        a: float(np.mean([r["root_n"].get(a, 0.0) for r in usable])) for a in acts
    }
    visited = [a for a in acts if q_obs[a]]
    if not visited:
        return None
    q_mean = {a: float(np.mean(q_obs[a])) for a in visited}
    v_mix = float(
        sum(n_pool[a] * q_mean[a] for a in visited)
        / max(sum(n_pool[a] for a in visited), 1e-12)
    )
    q_bar = np.array([q_mean.get(a, v_mix) for a in acts], dtype=np.float64)

    def pooled_mean_variance(obs: list) -> float:
        n_obs = len(obs)
        s2_node = float(np.var(obs, ddof=1)) if n_obs >= 2 else 0.0
        s2_blend = (shrink_nu * shrink_s2_global + (n_obs - 1) * s2_node) / (
            shrink_nu + n_obs - 1
        )
        return s2_blend / n_obs

    q_var = np.array(
        [
            pooled_mean_variance(q_obs[a]) if q_obs[a] else float(shrink_s2_global)
            for a in acts
        ],
        dtype=np.float64,
    )
    noise_var = float(np.mean([pooled_mean_variance(q_obs[a]) for a in visited]))
    signal_var = float(np.var(q_bar))
    shrink_w = max(0.0, 1.0 - noise_var / signal_var) if signal_var > 0.0 else 0.0

    prior = np.array(
        [np.mean([r["root_prior"][a] for r in usable]) for a in acts],
        dtype=np.float64,
    )
    max_visits = float(np.mean([max(r["root_n"].values() or [0.0]) for r in usable]))
    q_sorted = np.sort(q_bar)[::-1]
    return CommitteeSummary(
        actions=tuple(acts),
        q_mean=q_bar,
        q_var=q_var,
        n_mean=np.array([n_pool[a] for a in acts], dtype=np.float64),
        prior=prior,
        noise_var=noise_var,
        signal_var=signal_var,
        w=shrink_w,
        gap=float(q_sorted[0] - q_sorted[1]) if len(q_sorted) >= 2 else 0.0,
        spread=float(q_bar.max() - q_bar.min()),
        max_visits=max_visits,
    )


def build_ce_search_target(
    replicates: list,
    valid_actions,
    *,
    shrink_nu: float,
    shrink_s2_global: float,
    gumbel_c_visit: float,
    gumbel_c_scale: float,
    base_prior: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, dict] | None":
    """Committee-pooled CE target (CE_Teacher_Design §1.1-§1.2): the
    pi_gumbel deployment readout evaluated on the James-Stein-shrunk
    committee Q vector. Returns ``(target, info)`` with ``target`` a
    float32 distribution aligned to ``sorted(valid_actions)`` and ``info``
    the per-node telemetry scalars, or ``None`` when the committee produced
    no usable root statistics.

    The pooling and shrinkage live in ``summarize_committee``; this function
    adds only the tilt:

    target = softmax(log p_raw + scale * w * minmax_unit(q̄)) with p_raw the
    pooled UNMIXED root prior and scale = (c_visit + mean-per-replicate max
    N) * c_scale — the engine's pi_gumbel readout, so act-time and
    train-time semantics never diverge. The shrink factor multiplies the
    min-max NORMALIZED vector rather than preceding the normalization:
    min-max is affine-invariant, so ``minmax(w * (q̄ - mean))`` would erase
    every w except w = 0 — multiplying afterwards is what makes the tilt
    sharpen continuously with evidence, flat at w = 0 (target = prior, CE
    gradient ~ 0: abstention is the target's fixed point) and exactly the
    deployment readout at w = 1.

    ``base_prior`` (aligned to ``sorted(valid_actions)``) replaces the
    pooled root prior as the tilt baseline when given. The pooled prior
    is a mixture of the expert's policy over sampled worlds and is
    therefore softer than the acting policy at the observed info-state
    (Jensen), so with the default baseline the w = 0 fixed point is the
    POOLED PRIOR, not the student — a persistent entropy-raising CE pull
    at abstention rows. Passing the student's own label-time policy
    makes abstention exactly zero-gradient.

    KNOWN LIMITATION (CE_Teacher_Design §20.1): the tilt's sharpness is a
    visit-count scale, not a confidence in top-2 order, and w gates on
    whole-set spread — at near-tie leads this yields ~one-hot targets on
    coin-flip cards. The §20 policy-iteration trainer builds its targets
    from ``CommitteeSummary`` through a noise-calibrated pooled advantage
    instead; this readout stays as the ACT-time readout and the legacy
    ``search_target`` row field.
    """
    summary = summarize_committee(
        replicates,
        valid_actions,
        shrink_nu=shrink_nu,
        shrink_s2_global=shrink_s2_global,
    )
    if summary is None:
        return None
    return tilt_summary_to_target(
        summary,
        gumbel_c_visit=gumbel_c_visit,
        gumbel_c_scale=gumbel_c_scale,
        base_prior=base_prior,
    )


def tilt_summary_to_target(
    summary: CommitteeSummary,
    *,
    gumbel_c_visit: float,
    gumbel_c_scale: float,
    base_prior: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, dict]":
    """The pi_gumbel tilt of ``build_ce_search_target`` applied to an
    already-pooled ``CommitteeSummary``. Returns ``(target, info)``;
    ``info`` carries ``w`` / ``spread`` / ``gap`` for telemetry."""
    if base_prior is not None:
        prior = np.asarray(base_prior, dtype=np.float64)
        prior = prior / max(prior.sum(), 1e-12)
    else:
        prior = summary.prior
    scale = (gumbel_c_visit + summary.max_visits) * gumbel_c_scale
    logits = np.log(np.clip(prior, 1e-12, None)) + scale * summary.w * _minmax_unit(
        summary.q_mean
    )
    target = np.exp(logits - logits.max())
    target /= target.sum()
    info = {"w": summary.w, "spread": summary.spread, "gap": summary.gap}
    return target.astype(np.float32), info


def _finalize_rewards(
    episode_transitions: list,
    final_scores: list,
    is_leaster: bool,
    shaped: bool,
) -> list:
    # Compute rewards for training agent actions. Shaped: intermediate + final
    # (+ leaster bonus). Terminal: final_score-only on the last action, no shaping
    # and no leaster bonus (get_score scores leasters correctly).
    reward_fn = process_episode_rewards if shaped else process_terminal_rewards
    # Both reward functions require "a chronological list of action
    # transitions for one player only" — the terminal return belongs on each
    # player's own last action. Group the merged multi-seat list per player:
    # feeding it whole put the single terminal reward on the globally-last
    # actor and left every other collecting seat's stream all-zero.
    reward_map = {}
    by_player: dict[int, list] = {}
    for t in episode_transitions:
        if t["kind"] == "action":
            by_player.setdefault(t["player"].position, []).append(t)
    for group in by_player.values():
        for reward_data in reward_fn(group, final_scores, is_leaster):
            reward_map[id(reward_data["transition"])] = reward_data["reward"]

    # Build final episode event stream for storage
    episode_events = []
    for ev in episode_transitions:
        if ev["kind"] == "observation":
            out = {
                "kind": "observation",
                "state": ev["state"],
                "player_id": ev["player"].position,
            }
        else:
            seat_pos = ev["player"].position
            out = {
                "kind": "action",
                "state": ev["state"],
                "action": ev["action"],
                "log_prob": ev["log_prob"],
                "value": ev["value"],
                "valid_actions": ev["valid_actions"],
                "reward": reward_map[id(ev)],
                "player_id": seat_pos,
                "win_label": 1.0 if final_scores[seat_pos - 1] > 0 else 0.0,
                "final_return_label": float(final_scores[seat_pos - 1]),
                "secret_partner_label": ev.get("secret_partner_label", 0.0),
                "points_label": ev.get("points_label", None),
                "seen_trump_mask_label": ev.get("seen_trump_mask_label", None),
                "unseen_trump_higher_than_hand_label": ev.get(
                    "unseen_trump_higher_than_hand_label", None
                ),
            }
        if ev.get("oracle_state") is not None:
            out["oracle_state"] = ev["oracle_state"]
        episode_events.append(out)
    return episode_events


def play_population_game(
    training_agent: PPOAgent,
    opponents: list,
    partner_mode: int,
    training_agent_position: int = 1,
    shaping_weights: dict | None = None,
    reward_mode: str = "shaped",
    collect_oracle: bool = False,
    game_seed: int | None = None,
) -> tuple:
    """Play a single game with the training agent and population opponents.

    ``reward_mode`` selects the return: ``"shaped"`` applies the intermediate
    reward shaping + per-trick rewards and ``process_episode_rewards`` (the
    bootstrap phase); ``"terminal"`` skips all shaping and uses
    ``process_terminal_rewards`` (final_score-only; the league phases).

    ``collect_oracle``: attach a full-information ``oracle_state`` (captured at
    decision time, while the Game holds the hidden cards) to every training-agent
    event, for the privileged critic's GAE baseline (critic_mode="oracle").
    When False the event dicts are byte-identical to the historical schema.

    Returns:
        tuple: (game, episode_events, final_scores, training_agent_data, opponents_by_position)
    """
    game = (
        Game(partner_selection_mode=partner_mode, seed=game_seed)
        if game_seed is not None
        else Game(partner_selection_mode=partner_mode)
    )
    weights = shaping_weights or {"pick": 1.0, "partner": 1.0, "bury": 1.0, "play": 1.0}
    shaped = reward_mode == "shaped"

    # Reset recurrent states for all agents
    training_agent.reset_recurrent_state()
    for opponent in opponents:
        opponent.agent.reset_recurrent_state()

    agents, pos_to_pop_agent = _setup_seats(
        training_agent, opponents, training_agent_position
    )

    # Store transitions only for the training agent
    episode_transitions = []
    current_trick_transitions = []

    while not game.is_done():
        for player in game.players:
            current_agent = agents[player.position - 1]
            valid_actions = player.get_valid_action_ids()

            while valid_actions:
                state = player.get_state_dict()

                # Get action from appropriate agent
                if current_agent == training_agent:
                    action, log_prob, value = current_agent.act(
                        state, valid_actions, player.position
                    )

                    # Store transition for training agent
                    transition = {
                        "kind": "action",
                        "player": player,
                        "state": state,
                        "action": action,
                        "log_prob": log_prob,
                        "value": value,
                        "valid_actions": valid_actions.copy(),
                        "intermediate_reward": 0.0,
                        "secret_partner_label": 1.0
                        if player.is_secret_partner
                        else 0.0,
                        "points_label": compute_known_points_rel(player),
                        "seen_trump_mask_label": compute_seen_trump_mask(player),
                        "unseen_trump_higher_than_hand_label": compute_any_unseen_trump_higher_than_hand(
                            player
                        ),
                    }
                    if collect_oracle:
                        transition["oracle_state"] = player.get_oracle_state_dict()
                    episode_transitions.append(transition)

                    if shaped:
                        # Shared intermediate reward shaping and trick tracking
                        update_intermediate_rewards_for_action(
                            game,
                            player,
                            action,
                            transition,
                            current_trick_transitions,
                            pick_weight=weights["pick"],
                            partner_weight=weights["partner"],
                            bury_weight=weights["bury"],
                            play_weight=weights["play"],
                        )

                else:
                    # Opponent action (stochastic for diversity)
                    action, _, _ = current_agent.act(
                        state, valid_actions, player.position, deterministic=False
                    )

                player.act(action)

                # Handle trick completion; PFSP-specific observation propagation
                trick_completed = handle_trick_completion(
                    game, current_trick_transitions
                )
                if trick_completed and not game.is_done():
                    # Emit observations for the completed trick using dedicated accessor
                    for seat in game.players:
                        seat_agent = agents[seat.position - 1]
                        if seat_agent == training_agent:
                            # Update training agent's recurrent hidden state and also store for unroll
                            training_agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
                            obs_transition = {
                                "kind": "observation",
                                "player": seat,
                                "state": seat.get_last_trick_state_dict(),
                            }
                            if collect_oracle:
                                obs_transition["oracle_state"] = (
                                    seat.get_last_trick_oracle_state_dict()
                                )
                            episode_transitions.append(obs_transition)
                        else:
                            seat_agent.observe(
                                seat.get_last_trick_state_dict(), seat.position
                            )

                valid_actions = player.get_valid_action_ids()

    final_scores = [player.get_score() for player in game.players]

    # Return training agent specific data
    training_agent_score = final_scores[training_agent_position - 1]
    was_picker = game.picker == training_agent_position

    training_agent_data = {
        "score": training_agent_score,
        "was_picker": was_picker,
        "position": training_agent_position,
    }

    episode_events = _finalize_rewards(
        episode_transitions, final_scores, game.is_leaster, shaped
    )

    return (
        game,
        episode_events,
        final_scores,
        training_agent_data,
        dict(pos_to_pop_agent),
    )


def make_game_summary(game) -> dict:
    """Normalize the post-game public fields the training driver needs into a plain,
    picklable dict, so the per-episode bookkeeping path is identical whether the game
    was played in-process (sequential) or in a worker (parallel).

    seat_roles maps every seat (1-5) to picker/partner/defender/leaster using the same
    logic as the original role-perf loop.
    """
    is_leaster = bool(game.is_leaster)
    is_partner_seat = getattr(game, "is_partner_seat", None)
    seat_roles: dict[int, str] = {}
    for pos in range(1, 6):
        if is_leaster:
            seat_roles[pos] = "leaster"
        elif game.picker == pos:
            seat_roles[pos] = "picker"
        elif (is_partner_seat(pos) if callable(is_partner_seat) else False) or getattr(
            game.players[pos - 1], "is_partner", False
        ):
            seat_roles[pos] = "partner"
        else:
            seat_roles[pos] = "defender"

    if game.picker and not is_leaster:
        final_picker_points = game.get_final_picker_points()
        final_defender_points = game.get_final_defender_points()
    else:
        final_picker_points = None
        final_defender_points = None

    return {
        "picker": game.picker,
        "partner": game.partner,
        "is_leaster": is_leaster,
        "alone_called": bool(game.alone_called),
        "is_called_under": bool(game.is_called_under),
        "called_card": game.called_card,
        "seat_roles": seat_roles,
        "final_picker_points": final_picker_points,
        "final_defender_points": final_defender_points,
    }
