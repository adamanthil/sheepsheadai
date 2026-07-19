#!/usr/bin/env python3
"""
Extended long-term PPO training for Sheepshead.
"""

import os
import time
from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from sheepshead.agent import architectures
from sheepshead.training.config import SelfPlayHyperparams
from sheepshead.training.leaster_watchdog import LeasterWatchdog
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead import (
    ACTIONS,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    Game,
)
from sheepshead.training.training_utils import (
    analyze_strategic_decisions,
    append_csv_row,
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
    get_partner_selection_mode,
    handle_trick_completion,
    process_episode_rewards,
    save_training_plot,
    set_all_seeds,
    truncate_csv_rows_past_episode,
    update_intermediate_rewards_for_action,
)

SELFPLAY_HYPERPARAMS = SelfPlayHyperparams()  # fixed LRs + entropy decay schedule (bootstrap run)

# Frozen seed for the in-training anchored CRN probe: identical across all
# ablation arms, so every run's eval curve is paired on the same deal sets.
SELFPLAY_ANCHOR_EVAL_SEED = 20260703

# Fixed external yardsticks for the anchored eval curve (never trained on):
# the conventions ScriptedAgent, the historical self-play reference at 100k
# episodes (strength-matched to this regime), and the 30M-episode league
# final (absolute yardstick).
DEFAULT_ANCHOR_100K = (
    "runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt"
)
DEFAULT_ANCHOR_PFSP = "final_pfsp_swish_ppo.pt"


def _run_anchored_eval(
    agent,
    yardsticks: dict,
    n_deals: int,
) -> dict:
    """Paired CRN edges of the training agent vs each fixed yardstick."""
    from sheepshead.training.training_utils import paired_edge

    results = {}
    for name, anchor in yardsticks.items():
        results[name] = paired_edge(
            agent, anchor, anchor, n_deals, seed=SELFPLAY_ANCHOR_EVAL_SEED, log_every=0
        )
    return results


@dataclass
class EpisodeWindows:
    """Rolling per-episode outcome trackers threaded through the reporting
    and finalization helpers below (was a handful of closed-over locals in
    train_ppo's body; the main episode loop still appends to these directly,
    it just also hands the bundle to the extracted reporting functions)."""

    picker_scores: deque
    pick_decisions: list
    pass_decisions: list
    leaster_window: deque
    alone_call_window: deque
    called_ace_window: deque
    called_under_window: deque
    called_10_window: deque
    team_point_differences: deque


@dataclass
class AnchoredEvalConfig:
    """Static config for the periodic anchored-eval probe (constant for the
    duration of a training run; only ``episode``/``eval_wall_s``/etc. change
    call to call, so those stay as plain function parameters)."""

    agent: PPOAgent
    yardsticks: dict
    anchor_eval_deals: int
    anchored_csv: str
    start_time: float


def _setup_agent(
    num_episodes,
    update_interval,
    save_interval,
    strategic_eval_interval,
    arch,
    leaster_watchdog,
    resume_model,
):
    """Print the run header, construct the PPO agent, and (if requested)
    resume from a checkpoint. Returns ``(agent, watchdog, start_episode)``."""
    print("🚀 Starting PPO training...")
    print("=" * 60)
    print("TRAINING CONFIGURATION:")
    print(f"  Episodes: {num_episodes:,}")
    print(f"  Update interval: {update_interval}")
    print(f"  Save interval: {save_interval}")
    print(f"  Strategic evaluation interval: {strategic_eval_interval}")
    print(f"  Architecture: {arch}")
    print(f"  Leaster watchdog: {'ON' if leaster_watchdog else 'off'}")
    print("=" * 60)
    watchdog = LeasterWatchdog() if leaster_watchdog else None

    # Create agent with optimized hyperparameters
    agent = PPOAgent(
        len(ACTIONS),
        lr_actor=SELFPLAY_HYPERPARAMS.lr_actor,
        lr_critic=SELFPLAY_HYPERPARAMS.lr_critic,
        arch=arch,
    )
    n_enc = sum(p.numel() for p in agent.encoder.parameters())
    n_act = sum(p.numel() for p in agent.actor.parameters())
    n_cri = sum(p.numel() for p in agent.critic.parameters())
    print(
        f"  Parameters: encoder {n_enc:,} + actor {n_act:,} + critic {n_cri:,} "
        f"= {n_enc + n_act + n_cri:,}"
    )

    # Resume from specified model or try to load best existing
    start_episode = 0
    if resume_model:
        try:
            agent.load(resume_model, load_optimizers=True)
            print(f"✅ Loaded {resume_model} for continuation")
            # Try to extract episode number from filename
            if "checkpoint_" in resume_model:
                start_episode = int(resume_model.split("_")[-1].split(".")[0])
                print(f"📍 Resuming from episode {start_episode:,}")
        except Exception as e:
            print(f"❌ Could not load {resume_model}: {e}")
    else:
        print("🆕 Starting fresh training")

    return agent, watchdog, start_episode


def _setup_anchored_eval_yardsticks(anchor_eval_interval, anchor_100k, anchor_pfsp):
    """Build the fixed-yardstick dict for the anchored CRN probe (empty/
    disabled when ``anchor_eval_interval`` is falsy)."""
    yardsticks = {}
    if anchor_eval_interval:
        from sheepshead.scripted_agent import ScriptedAgent

        yardsticks["scripted"] = ScriptedAgent()
        for name, path in (("selfplay100k", anchor_100k), ("final_pfsp", anchor_pfsp)):
            if path and os.path.exists(path):
                yardsticks[name] = load_agent(path)
            else:
                print(f"⚠️  Anchored-eval yardstick missing, skipping: {path}")
    return yardsticks


def _log_anchored_eval(
    cfg: AnchoredEvalConfig,
    episode,
    eval_wall_s,
    updates_done,
    transitions_done,
):
    """Run the anchored CRN probe, append a row to ``anchored_eval.csv``, and
    print the per-yardstick edges. Returns the updated ``eval_wall_s``."""
    t_eval = time.time()
    print(f"⚓ Anchored eval... (Episode {episode:,})")
    results = _run_anchored_eval(cfg.agent, cfg.yardsticks, cfg.anchor_eval_deals)
    eval_wall_s += time.time() - t_eval
    train_wall_s = (time.time() - cfg.start_time) - eval_wall_s
    row = {
        "episode": episode,
        "train_wall_s": round(train_wall_s, 1),
        "eval_wall_s": round(eval_wall_s, 1),
        "updates_done": updates_done,
        "transitions_done": transitions_done,
    }
    for name in ("scripted", "selfplay100k", "final_pfsp"):
        r = results.get(name)
        row[f"edge_{name}"] = round(r["edge"], 4) if r else ""
        row[f"se_{name}"] = round(r["se"], 4) if r else ""
    append_csv_row(cfg.anchored_csv, list(row.keys()), row)
    for name, r in results.items():
        print(f"   edge vs {name}: {r['edge']:+.3f} ± {r['se']:.3f}")
    return eval_wall_s


def _run_strategic_eval(agent, episode, training_data):
    """Sample strategic decisions, log the metrics, and record them into
    ``training_data`` (mutated in place)."""
    print(f"🧠 Analyzing strategic decisions... (Episode {episode:,})")
    strategic_metrics = analyze_strategic_decisions(agent, num_samples=200)

    # Store strategic metrics
    training_data["strategic_episodes"].append(episode)
    training_data["pick_hand_correlation"].append(
        strategic_metrics["pick_hand_correlation"]
    )
    training_data["picker_trump_rate"].append(strategic_metrics["picker_trump_rate"])
    training_data["defender_trump_rate"].append(
        strategic_metrics["defender_trump_rate"]
    )
    training_data["bury_quality_rate"].append(strategic_metrics["bury_quality_rate"])

    print(
        f"   Pick-Hand Correlation: {strategic_metrics['pick_hand_correlation']:.3f}"
    )
    print(f"   Picker Trump Rate: {strategic_metrics['picker_trump_rate']:.1f}%")
    print(f"   Defender Trump Rate: {strategic_metrics['defender_trump_rate']:.1f}%")
    print(f"   Bury Quality Rate: {strategic_metrics['bury_quality_rate']:.1f}%")


def _report_progress(
    episode,
    num_episodes,
    agent,
    checkpoint_dir,
    arch,
    training_data,
    windows: EpisodeWindows,
    start_time,
    best_team_difference,
):
    """Every-1000-episode progress report: rolling-window rates, training
    plot data collection, and the best-team-point-difference checkpoint
    save. Returns the (possibly updated) ``best_team_difference``."""
    current_avg_picker_score = (
        np.mean(windows.picker_scores) if windows.picker_scores else 0
    )
    # Compute pick-rate over all individual decisions (weighting games by the number of decisions they contributed)
    total_called_picks = sum(windows.pick_decisions[PARTNER_BY_CALLED_ACE])
    total_called_passes = sum(windows.pass_decisions[PARTNER_BY_CALLED_ACE])
    total_jd_picks = sum(windows.pick_decisions[PARTNER_BY_JD])
    total_jd_passes = sum(windows.pass_decisions[PARTNER_BY_JD])
    current_called_pick_rate = (
        (100 * total_called_picks / (total_called_picks + total_called_passes))
        if (total_called_picks + total_called_passes) > 0
        else 0
    )
    current_jd_pick_rate = (
        (100 * total_jd_picks / (total_jd_picks + total_jd_passes))
        if (total_jd_picks + total_jd_passes) > 0
        else 0
    )
    current_team_diff = (
        np.mean(windows.team_point_differences) if windows.team_point_differences else 0
    )
    # --- Rolling-window rates ---
    current_leaster_rate = (
        (sum(windows.leaster_window) / len(windows.leaster_window)) * 100
        if windows.leaster_window
        else 0
    )
    current_alone_rate = (
        (sum(windows.alone_call_window) / len(windows.alone_call_window)) * 100
        if windows.alone_call_window
        else 0
    )

    ca_denominator = sum(windows.called_ace_window) or 1  # avoid divide-by-zero
    current_called_under_rate = (
        sum(windows.called_under_window) / ca_denominator
    ) * 100
    current_called_10s_rate = (
        sum(windows.called_10_window) / ca_denominator
    ) * 100
    elapsed = time.time() - start_time

    # Collect data for plotting
    training_data["episodes"].append(episode)
    training_data["picker_avg"].append(current_avg_picker_score)
    training_data["called_pick_rate"].append(current_called_pick_rate)
    training_data["jd_pick_rate"].append(current_jd_pick_rate)
    training_data["learning_rate"].append(
        agent.actor_optimizer.param_groups[0]["lr"]
    )
    training_data["time_elapsed"].append(elapsed)
    training_data["team_point_diff"].append(current_team_diff)
    training_data["alone_rate"].append(current_alone_rate)
    training_data["leaster_rate"].append(current_leaster_rate)

    # Strategic metrics are collected separately during strategic evaluation intervals
    # Don't try to collect them here as they're not always available

    # Calculate training speed
    games_per_min = episode / (elapsed / 60) if elapsed > 0 else 0

    print(
        f"📊 Episode {episode:,}/{num_episodes:,} ({episode / num_episodes * 100:.1f}%)"
    )
    print("   " + "-" * 40)
    print(f"   Picker avg: {current_avg_picker_score:+.3f}")
    print(f"   Team point diff: {current_team_diff:+.1f}")
    print(f"   Called Ace Pick rate: {current_called_pick_rate:.1f}%")
    print(f"   JD Pick rate: {current_jd_pick_rate:.1f}%")
    print("   " + "-" * 20)
    print(f"   Leaster Rate: {current_leaster_rate:.2f}%")
    print(f"   Alone Call Rate: {current_alone_rate:.2f}%")
    print(f"   Called Under Rate: {current_called_under_rate:.2f}%")
    print(f"   Called 10s Rate: {current_called_10s_rate:.2f}%")
    print("   " + "-" * 40)
    print(f"   Training speed: {games_per_min:.1f} games/min")
    print(f"   Time elapsed: {elapsed / 60:.1f} min")
    print("   " + "-" * 40)

    # Save best model based on team point difference (lower is better)
    # We want the absolute value to be as small as possible
    if current_team_diff < best_team_difference:
        best_team_difference = current_team_diff
        agent.save(os.path.join(checkpoint_dir, f"best_{arch}.pt"))
        print(
            f"   🏆 New best team point difference: {best_team_difference:.1f}! Model saved."
        )

    return best_team_difference


def _save_checkpoint(
    episode,
    agent,
    checkpoint_dir,
    arch,
    save_interval,
    training_data,
    num_episodes,
    last_checkpoint_time,
):
    """Periodic checkpoint + training-plot save. Returns the updated
    ``last_checkpoint_time``."""
    checkpoint_path = f"{checkpoint_dir}/{arch}_checkpoint_{episode}.pt"
    agent.save(checkpoint_path)

    # Save enhanced training plot
    if len(training_data["episodes"]) > 10:
        plot_path = f"{checkpoint_dir}/training_progress_{episode}.png"
        save_training_plot(training_data, plot_path)

    # Calculate time since last checkpoint
    checkpoint_time = time.time()
    time_since_last = checkpoint_time - last_checkpoint_time
    last_checkpoint_time = checkpoint_time

    print(f"💾 Checkpoint saved at episode {episode:,}")
    print(
        f"   Time for last {save_interval:,} episodes: {time_since_last / 60:.1f} min"
    )
    remaining_episodes = num_episodes - episode
    if remaining_episodes > 0:
        estimated_time = (
            remaining_episodes * (time_since_last / save_interval) / 60
        )
        print(f"   Estimated time remaining: {estimated_time:.1f} min")

    return last_checkpoint_time


def _finalize_training(
    agent,
    checkpoint_dir,
    arch,
    training_data,
    start_time,
    start_episode,
    num_episodes,
    windows: EpisodeWindows,
    best_team_difference,
):
    """Discard leftover buffered events, save the final checkpoint/plot, and
    print the end-of-run summary."""
    # Final save. Leftover buffered transitions (< update_interval) are
    # intentionally discarded: a flush update would train on a smaller
    # sample than every other update (and at update()'s default epochs),
    # so the saved weights would not be the product of the specified
    # hyperparameters. final_<arch>.pt therefore equals the last
    # threshold update's weights (== the last checkpoint when episodes
    # is a multiple of save_interval).
    if agent.events:
        n_leftover = len(agent.events)
        agent.events.clear()
        print(f"   Discarding {n_leftover} leftover buffered events (no flush update)")

    agent.save(os.path.join(checkpoint_dir, f"final_{arch}.pt"))

    # Save final enhanced training plot
    if len(training_data["episodes"]) > 0:
        save_training_plot(
            training_data,
            os.path.join(checkpoint_dir, f"final_{arch}_training.png"),
        )

    total_time = time.time() - start_time
    print("\n🎉 Training completed!")
    print(
        f"   Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.1f} hours)"
    )
    print(
        f"   Final picker average: {np.mean(windows.picker_scores) if windows.picker_scores else 0:.3f}"
    )
    print(
        f"   Final team point difference: {np.mean(windows.team_point_differences) if windows.team_point_differences else 0:.1f}"
    )
    print(f"   Best team point difference: {best_team_difference:.1f}")
    total_called_picks = sum(windows.pick_decisions[PARTNER_BY_CALLED_ACE])
    total_called_passes = sum(windows.pass_decisions[PARTNER_BY_CALLED_ACE])
    total_jd_picks = sum(windows.pick_decisions[PARTNER_BY_JD])
    total_jd_passes = sum(windows.pass_decisions[PARTNER_BY_JD])
    final_called_pick_rate = (
        (100 * total_called_picks / (total_called_picks + total_called_passes))
        if (total_called_picks + total_called_passes) > 0
        else 0
    )
    final_jd_pick_rate = (
        (100 * total_jd_picks / (total_jd_picks + total_jd_passes))
        if (total_jd_picks + total_jd_passes) > 0
        else 0
    )
    print(f"   Final called Ace Pick rate: {final_called_pick_rate:.1f}%")
    print(f"   Final JD Pick rate: {final_jd_pick_rate:.1f}%")
    print(
        f"   Training speed: {(num_episodes - start_episode) / (total_time / 60):.1f} episodes/min"
    )


def train_ppo(
    num_episodes=300000,
    update_interval=2048,
    save_interval=5000,
    strategic_eval_interval=10000,
    resume_model=None,
    run_name="selfplay_ppo",
    arch="full",
    anchor_eval_interval=5000,
    anchor_eval_deals=300,
    anchor_100k=DEFAULT_ANCHOR_100K,
    anchor_pfsp=DEFAULT_ANCHOR_PFSP,
    seed=None,
    leaster_watchdog=False,
):
    """
    PPO training with strategic evaluation metrics.

    All artifacts (checkpoints, best/final model, plots) are written under
    runs/<run_name>/ so nothing collides with committed/frozen files at the
    repo root.
    """
    agent, watchdog, start_episode = _setup_agent(
        num_episodes,
        update_interval,
        save_interval,
        strategic_eval_interval,
        arch,
        leaster_watchdog,
        resume_model,
    )

    picker_scores = deque(maxlen=3000)
    pick_decisions = [deque(maxlen=3000), deque(maxlen=3000)]
    pass_decisions = [deque(maxlen=3000), deque(maxlen=3000)]

    leaster_window = deque(maxlen=3000)  # 1 ⇒ leaster, 0 ⇒ regular game
    alone_call_window = deque(maxlen=3000)  # 1 ⇒ ALONE called (non-leaster games)
    called_ace_window = deque(maxlen=3000)  # 1 ⇒ partner mode = Called-Ace, else 0
    called_under_window = deque(maxlen=3000)  # 1 ⇒ called-under occurred that game
    called_10_window = deque(maxlen=3000)  # 1 ⇒ called-10s occurred that game
    team_point_differences = deque(maxlen=3000)
    best_team_difference = float("inf")  # Lower is better (smaller point difference)

    training_data = {
        "episodes": [],
        "recent_avg": [],
        "overall_avg": [],
        "picker_avg": [],
        "called_pick_rate": [],
        "jd_pick_rate": [],
        "alone_rate": [],
        "leaster_rate": [],
        "learning_rate": [],
        "time_elapsed": [],
        "pick_hand_correlation": [],
        "picker_trump_rate": [],
        "defender_trump_rate": [],
        "bury_quality_rate": [],
        "team_point_diff": [],
        "strategic_episodes": [],
    }

    # Running picker baseline for reward shaping

    # All artifacts live under the run dir (runs/<run_name>/).
    checkpoint_dir = os.path.join("runs", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Anchored CRN eval: fixed external yardsticks, paired deals across all
    # probes (and across runs) via SELFPLAY_ANCHOR_EVAL_SEED. Eval wall-clock is
    # tracked separately so throughput comparisons exclude probe time.
    anchored_csv = os.path.join(checkpoint_dir, "anchored_eval.csv")
    # Crash-resume dedupe: drop rows a crashed run wrote past the resume
    # episode, or the replayed episodes would duplicate them.
    _n = truncate_csv_rows_past_episode(anchored_csv, start_episode)
    if _n:
        print(
            f"🧹 Trimmed {_n} stale rows past episode {start_episode:,} "
            "from anchored_eval.csv"
        )
    yardsticks = _setup_anchored_eval_yardsticks(
        anchor_eval_interval, anchor_100k, anchor_pfsp
    )
    eval_wall_s = 0.0
    updates_done = 0
    transitions_done = 0

    start_time = time.time()
    game_count = 0
    last_checkpoint_time = start_time

    print(f"\n🎮 Beginning training... (target: {num_episodes:,} episodes)")
    print("-" * 60)

    transitions_since_update = 0
    for episode in range(start_episode + 1, num_episodes + 1):
        partner_mode = get_partner_selection_mode(episode)
        # Deterministic deals when a seed is given: makes ablation runs with
        # the same --seed fully reproducible (deal distribution unchanged).
        game_seed = None if seed is None else seed * 1_000_003 + episode
        game = Game(partner_selection_mode=partner_mode, seed=game_seed)
        # Reset recurrent hidden states in the actor at the start of each game
        agent.reset_recurrent_state()
        episode_scores = []
        episode_picks = 0
        episode_passes = 0

        # Store all transitions for this episode independently for each player
        episode_transitions = {player.position: [] for player in game.players}

        # Track PLAY transitions for the current trick
        current_trick_transitions = []

        # Play full game with self-play
        while not game.is_done():
            for player in game.players:
                valid_actions = player.get_valid_action_ids()

                while valid_actions:
                    state = player.get_state_dict()
                    action, log_prob, value = agent.act(
                        state, valid_actions, player.position
                    )

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

                    action_name = ACTIONS[action - 1]

                    # Track pick/pass decisions for statistics
                    if action_name == "PICK":
                        episode_picks += 1
                    elif action_name == "PASS":
                        episode_passes += 1

                    episode_transitions[player.position].append(transition)

                    # Apply shared intermediate rewards and track trick transitions
                    update_intermediate_rewards_for_action(
                        game,
                        player,
                        action,
                        transition,
                        current_trick_transitions,
                    )

                    player.act(action)

                    # Trick resolution and observation frames
                    trick_completed = handle_trick_completion(
                        game, current_trick_transitions
                    )
                    if trick_completed and not game.is_done():
                        # ------------------------------------------------
                        # Add post-trick observation frames for all seats
                        # (stored for training-time recurrent unroll)
                        # Also update the online recurrent hidden state
                        # ------------------------------------------------
                        for seat in game.players:
                            # Update online recurrent state
                            agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
                            # Store for training-time unroll
                            episode_transitions[seat.position].append(
                                {
                                    "kind": "observation",
                                    "player": seat,
                                    "state": seat.get_last_trick_state_dict(),
                                }
                            )

                    valid_actions = player.get_valid_action_ids()

        final_scores = [player.get_score() for player in game.players]
        episode_scores = final_scores[:]

        # ---------------------------------------------
        # Compute rewards per player and store via episode API
        # ---------------------------------------------
        # Build per-player action lists
        actions_by_player = {
            pos: [t for t in episode_transitions[pos] if t["kind"] == "action"]
            for pos in episode_transitions
        }

        # Compute rewards per player
        reward_maps_by_player = {}
        for pos, acts in actions_by_player.items():
            reward_map = {}
            for reward_data in process_episode_rewards(
                acts, final_scores, game.is_leaster
            ):
                tr = reward_data["transition"]
                reward_map[id(tr)] = reward_data["reward"]
            reward_maps_by_player[pos] = reward_map

        # Build annotated event streams and ingest per player
        for pos, seq in episode_transitions.items():
            events = []
            rmap = reward_maps_by_player.get(pos, {})
            for ev in seq:
                if ev.get("kind") == "observation":
                    events.append(
                        {
                            "kind": "observation",
                            "state": ev["state"],
                            "player_id": pos,
                        }
                    )
                else:
                    reward = rmap.get(id(ev), 0.0)
                    events.append(
                        {
                            "kind": "action",
                            "state": ev["state"],
                            "action": ev["action"],
                            "log_prob": ev["log_prob"],
                            "value": ev["value"],
                            "valid_actions": ev["valid_actions"],
                            "reward": reward,
                            "player_id": pos,
                            "win_label": 1.0 if episode_scores[pos - 1] > 0 else 0.0,
                            "final_return_label": float(episode_scores[pos - 1]),
                            "secret_partner_label": ev.get("secret_partner_label", 0.0),
                            "points_label": ev.get("points_label", None),
                            "seen_trump_mask_label": ev.get(
                                "seen_trump_mask_label", None
                            ),
                            "unseen_trump_higher_than_hand_label": ev.get(
                                "unseen_trump_higher_than_hand_label", None
                            ),
                        }
                    )
            agent.store_episode_events(events)
            n_new = sum(1 for e in events if e["kind"] == "action")
            transitions_since_update += n_new
            transitions_done += n_new

        # Track statistics
        picker_score = episode_scores[game.picker - 1] if game.picker else 0

        # Calculate team point difference (picker team points - defender team points)
        if game.picker and not game.is_leaster:
            picker_team_points = game.get_final_picker_points()
            defender_team_points = game.get_final_defender_points()
            team_point_diff = abs(picker_team_points - defender_team_points)
        else:
            team_point_diff = 0  # No team difference in leaster games

        # --------------------------------------------------
        # Append episode outcome to rolling windows
        # --------------------------------------------------
        is_leaster_ep = 1 if game.is_leaster else 0
        leaster_window.append(is_leaster_ep)

        is_called_ace_ep = 1 if partner_mode == PARTNER_BY_CALLED_ACE else 0
        called_ace_window.append(is_called_ace_ep)

        # Only meaningful for Called-Ace, non-leaster games
        if is_called_ace_ep and not is_leaster_ep:
            called_under_window.append(1 if game.is_called_under else 0)
            called_10_window.append(
                1 if (game.called_card and game.called_card.startswith("10")) else 0
            )
        elif is_called_ace_ep:
            called_under_window.append(0)
            called_10_window.append(0)

        picker_scores.append(picker_score)
        # ALONE tracking for games with a picker (exclude leaster)
        if not game.is_leaster:
            alone_call_window.append(1 if game.alone_called else 0)
        pick_decisions[get_partner_selection_mode(episode)].append(episode_picks)
        pass_decisions[get_partner_selection_mode(episode)].append(episode_passes)
        team_point_differences.append(team_point_diff)
        game_count += 1

        # Update model periodically by transition count (action transitions only)
        if transitions_since_update >= update_interval:
            print(
                f"🔄 Updating model after {transitions_since_update} transitions... (Episode {episode:,})"
            )

            # Separate entropy decay schedules (config.SelfPlayHyperparams).
            decay_fraction = min(episode / num_episodes, 1.0)
            agent.entropy_coeff_play = (
                SELFPLAY_HYPERPARAMS.entropy_play_start
                + (SELFPLAY_HYPERPARAMS.entropy_play_end - SELFPLAY_HYPERPARAMS.entropy_play_start) * decay_fraction
            )
            agent.entropy_coeff_pick = (
                SELFPLAY_HYPERPARAMS.entropy_pick_start
                + (SELFPLAY_HYPERPARAMS.entropy_pick_end - SELFPLAY_HYPERPARAMS.entropy_pick_start) * decay_fraction
            )
            agent.entropy_coeff_partner = (
                SELFPLAY_HYPERPARAMS.entropy_partner_start
                + (SELFPLAY_HYPERPARAMS.entropy_partner_end - SELFPLAY_HYPERPARAMS.entropy_partner_start) * decay_fraction
            )
            agent.entropy_coeff_bury = (
                SELFPLAY_HYPERPARAMS.entropy_bury_start
                + (SELFPLAY_HYPERPARAMS.entropy_bury_end - SELFPLAY_HYPERPARAMS.entropy_bury_start) * decay_fraction
            )

            if watchdog is not None:
                watchdog.tick(agent, leaster_window)

            update_stats = agent.update(epochs=4, batch_size=256)

            # Log advantage and value target statistics
            if update_stats:
                adv_stats = update_stats["advantage_stats"]
                val_stats = update_stats["value_target_stats"]
                num_transitions = update_stats["num_transitions"]
                approx_kl = update_stats.get("approx_kl", None)
                early_stop = update_stats.get("early_stop", False)

                print(f"   Transitions: {num_transitions}")
                print(
                    f"   Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]"
                )
                print(
                    f"   Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]"
                )
                if approx_kl is not None:
                    print(f"   PPO KL: {approx_kl:.4f}  Early stop: {early_stop}")
                if "timing" in update_stats:
                    t = update_stats["timing"]
                    print(
                        f"   Timing - build: {t['build_s']:.3f}s, forward: {t['forward_s']:.3f}s, "
                        f"backward: {t['backward_s']:.3f}s, step: {t['step_s']:.3f}s, total: {t['total_update_s']:.3f}s, "
                        f"opt_steps: {t['optimizer_steps']}"
                    )
                head_entropy = update_stats.get("head_entropy")
                if head_entropy:
                    print(
                        f"   Entropy - pick: {head_entropy.get('pick', 0.0):.3f}, "
                        f"partner: {head_entropy.get('partner', 0.0):.3f}, "
                        f"bury: {head_entropy.get('bury', 0.0):.3f}, "
                        f"play: {head_entropy.get('play', 0.0):.3f}"
                    )

            game_count = 0
            transitions_since_update = 0
            updates_done += 1

        # Anchored CRN eval at intervals (fixed yardsticks, paired deals)
        if anchor_eval_interval and yardsticks and episode % anchor_eval_interval == 0:
            anchor_cfg = AnchoredEvalConfig(
                agent=agent,
                yardsticks=yardsticks,
                anchor_eval_deals=anchor_eval_deals,
                anchored_csv=anchored_csv,
                start_time=start_time,
            )
            eval_wall_s = _log_anchored_eval(
                anchor_cfg, episode, eval_wall_s, updates_done, transitions_done
            )

            # Strategic evaluation at intervals
        if episode % strategic_eval_interval == 0:
            _run_strategic_eval(agent, episode, training_data)

        windows = EpisodeWindows(
            picker_scores=picker_scores,
            pick_decisions=pick_decisions,
            pass_decisions=pass_decisions,
            leaster_window=leaster_window,
            alone_call_window=alone_call_window,
            called_ace_window=called_ace_window,
            called_under_window=called_under_window,
            called_10_window=called_10_window,
            team_point_differences=team_point_differences,
        )

        # Progress reporting and data collection
        if episode % 1000 == 0:
            best_team_difference = _report_progress(
                episode,
                num_episodes,
                agent,
                checkpoint_dir,
                arch,
                training_data,
                windows,
                start_time,
                best_team_difference,
            )

        # Save regular checkpoints
        if episode % save_interval == 0:
            last_checkpoint_time = _save_checkpoint(
                episode,
                agent,
                checkpoint_dir,
                arch,
                save_interval,
                training_data,
                num_episodes,
                last_checkpoint_time,
            )

    windows = EpisodeWindows(
        picker_scores=picker_scores,
        pick_decisions=pick_decisions,
        pass_decisions=pass_decisions,
        leaster_window=leaster_window,
        alone_call_window=alone_call_window,
        called_ace_window=called_ace_window,
        called_under_window=called_under_window,
        called_10_window=called_10_window,
        team_point_differences=team_point_differences,
    )
    _finalize_training(
        agent,
        checkpoint_dir,
        arch,
        training_data,
        start_time,
        start_episode,
        num_episodes,
        windows,
        best_team_difference,
    )


def main():
    parser = ArgumentParser(description="PPO training for Sheepshead")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100000,
        help="Number of training episodes (default: 100,000)",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=4096,
        help="Number of games between model updates",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Number of episodes between checkpoints",
    )
    parser.add_argument(
        "--strategic-eval-interval",
        type=int,
        default=10000,
        help="Number of episodes between strategic evaluations",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Model file to resume from"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="selfplay_ppo",
        help="Run name; all artifacts go under runs/<run-name>/ (default: selfplay_ppo)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="full",
        choices=architectures.available_architectures(),
        help="Network architecture variant (see the architectures package)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for python/numpy/torch (default: 42)",
    )
    parser.add_argument(
        "--anchor-eval-interval",
        type=int,
        default=5000,
        help="Episodes between anchored CRN evals (0 disables; default: 5000)",
    )
    parser.add_argument(
        "--anchor-eval-deals",
        type=int,
        default=300,
        help="Paired deals per yardstick per anchored eval (default: 300)",
    )
    parser.add_argument(
        "--anchor-100k",
        type=str,
        default=DEFAULT_ANCHOR_100K,
        help="Strength-matched anchor checkpoint for the eval curve",
    )
    parser.add_argument(
        "--anchor-pfsp",
        type=str,
        default=DEFAULT_ANCHOR_PFSP,
        help="Absolute-yardstick anchor checkpoint for the eval curve",
    )
    parser.add_argument(
        "--leaster-watchdog",
        action="store_true",
        help="Selective pick-entropy kick against the always-PASS collapse "
        "(engages at 90%% rolling leaster rate, releases below 30%%)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_all_seeds(args.seed)
    print(f"🎲 Seed: {args.seed}")

    # Ensure matplotlib uses a non-interactive backend
    plt.switch_backend("Agg")

    train_ppo(
        args.episodes,
        args.update_interval,
        args.save_interval,
        args.strategic_eval_interval,
        args.resume,
        args.run_name,
        arch=args.arch,
        anchor_eval_interval=args.anchor_eval_interval,
        anchor_eval_deals=args.anchor_eval_deals,
        anchor_100k=args.anchor_100k,
        anchor_pfsp=args.anchor_pfsp,
        seed=args.seed,
        leaster_watchdog=args.leaster_watchdog,
    )


if __name__ == "__main__":
    main()
