#!/usr/bin/env python3
"""Re-save a checkpoint under a registry architecture that is a strict
superset of its own (CE_Teacher_Design §20.9): every tensor the source
carries is copied, the new architecture's extra actor parameters keep
their construction-time init, and the result is an ordinary checkpoint
for ``load_agent`` / the game server. Optimizer state is NOT carried
(the parameter groups differ); the consumers of a migrated checkpoint
are the supervised phases, which set their own learning rates.

The only migration this tool accepts is one where the extra parameters
are inert at init — it verifies that the migrated agent's action
probabilities equal the source agent's on a sample of real game states
to float precision, and refuses otherwise.

Usage:
  uv run python -m sheepshead.analysis.migrate_arch_checkpoint \\
      --src runs/.../checkpoint_8000000.pt --arch perceiver-shared-v2-bp \\
      --out runs/policy_iteration_202609/theta_k_bp.pt
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game
from sheepshead.agent import ppo as ppo_module
from sheepshead.agent.ppo import PPOAgent, load_agent


def migrate(src: str, arch: str) -> tuple[PPOAgent, list[str]]:
    """Build an ``arch`` agent carrying every network tensor of ``src``.
    Returns the agent and the actor keys that kept their fresh init."""
    checkpoint = torch.load(src, map_location=ppo_module.device)
    # Same construction as load_agent: critic mode and oracle aux flag from
    # the source, so an oracle-mode checkpoint keeps its privileged critic.
    agent = PPOAgent(
        len(ACTIONS),
        critic_mode=checkpoint.get("critic_mode", "limited"),
        arch=arch,
        oracle_aux_heads=bool(checkpoint.get("oracle_aux_heads", False)),
    )
    agent.gamma = float(checkpoint.get("gamma", agent.gamma))
    agent.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    missing, unexpected = agent.actor.load_state_dict(
        checkpoint["actor_state_dict"], strict=False
    )
    if unexpected:
        raise SystemExit(f"source actor has tensors {arch} lacks: {unexpected}")
    c_missing, c_unexpected = agent.critic.load_state_dict(
        checkpoint["critic_state_dict"], strict=False
    )
    if c_missing or c_unexpected:
        raise SystemExit(
            f"critic mismatch: missing={c_missing} unexpected={c_unexpected}"
        )
    if (agent.oracle_critic is None) != ("oracle_state_dict" not in checkpoint):
        raise SystemExit(
            "oracle critic presence differs between source and target agent"
        )
    if agent.oracle_critic is not None and "oracle_state_dict" in checkpoint:
        o_missing, o_unexpected = agent.oracle_critic.load_state_dict(
            checkpoint["oracle_state_dict"], strict=False
        )
        bad = [k for k in (*o_missing, *o_unexpected) if not k.startswith("team_")]
        if bad:
            raise SystemExit(f"oracle mismatch: {bad}")
    agent.optimizer_steps_total = int(checkpoint.get("optimizer_steps_total", 0))
    return agent, list(missing)


def max_prob_divergence(
    a: PPOAgent, b: PPOAgent, games: int = 6, seed: int = 0
) -> float:
    """Max |p_a - p_b| over every decision of ``games`` self-play games
    driven by ``a`` (both agents observe the same stream)."""
    worst = 0.0
    for g in range(games):
        mode = PARTNER_BY_CALLED_ACE if g % 2 == 0 else PARTNER_BY_JD
        game = Game(partner_selection_mode=mode, seed=seed + g)
        a.reset_recurrent_state()
        b.reset_recurrent_state()
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    state = player.get_state_dict()
                    action, _, _ = a.act(
                        state, valid, player.position, deterministic=True
                    )
                    pa = np.asarray(a.last_action_probs, dtype=np.float64)
                    b.act(state, valid, player.position, deterministic=True)
                    pb = np.asarray(b.last_action_probs, dtype=np.float64)
                    worst = max(worst, float(np.abs(pa - pb).max()))
                    player.act(action)
                    if game.was_trick_just_completed and not game.is_done():
                        for seat in game.players:
                            obs = seat.get_last_trick_state_dict()
                            a.observe(obs, player_id=seat.position)
                            b.observe(obs, player_id=seat.position)
                    valid = player.get_valid_action_ids()
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--arch", required=True, help="target registry architecture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args(argv)

    agent, fresh = migrate(args.src, args.arch)
    source = load_agent(args.src, load_optimizers=False)
    source.stash_action_probs = True
    agent.stash_action_probs = True
    div = max_prob_divergence(source, agent)
    if div > args.tol:
        raise SystemExit(
            f"migrated agent diverges from the source (max |dp| {div:g} > {args.tol}); "
            "the extra parameters are not inert at init — refusing to write"
        )
    agent.save(args.out)
    print(
        f"wrote {args.out} (arch {args.arch}); fresh actor tensors: {fresh}; "
        f"max action-prob divergence vs source {div:g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
