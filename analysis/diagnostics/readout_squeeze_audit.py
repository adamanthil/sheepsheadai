"""Shared-readout squeeze audit: are the 4q x 4h = 16 attention
distributions saturated (partitioning the 19 tokens with little slack) or
redundant (overlapping)? Informs the channel-count dial (v2 uses 16q).

For each checkpoint: play N seeded games with the agent, capture the
encoder readout's per-head attention maps (need_weights=True,
average_attn_weights=False), and report:

  ent_mean / ent_min   per-distribution entropy over valid tokens (nats;
                       uniform over 19 = 2.94)
  coverage             fraction of valid tokens receiving > 50% of uniform
                       mass when attention is summed over all distributions
  overlap_mean         mean pairwise cosine similarity between the 16
                       mean attention maps (1.0 = fully redundant channels,
                       ~0 = disjoint partition)
"""

import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/Volumes/Nargothrond/dev/sheepsheadai")

from ppo import load_agent
from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game

N_GAMES = 30
BASE_SEED = 20260709
DEVICE = torch.device("cpu")


def audit(ckpt: str) -> dict:
    agent = load_agent(ckpt)
    enc = agent.encoder
    ents = []
    map_sum = None  # (n_dist, 19) summed attention maps
    map_count = 0

    for g in range(N_GAMES):
        mode = PARTNER_BY_CALLED_ACE if g % 2 == 0 else PARTNER_BY_JD
        random.seed(BASE_SEED + g)
        torch.manual_seed(BASE_SEED + g)
        game = Game(partner_selection_mode=mode, seed=BASE_SEED + g)
        agent.reset_recurrent_state()
        while not game.is_done():
            for player in game.players:
                va = player.get_valid_action_ids()
                while va:
                    s = player.get_state_dict()
                    mem = agent.get_recurrent_memory(player.position, device=DEVICE)
                    with torch.no_grad():
                        out = enc.encode_batch(
                            [s], memory_in=mem.unsqueeze(0), device=DEVICE
                        )
                        agent.set_recurrent_memory(
                            player.position, out["memory_out"][0]
                        )
                        q = enc.readout_query.unsqueeze(0)
                        _, w = enc.readout_mha(
                            q,
                            out["all_tokens"],
                            out["all_tokens"],
                            key_padding_mask=~out["all_mask"],
                            need_weights=True,
                            average_attn_weights=False,
                        )
                        # w: (1, n_heads, n_queries, 19)
                        w = w[0]  # (H, Q, 19)
                        H, Q, S = w.shape
                        dists = w.reshape(H * Q, S)  # 16 distributions
                        valid = out["all_mask"][0]  # (19,)
                        p = dists[:, valid]
                        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-9)
                        ents.append(
                            (-(p * (p.clamp_min(1e-9)).log()).sum(dim=1)).numpy()
                        )
                        full_map = torch.zeros(H * Q, S)
                        full_map[:, valid] = p
                        map_sum = full_map if map_sum is None else map_sum + full_map
                        map_count += 1
                    a, _, _ = agent.act(s, va, player.position)
                    player.act(a)
                    if game.was_trick_just_completed and not game.is_done():
                        for seat in game.players:
                            agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
                    va = player.get_valid_action_ids()

    ent = np.concatenate([e[None, :] for e in ents], axis=0)  # (N, 16)
    mean_maps = (map_sum / map_count).numpy()  # (16, 19)
    # pairwise cosine overlap between the 16 mean maps
    norms = np.linalg.norm(mean_maps, axis=1, keepdims=True).clip(1e-9)
    unit = mean_maps / norms
    cos = unit @ unit.T
    iu = np.triu_indices(cos.shape[0], k=1)
    # coverage: tokens with summed mass > 0.5 * uniform share
    total_mass = mean_maps.sum(axis=0)  # (19,)
    thresh = 0.5 * total_mass.sum() / 19
    return {
        "ent_mean": float(ent.mean()),
        "ent_min": float(ent.mean(axis=0).min()),
        "ent_max": float(ent.mean(axis=0).max()),
        "coverage": float((total_mass > thresh).mean()),
        "overlap_mean": float(cos[iu].mean()),
        "overlap_max": float(cos[iu].max()),
    }


def main():
    for tag, path in [
        (
            "shared_s42@100k",
            "runs/ablate_perceiver-shared_s42/perceiver-shared_checkpoint_100000.pt",
        ),
        (
            "shared_s42@200k",
            "runs/ablate_perceiver-shared_s42/perceiver-shared_checkpoint_200000.pt",
        ),
        (
            "shared_s1042@200k",
            "runs/ablate_perceiver-shared_s1042/perceiver-shared_checkpoint_200000.pt",
        ),
        (
            "shared_s2042@200k",
            "runs/ablate_perceiver-shared_s2042/perceiver-shared_checkpoint_200000.pt",
        ),
    ]:
        r = audit(path)
        print(
            f"{tag}: ent mean {r['ent_mean']:.2f} (per-dist mean range "
            f"{r['ent_min']:.2f}-{r['ent_max']:.2f}; uniform 2.94)  "
            f"coverage {r['coverage']:.2f}  "
            f"overlap mean {r['overlap_mean']:.2f} max {r['overlap_max']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
