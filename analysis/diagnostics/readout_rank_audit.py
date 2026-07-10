"""Effective-rank audit of the attention->features linear map.

Question (operator, 2026-07-09): is perceiver-shared-v2's single
Linear(1024->256) readout projection an information bottleneck relative
to full's pooled path? Both paths are purely linear before the features
LayerNorm, so they are directly comparable as matrices:

  v2:   W_v2 (256 x 1024)  = readout_proj Linear weight
  full: W_full (256 x 1088) = feature_proj Linear  @  blockdiag(
            pool_hand.proj (64x256), pool_trick.proj (64x256),
            pool_blind.proj (32x256), pool_bury.proj (32x256), I_64)
        (input layout = [hand attn 256; trick 256; blind 256; bury 256;
         raw context 64], matching the encoder's concat order)

Reported per checkpoint:
  weight spectrum   entropy effective rank exp(H(sigma^2)), ranks
                    capturing 90%/99% of squared-singular-value energy
  feature covariance (post-LN features over ~15 CRN games): participation
                    ratio (sum(lam))^2/sum(lam^2), ranks for 90%/99%
                    variance — how much of the 256-d trunk is actually
                    used on-distribution.
"""

import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/Volumes/Nargothrond/dev/sheepsheadai")

from ppo import load_agent
from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game

N_GAMES = 15
BASE_SEED = 20260710
DEVICE = torch.device("cpu")


def spectrum_stats(w: torch.Tensor) -> dict:
    s = torch.linalg.svdvals(w.detach()).numpy()
    e = s**2
    p = e / e.sum()
    eff = float(np.exp(-(p * np.log(p.clip(1e-12))).sum()))
    cum = np.cumsum(e) / e.sum()
    return {
        "eff_rank": eff,
        "r90": int(np.searchsorted(cum, 0.90) + 1),
        "r99": int(np.searchsorted(cum, 0.99) + 1),
        "sv_max": float(s[0]),
        "sv_min": float(s[-1]),
        "shape": tuple(w.shape),
    }


def composite_full_map(enc) -> torch.Tensor:
    wf = enc.feature_proj[0].weight.detach()  # (256, 256)
    blocks = [
        enc.pool_hand.proj.weight.detach(),  # (64, 256)
        enc.pool_trick.proj.weight.detach(),  # (64, 256)
        enc.pool_blind.proj.weight.detach(),  # (32, 256)
        enc.pool_bury.proj.weight.detach(),  # (32, 256)
    ]
    d_in = sum(b.shape[1] for b in blocks) + 64  # 1088
    d_mid = sum(b.shape[0] for b in blocks) + 64  # 256
    B = torch.zeros(d_mid, d_in)
    r = c = 0
    for b in blocks:
        B[r : r + b.shape[0], c : c + b.shape[1]] = b
        r += b.shape[0]
        c += b.shape[1]
    B[r : r + 64, c : c + 64] = torch.eye(64)  # raw context passthrough
    return wf @ B  # (256, 1088)


def feature_cov_stats(agent) -> dict:
    feats = []
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
                        out = agent.encoder.encode_batch(
                            [s], memory_in=mem.unsqueeze(0), device=DEVICE
                        )
                        agent.set_recurrent_memory(
                            player.position, out["memory_out"][0]
                        )
                        feats.append(out["features"][0].numpy())
                    a, _, _ = agent.act(s, va, player.position)
                    player.act(a)
                    if game.was_trick_just_completed and not game.is_done():
                        for seat in game.players:
                            agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
                    va = player.get_valid_action_ids()
    X = np.array(feats)
    X = X - X.mean(axis=0, keepdims=True)
    lam = np.linalg.eigvalsh(np.cov(X.T))[::-1].clip(0)
    pr = float(lam.sum() ** 2 / (lam**2).sum())
    cum = np.cumsum(lam) / lam.sum()
    return {
        "n_states": len(X),
        "participation": pr,
        "v90": int(np.searchsorted(cum, 0.90) + 1),
        "v99": int(np.searchsorted(cum, 0.99) + 1),
    }


def main():
    targets = [
        ("full_s42", "runs/ablate_full_s42/full_checkpoint_200000.pt"),
        ("full_s2042", "runs/ablate_full_s2042/full_checkpoint_200000.pt"),
        (
            "v2_s42",
            "runs/ablate_perceiver-shared-v2_s42/perceiver-shared-v2_checkpoint_200000.pt",
        ),
        (
            "v2_s2042",
            "runs/ablate_perceiver-shared-v2_s2042/perceiver-shared-v2_checkpoint_200000.pt",
        ),
    ]
    for tag, path in targets:
        agent = load_agent(path)
        enc = agent.encoder
        if hasattr(enc, "readout_proj"):
            lin = (
                enc.readout_proj[0]
                if isinstance(enc.readout_proj, torch.nn.Sequential)
                else enc.readout_proj
            )
            w = lin.weight
        else:
            w = composite_full_map(enc)
        ws = spectrum_stats(w)
        cs = feature_cov_stats(agent)
        print(
            f"{tag:10s} W{ws['shape']}: eff_rank {ws['eff_rank']:6.1f}  "
            f"r90 {ws['r90']:3d}  r99 {ws['r99']:3d}  "
            f"sv {ws['sv_max']:.2f}/{ws['sv_min']:.3f} | "
            f"features: PR {cs['participation']:6.1f}  v90 {cs['v90']:3d}  "
            f"v99 {cs['v99']:3d}  (n={cs['n_states']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
