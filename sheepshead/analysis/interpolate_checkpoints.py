#!/usr/bin/env python3
"""WiSE-FT weight interpolation between two checkpoints of one architecture
(Wortsman et al. 2022, arXiv:2109.01903; CE_Teacher_Design §17.16).

theta_alpha = (1 - alpha) * theta_base + alpha * theta_tuned, elementwise
over every floating-point parameter and buffer of every module the
checkpoint carries (encoder / actor / critic / oracle critic); integer
buffers must be identical. Saved through ``agent.save()`` so the result is
an ordinary checkpoint for the cert battery and ``load_agent``.

The §17.16 finding this operationalizes: fine-tune damage decays faster
along the path than a coherently installed behavior shift, so walking a
hot distill back toward theta_k finds an EV-parity point that keeps most
of the installed convention.

Usage:
  uv run python -m sheepshead.analysis.interpolate_checkpoints \\
      --base runs/league_retention_pg/checkpoints/..._checkpoint_8000000.pt \\
      --tuned runs/policy_iteration_202609/iter1/distill_epoch1.pt \\
      --alpha 0.5 --out runs/policy_iteration_202609/iter1/interp_a50.pt
"""

from __future__ import annotations

import argparse
import sys

import torch

from sheepshead.agent.ppo import load_agent

MODULES = ("encoder", "actor", "critic", "oracle_critic")


def interpolate(base_path: str, tuned_path: str, alpha: float, out_path: str) -> None:
    mix = load_agent(tuned_path)
    base = load_agent(base_path)
    for name in MODULES:
        m_mix = getattr(mix, name, None)
        m_base = getattr(base, name, None)
        if m_mix is None or m_base is None:
            if (m_mix is None) != (m_base is None):
                raise SystemExit(f"module {name} present in only one checkpoint")
            continue
        mix_state = m_mix.state_dict()
        base_state = m_base.state_dict()
        if mix_state.keys() != base_state.keys():
            raise SystemExit(f"{name}: state dict keys differ")
        merged = {}
        for k, t_mix in mix_state.items():
            t_base = base_state[k]
            if t_mix.dtype.is_floating_point:
                merged[k] = torch.lerp(t_base, t_mix, alpha)
            else:
                if not torch.equal(t_mix, t_base):
                    raise SystemExit(f"{name}.{k}: non-float buffer differs")
                merged[k] = t_mix
        m_mix.load_state_dict(merged)
    mix.save(out_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="theta_k (alpha = 0)")
    ap.add_argument(
        "--tuned", required=True, help="the fine-tuned checkpoint (alpha = 1)"
    )
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    interpolate(args.base, args.tuned, args.alpha, args.out)
    print(f"alpha={args.alpha}: saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
