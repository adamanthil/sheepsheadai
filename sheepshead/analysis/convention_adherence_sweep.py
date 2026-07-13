#!/usr/bin/env python3
"""Convention adherence sweep across training checkpoints (E1).

Runs both convention probes — C1 defender trump-lead incidence
(``trump_lead_probe``) and C2 called-suit-lead adherence (``called_suit_probe``)
— over a list of checkpoints plus the ScriptedAgent anchor, on the shared
frozen CRN deal set. Emits the adherence-vs-training-compute table that decides
the empirical half of Q2 (is the convention being learned under terminal-only
reward?). See notebooks/Convention_Optimality_202607.md.

The ScriptedAgent anchor doubles as the instrument self-check: C1 rate must be
exactly 0 and C2 trick-0 adherence exactly 1.0, or the run is rejected.

Usage (from repo root):

    uv run python -m sheepshead.analysis.convention_adherence_sweep \
        --deals 1000 --out runs/convention_optimality_202607/adherence_sweep.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from sheepshead.analysis import called_suit_probe, trump_lead_probe
from sheepshead.scripted_agent import ScriptedAgent
from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD

# The reference lineage (PANEL-A members, oldest first). Stage-1 league gens
# can be appended via --ckpts as they land.
DEFAULT_CKPTS = [
    "runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt",
    "runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_1000000.pt",
    "runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_5000000.pt",
    "runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt",
    "final_pfsp_swish_ppo.pt",
]

# final_pfsp is the 30M run's terminal artifact (see
# memory: project_final_model_provenance) — no episode count in its filename.
KNOWN_EPISODES = {"final_pfsp_swish_ppo.pt": 30_000_000}


def _episodes_from_name(path: str) -> int | None:
    if path in KNOWN_EPISODES:
        return KNOWN_EPISODES[path]
    m = re.search(r"checkpoint_(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def _row(label: str, episodes: int | None, c1_called: dict, c1_jd: dict | None, c2: dict) -> dict:
    return {
        "label": label,
        "episodes": episodes,
        "c1_lead_rate_called": c1_called["lead_rate"],
        "c1_opportunities_called": c1_called["opportunities"],
        "c1_lead_rate_trump_rich_called": c1_called["lead_rate_trump_rich"],
        "c1_lead_rate_jd": c1_jd["lead_rate"] if c1_jd else None,
        "c1_opportunities_jd": c1_jd["opportunities"] if c1_jd else None,
        "c2_adherence_rate": c2["adherence_rate"],
        "c2_adherence_rate_trick0": c2["adherence_rate_trick0"],
        "c2_adherence_rate_first_opp": c2["adherence_rate_first_opp"],
        "c2_eligible": c2["eligible"],
        "c2_eligible_trick0": c2["eligible_trick0"],
        "c1_raw_called": c1_called,
        "c1_raw_jd": c1_jd,
        "c2_raw": c2,
    }


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(
        f"{'checkpoint':<28} {'episodes':>10} | {'C1 trump-lead% (n)':>20} "
        f"{'rich%':>6} | {'C2 adh%':>8} {'t0%':>7} {'1st%':>7} {'(n)':>6}"
    )
    print("-" * 100)
    for r in rows:
        ep = f"{r['episodes']:,}" if r["episodes"] is not None else "-"
        print(
            f"{r['label']:<28} {ep:>10} | "
            f"{100 * r['c1_lead_rate_called']:>13.2f} ({r['c1_opportunities_called']:>4}) "
            f"{100 * r['c1_lead_rate_trump_rich_called']:>6.2f} | "
            f"{100 * r['c2_adherence_rate']:>8.2f} "
            f"{100 * r['c2_adherence_rate_trick0']:>7.2f} "
            f"{100 * r['c2_adherence_rate_first_opp']:>7.2f} "
            f"({r['c2_eligible']:>4})"
        )
    print("=" * 100)
    print(
        "C1 = defender trump-lead rate at tricks 0-1, fail legal (convention: 0%).\n"
        "C2 = called-suit lead adherence over eligible nodes (convention: 100%)."
    )


def probe_hero(hero, deals: int, seed: int, both_modes: bool) -> tuple[dict, dict | None, dict]:
    c1_called = trump_lead_probe.probe_agent(
        hero, deals, PARTNER_BY_CALLED_ACE, seed=seed
    )
    c1_jd = (
        trump_lead_probe.probe_agent(hero, deals, PARTNER_BY_JD, seed=seed)
        if both_modes
        else None
    )
    c2 = called_suit_probe.probe_agent(hero, deals, seed=seed)
    return c1_called, c1_jd, c2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpts",
        nargs="*",
        default=DEFAULT_CKPTS,
        help="Checkpoints to sweep, oldest first (default: reference lineage).",
    )
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=trump_lead_probe.PROBE_SEED)
    ap.add_argument(
        "--no-scripted", action="store_true", help="Skip the ScriptedAgent anchor."
    )
    ap.add_argument(
        "--both-modes",
        action="store_true",
        help="Also run the C1 probe in JD mode (C2 is called-ace only).",
    )
    ap.add_argument("--out", default=None, help="Path for the JSON report.")
    args = ap.parse_args()

    rows: list[dict] = []

    if not args.no_scripted:
        t0 = time.time()
        c1_called, c1_jd, c2 = probe_hero(
            ScriptedAgent(), args.deals, args.seed, args.both_modes
        )
        # Instrument self-check (the anchor is also the harness gate).
        if c1_called["trump_leads"] != 0 or c2["adherence_rate_trick0"] != 1.0:
            raise SystemExit(
                "ScriptedAgent anchor violated its by-construction rates "
                f"(C1 leads {c1_called['trump_leads']}, "
                f"C2 t0 {c2['adherence_rate_trick0']:.3f}); instrument broken."
            )
        rows.append(_row("scripted-anchor", None, c1_called, c1_jd, c2))
        print(f"[scripted-anchor] ok ({time.time() - t0:.0f}s)", flush=True)

    for ckpt in args.ckpts:
        from sheepshead.agent.ppo import load_agent

        t0 = time.time()
        hero = load_agent(ckpt)
        c1_called, c1_jd, c2 = probe_hero(hero, args.deals, args.seed, args.both_modes)
        label = Path(ckpt).stem.replace("pfsp_swish_checkpoint_", "pfsp_").replace(
            "swish_checkpoint_", "selfplay_"
        )
        rows.append(_row(label, _episodes_from_name(ckpt), c1_called, c1_jd, c2))
        print(f"[{label}] done ({time.time() - t0:.0f}s)", flush=True)
        del hero

    _print_table(rows)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "probe_seed": args.seed,
                "deals": args.deals,
                "ckpts": args.ckpts,
                "both_modes": args.both_modes,
            },
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
