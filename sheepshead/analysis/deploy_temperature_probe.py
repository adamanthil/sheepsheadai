"""Deployment temperature probe (Training_Program_Redesign §4.5).

Sweeps deployment samplers for one checkpoint on the cert instrument: the
sampled hero in all five seats per CRN deal against an argmax field of
``--field`` (default: the same checkpoint, so the T = 0 arm is the exact
null and every edge is "what sampling costs or buys against the policy's
own greedy self"), both partner modes, deal-bootstrap SE
(``league_progress_eval.h2h_duplicate``). Arms share deal seeds and the
hero's per-deal uniform stream, so arms differing only in T are
monotonically coupled.

Arms: for each head group selection (``--heads play``, ``bidding`` =
pick+partner+bury, or ``all``) and each temperature, optionally crossed
with logit bands. Reports edge ± SE, the per-mode and leaster splits, the
fraction of deals the hero changed the outcome on, and the arm's expected
argmax-deviation rate per head measured offline on ``--corpus-rows``
(a distill holdout: how often the deployed policy would not play the
argmax, and how often that is a tie-equivalent card).

A fixed argmax field rewards sharpening by construction (§4.5), so this
read fixes the cost side of T; the exploitability/predictability side is
the audit and the ceiling read. Repeat with ``--field`` set to the 30M or
the v2 release to see whether the ranking of T holds across opponents.

    uv run python -m sheepshead.analysis.deploy_temperature_probe \\
        --ckpt runs/202609_recall_rc/final/release.pt \\
        --heads play --temps 0.25 0.5 0.75 1.0 \\
        --deals-per-mode 8000 --out-json runs/202609_recall_rc/final/temperature/play.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sheepshead.agent.sampling import HEAD_GROUPS, DeploySampling
from sheepshead.analysis.league_progress_eval import h2h_duplicate

GROUPS = {
    "play": ("play",),
    "pick": ("pick",),
    "partner": ("partner",),
    "bury": ("bury",),
    "bidding": ("pick", "partner", "bury"),
    "all": HEAD_GROUPS,
}


def arms(
    heads: str, temps: list[float], bands: list[float | None]
) -> list[DeploySampling]:
    out = []
    for band in bands:
        for t in temps:
            out.append(
                DeploySampling(
                    temperature={
                        h: (t if h in GROUPS[heads] else 0.0) for h in HEAD_GROUPS
                    },
                    band=band,
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="the policy being deployed")
    ap.add_argument(
        "--field", default=None, help="argmax field checkpoint (default: --ckpt)"
    )
    ap.add_argument("--heads", choices=sorted(GROUPS), default="play")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument(
        "--bands",
        type=float,
        nargs="*",
        default=None,
        help="logit bands (nats) to cross with the temperatures; omit for none",
    )
    ap.add_argument("--deals-per-mode", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)
    field = args.field or args.ckpt
    bands: list[float | None] = [None]
    bands.extend(float(b) for b in (args.bands or []))
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if out.exists():
        results = json.load(open(out)).get("arms", [])
    done = {r["label"] for r in results}
    for sampler in arms(args.heads, args.temps, bands):
        label = sampler.label()
        if label in done:
            print(f"{label}: done, skipped", flush=True)
            continue
        t0 = time.time()
        print(
            f"{label}: h2h vs argmax {field} ({args.deals_per_mode} deals/mode) ...",
            flush=True,
        )
        res = h2h_duplicate(
            args.ckpt,
            field,
            n_deals_per_mode=args.deals_per_mode,
            seed=args.seed,
            workers=args.workers,
            hero_sampling=sampler,
        )
        rec = {
            "label": label,
            "temperature": dict(sampler.temperature),
            "band": sampler.band,
            "edge": res["edge"],
            "se": res["se"],
            "modes": res["modes"],
            "leaster": {k: res["leaster"][k] for k in ("edge", "se", "n")},
            "non_leaster": {k: res["non_leaster"][k] for k in ("edge", "se", "n")},
            "deviating_frac": res["deviating_frac"],
            "minutes": (time.time() - t0) / 60.0,
        }
        results.append(rec)
        print(
            f"{label}: edge {rec['edge']:+.4f} ± {rec['se']:.4f} "
            f"(called {res['modes']['called']['edge']:+.4f} / jd {res['modes']['jd']['edge']:+.4f}; "
            f"leaster {rec['leaster']['edge']:+.4f} ± {rec['leaster']['se']:.4f}; "
            f"deviating {rec['deviating_frac']:.3f}) in {rec['minutes']:.0f} min",
            flush=True,
        )
        json.dump(
            {
                "ckpt": args.ckpt,
                "field": field,
                "heads": args.heads,
                "deals_per_mode": args.deals_per_mode,
                "seed": args.seed,
                "arms": results,
            },
            open(out, "w"),
            indent=1,
        )
    print(
        f"{'arm':<28}{'edge':>9}{'se':>8}{'called':>9}{'jd':>9}{'leaster':>9}{'dev':>7}"
    )
    for r in results:
        print(
            f"{r['label']:<28}{r['edge']:>+9.4f}{r['se']:>8.4f}"
            f"{r['modes']['called']['edge']:>+9.4f}{r['modes']['jd']['edge']:>+9.4f}"
            f"{r['leaster']['edge']:>+9.4f}{r['deviating_frac']:>7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
