#!/usr/bin/env python3
"""
Per-generation endpoint evaluation for the extended league run.

Called by analysis/run_extended_league.py after each generation; importable
and usable standalone for manual re-runs. Reuses the anchored-gauntlet
machinery from analysis/rigorous_eval.py in-process because the stopping rule
needs the raw per-deal score vectors, which the CSV output discards.

Endpoint design: interleaved three-checkpoint composite
-------------------------------------------------------
The standing measurement rule says endpoints are the mean of the last THREE
checkpoint panels (single checkpoints churn +-0.15-0.2 per 25k). Three full
4000-deal panels would triple the cost, so instead the fixed deal list is
split by deal_index mod 3 and deal i is evaluated by checkpoint
c[(i mod 3) + 1] (c1..c3 = the last three 50k-cadence checkpoints of the
generation). Properties:

  * deal-noise SE identical to a single N-deal panel (every deal appears
    exactly once) -> MDE ~= 0.035 at N=4000 is preserved;
  * snapshot noise averaged over 3 checkpoints;
  * the deal -> checkpoint-rank mapping, the deals, and the frozen PANEL-A
    field assignment are identical every generation, so per-deal differences
    between any two generations stay CRN-paired and deal luck cancels.

Both partner modes are evaluated (N/2 deals each, called-ace first) and
concatenated into one per-deal vector.

Trump-lead leak telemetry rides along for free: a passive DecisionProbe
(rigorous_eval.DecisionProbe) watches hero decisions during the same games and
records the trick-0/1 defender trump-lead rate and trump probability mass --
only at nodes where a legal non-trump lead existed (forced all-trump leads are
tallied separately; they are not a leak). Node definition matches
analysis/trump_lead_probe.py; the probe is provably passive (scores are
bit-identical with it on or off).

Example
-------
  uv run python analysis/league_progress_eval.py \
    --ckpts ck_900000.pt ck_950000.pt ck_1000000.pt --out panel_gen1.npz
  uv run python analysis/league_progress_eval.py --h2h gen2.pt gen1.pt
"""

from __future__ import annotations

import sys


import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from sheepshead.analysis.bootstrap import IntervalStat, bootstrap_interval
from sheepshead.analysis.rigorous_eval import (
    DecisionProbe,
    Model,
    ModelRegistry,
    evaluate_hero_in_field,
    make_panel_field_fn,
)
from sheepshead.analysis.panels import PANEL_A
from sheepshead.analysis.trump_lead_probe import _is_secret_partner, _lead_options
from sheepshead.agent.ppo import load_agent
from sheepshead import (
    ACTION_LOOKUP,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    TRUMP_SET,
)
from sheepshead.training.training_utils import paired_edge

# Frozen experiment constants (pre-registered in notebooks/Extended_League_202607.md)
FIELD_SEED = 20260619  # rigorous_eval.run_gauntlet's frozen field-assignment seed
PANEL_SEED = 42  # longitudinal deal panel
CONFIRM_SEED = 20260706  # fresh-deal confirmation panel (stop-time only)
H2H_SEED = 20260708  # gen-vs-previous head-to-head deal set
N_BOOT = 10_000
N_CHECKPOINTS = 3  # composite width; also the interleave modulus
# Fixed mode order: called-ace half first, then JD.
MODES = (PARTNER_BY_CALLED_ACE, PARTNER_BY_JD)
MODE_NAMES = {PARTNER_BY_CALLED_ACE: "called", PARTNER_BY_JD: "jd"}


def panel_deal_seeds(n: int, seed: int) -> List[int]:
    """Deal seeds exactly as rigorous_eval.main derives them from --seed."""
    rng = random.Random(seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]


def design_hash(
    seed: int, n_deals: int, field_seed: int, panel_paths: Sequence[str]
) -> str:
    """Fingerprint of everything that must match for per-deal vectors to be
    paired across generations. Deliberately excludes the hero checkpoints."""
    payload = json.dumps(
        {
            "seed": seed,
            "n_deals": n_deals,
            "field_seed": field_seed,
            "panel": sorted(str(p) for p in panel_paths),
            "modes": [MODE_NAMES[m] for m in MODES],
            "interleave": N_CHECKPOINTS,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Trump-lead leak collector (passive; node definition = trump_lead_probe.py)
# --------------------------------------------------------------------------- #
class TrumpLeadCollector:
    """Tally hero defender trump-leads at tricks 0-1 during panel games.

    An *opportunity* is a hero decision where the hero leads trick 0 or 1 as a
    plain defender and has BOTH a legal trump lead and a legal non-trump lead.
    All-trump nodes are forced (counted separately, excluded from the rate);
    no-trump nodes cannot leak and are ignored entirely.
    """

    def __init__(self) -> None:
        self.stats: Dict[int, Dict[str, float]] = {
            t: {"opportunities": 0, "forced": 0, "trump_leads": 0, "prob_mass_sum": 0.0}
            for t in (0, 1)
        }

    def wants(self, game, player, valid_actions) -> bool:
        return bool(
            game.play_started
            and not game.is_leaster
            and game.current_trick <= 1
            and game.leader == player.position
            and game.cards_played == 0
            and not (
                player.is_picker
                or player.is_partner
                or game.partner == player.position
                or _is_secret_partner(game, player)
            )
        )

    def record(self, game, player, valid_actions, action, probs) -> None:
        trumps, fails = _lead_options(player)
        tally = self.stats[int(game.current_trick)]
        if trumps and not fails:
            tally["forced"] += 1
            return
        if not (trumps and fails):
            return  # no trump available: cannot leak
        tally["opportunities"] += 1
        name = ACTION_LOOKUP[action]
        if name.startswith("PLAY ") and name.split(" ", 1)[1] in TRUMP_SET:
            tally["trump_leads"] += 1
        tally["prob_mass_sum"] += sum(
            float(probs[a - 1])
            for a in valid_actions
            if ACTION_LOOKUP[a].startswith("PLAY ")
            and ACTION_LOOKUP[a].split(" ", 1)[1] in TRUMP_SET
        )

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for t, tally in self.stats.items():
            n = tally["opportunities"]
            out[f"t{t}_def_leads"] = int(n)
            out[f"t{t}_forced"] = int(tally["forced"])
            out[f"t{t}_trump_lead_rate"] = (
                tally["trump_leads"] / n if n else float("nan")
            )
            out[f"t{t}_trump_prob_mass"] = (
                tally["prob_mass_sum"] / n if n else float("nan")
            )
        return out


# --------------------------------------------------------------------------- #
# Endpoint evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Endpoint:
    """A generation endpoint: composite panel score + leak telemetry."""

    score: IntervalStat
    per_deal: np.ndarray  # both modes concatenated, called half first
    mode_means: Dict[str, float]
    trump_lead: Dict[str, float]
    ckpts: List[str]
    seed: int
    hash: str


def eval_endpoint(
    ckpt_paths: Sequence[str],
    n_deals: int = 3996,  # nearest multiple of 2 modes x 3 ckpts to 4000
    seed: int = PANEL_SEED,
    field_seed: int = FIELD_SEED,
    panel_paths: Sequence[str] = tuple(PANEL_A),
    out_npz: Optional[Path] = None,
    registry: Optional[ModelRegistry] = None,
    n_boot: int = N_BOOT,
    log: bool = True,
) -> Endpoint:
    """Evaluate a 3-checkpoint composite endpoint on the frozen PANEL-A field.

    ckpt_paths: the generation's last three checkpoints, OLDEST FIRST
    (boundary-100k, boundary-50k, boundary). A single-checkpoint endpoint
    (e.g. the gen-0 resume baseline) is expressed by repeating the path.
    """
    if len(ckpt_paths) != N_CHECKPOINTS:
        raise ValueError(f"need exactly {N_CHECKPOINTS} checkpoints, oldest first")
    if n_deals % (2 * N_CHECKPOINTS) != 0:
        raise ValueError(f"n_deals must be divisible by {2 * N_CHECKPOINTS}")

    registry = registry or ModelRegistry()
    heroes: List[Model] = [registry.get(Path(p)) for p in ckpt_paths]
    panel = [registry.get(Path(p)) for p in panel_paths]
    panel.sort(key=lambda m: str(m.filepath))  # order-canonical, as run_gauntlet

    per_mode = n_deals // 2
    deal_seeds = panel_deal_seeds(per_mode, seed)
    collector = TrumpLeadCollector()
    probe = DecisionProbe(wants=collector.wants, record=collector.record)

    halves: List[np.ndarray] = []
    mode_means: Dict[str, float] = {}
    for mode in MODES:
        field_fn = make_panel_field_fn(panel, per_mode, rng_seed=field_seed)
        per_deal_mode = np.zeros(per_mode, dtype=np.float64)
        for j, hero in enumerate(heroes):
            global_idx = list(range(j, per_mode, N_CHECKPOINTS))
            sub_seeds = [deal_seeds[d] for d in global_idx]

            def sub_field(d_local: int, k: int, _idx=global_idx) -> Dict[int, Model]:
                return field_fn(_idx[d_local], k)

            ev = evaluate_hero_in_field(hero, sub_field, sub_seeds, mode, probe=probe)
            per_deal_mode[global_idx] = ev.deal_score
            if log:
                print(
                    f"  [{MODE_NAMES[mode]}] ckpt {j + 1}/{N_CHECKPOINTS} "
                    f"({Path(ckpt_paths[j]).name}): {len(global_idx)} deals, "
                    f"mean {ev.deal_score.mean():+.3f}",
                    flush=True,
                )
        halves.append(per_deal_mode)
        mode_means[MODE_NAMES[mode]] = float(per_deal_mode.mean())

    per_deal = np.concatenate(halves)
    boot_idx = _endpoint_boot_idx(n_deals, n_boot, seed)
    score = bootstrap_interval(per_deal, boot_idx)
    endpoint = Endpoint(
        score=score,
        per_deal=per_deal,
        mode_means=mode_means,
        trump_lead=collector.summary(),
        ckpts=[str(p) for p in ckpt_paths],
        seed=seed,
        hash=design_hash(seed, n_deals, field_seed, panel_paths),
    )
    if out_npz is not None:
        save_endpoint(endpoint, Path(out_npz))
    return endpoint


def _endpoint_boot_idx(n_deals: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_deals, size=(n_boot, n_deals))


def save_endpoint(e: Endpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        per_deal=e.per_deal,
        meta=json.dumps(
            {
                "score": vars(e.score),
                "mode_means": e.mode_means,
                "trump_lead": e.trump_lead,
                "ckpts": e.ckpts,
                "seed": e.seed,
                "hash": e.hash,
            }
        ),
    )
    print(f"Wrote endpoint: {path}")


def load_endpoint(path: Path) -> Endpoint:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return Endpoint(
        score=IntervalStat(**meta["score"]),
        per_deal=data["per_deal"],
        mode_means=meta["mode_means"],
        trump_lead=meta["trump_lead"],
        ckpts=meta["ckpts"],
        seed=meta["seed"],
        hash=meta["hash"],
    )


def paired_gain(a: Endpoint, b: Endpoint, n_boot: int = N_BOOT) -> IntervalStat:
    """Paired per-deal gain of endpoint a over endpoint b (a - b)."""
    if a.hash != b.hash:
        raise ValueError(
            "endpoints were evaluated under different designs "
            f"({a.hash[:12]} vs {b.hash[:12]}); per-deal pairing is invalid"
        )
    boot_idx = _endpoint_boot_idx(len(a.per_deal), n_boot, a.seed)
    return bootstrap_interval(a.per_deal - b.per_deal, boot_idx)


# --------------------------------------------------------------------------- #
# Head-to-head vs the previous generation
# --------------------------------------------------------------------------- #
def h2h(
    gen_ckpt: str, prev_ckpt: str, n_deals: int = 2000, seed: int = H2H_SEED
) -> Dict:
    """CRN paired edge of this generation over the previous one, measured in
    the previous generation's field (training_utils.paired_edge). The frozen
    seed keeps the deal set constant across generations.

    SUPERSEDED for the extended-league stop rule by h2h_duplicate (amendment
    2026-07-19): at 2000 deals this instrument's se ≈ 0.055, which makes
    criterion B (edge >= 0.05 at 2 se) unreachable below +0.11. Kept for
    comparability with earlier recorded numbers."""
    challenger = load_agent(gen_ckpt)
    incumbent = load_agent(prev_ckpt)
    return paired_edge(challenger, incumbent, incumbent, n_deals, seed=seed)


def h2h_duplicate(
    gen_ckpt: str,
    prev_ckpt: str,
    n_deals_per_mode: int = 2000,
    seed: int = 42,
    n_boot: int = 5000,
) -> Dict:
    """Duplicate-bridge h2h (Extended_League amendment 2026-07-19).

    The candidate is seated in all 5 seats per CRN deal against a field made
    entirely of the previous checkpoint (rigorous_eval gauntlet with a
    one-member panel), both partner modes. The anchor's score in its own
    field is 0 by symmetry, so the candidate's score/hand IS the paired
    edge — at ~3.5x less deal noise than training_utils.paired_edge at
    equal deals (se ~0.015 combined vs ~0.055). Pipeline and seed (42)
    match the 2026-07-19 hypothesis-battery runs exactly, so gens 1-2 of
    the perceiver-shared-v2 continuation carry those recorded values.
    """
    import random as _random

    from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
    from sheepshead.analysis.rigorous_eval import (
        ModelRegistry,
        _bootstrap_deal_indices,
        run_gauntlet,
    )

    registry = ModelRegistry()
    cand = registry.get(Path(gen_ckpt))
    anchor = registry.get(Path(prev_ckpt))
    seed_rng = _random.Random(seed)
    deal_seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(n_deals_per_mode)]
    boot_idx = _bootstrap_deal_indices(
        n_deals_per_mode, n_boot, np.random.default_rng(seed)
    )

    mode_edges: Dict[str, Dict[str, float]] = {}
    deal_scores = []
    for mode, name in ((PARTNER_BY_CALLED_ACE, "called"), (PARTNER_BY_JD, "jd")):
        rep = run_gauntlet([cand], [anchor], deal_seeds, mode, boot_idx)[0]
        mode_edges[name] = {"edge": rep.score.mean, "se": rep.score.se}
        deal_scores.append(rep.deal_score)
    edge = (mode_edges["called"]["edge"] + mode_edges["jd"]["edge"]) / 2.0
    se = 0.5 * float(
        np.sqrt(mode_edges["called"]["se"] ** 2 + mode_edges["jd"]["se"] ** 2)
    )
    pooled = np.concatenate(deal_scores)
    return {
        "edge": edge,
        "se": se,
        # paired_edge-compatible analogs (ties get half credit; a nonzero
        # duplicate deal score means the candidate's presence changed the
        # outcome vs the anchor's symmetric 0).
        "win_frac": float(((pooled > 0) + 0.5 * (pooled == 0)).mean()),
        "deviating_frac": float((pooled != 0).mean()),
        "n_deals": 2 * n_deals_per_mode,
        "instrument": "duplicate_bridge",
        "modes": mode_edges,
    }


# --------------------------------------------------------------------------- #
# CLI (manual re-runs)
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="League generation endpoint evaluation")
    p.add_argument("--ckpts", nargs=3, help="last three checkpoints, oldest first")
    p.add_argument("--out", type=str, default=None, help="output .npz path")
    p.add_argument("--deals", type=int, default=3996)
    p.add_argument("--seed", type=int, default=PANEL_SEED)
    p.add_argument("--h2h", nargs=2, metavar=("GEN", "PREV"), default=None)
    p.add_argument("--h2h-deals", type=int, default=2000)
    args = p.parse_args(argv)

    if args.h2h:
        res = h2h(args.h2h[0], args.h2h[1], n_deals=args.h2h_deals)
        print(json.dumps(res, indent=2))
        return 0
    if not args.ckpts:
        p.error("provide --ckpts or --h2h")
    e = eval_endpoint(
        args.ckpts,
        n_deals=args.deals,
        seed=args.seed,
        out_npz=Path(args.out) if args.out else None,
    )
    print(
        f"endpoint: {e.score.mean:+.4f} [{e.score.lo:+.4f}, {e.score.hi:+.4f}] "
        f"modes={e.mode_means}"
    )
    print(f"trump-lead leak: {json.dumps(e.trump_lead, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
