"""Hard-stop gates for the league trainer: the n=1000 convention
adherence guard (run inside the main-phase loop) and the generation-
boundary expert-refresh cert (run between phases). Both halt the run for
operator review via GateExit, whose exit codes are part of the trainer's
external contract (the orchestrator and operators dispatch on them).

Split out of train_league_ppo.py (Stage 2 of the league-trainer
maintainability refactor); the probe/eval collaborators
(greedy_health_probe, paired_edge, load_agent) resolve in THIS module.
"""

from __future__ import annotations

import json
import os

import numpy as np

from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead.training.training_utils import greedy_health_probe, paired_edge

# Fixed deal-set seed for the anchored strength probe: every probe replays the
# SAME deals, so consecutive probe values are paired and the trend line is
# policy movement, not deal luck.
LEAGUE_ANCHOR_EVAL_SEED = 20260701
# Fixed seed for the n=1000 adherence guard probe: successive guard readings
# share the deal stream, so probe-to-probe deltas are paired (policy-driven,
# not deal-luck). Same seed as the offline monitoring probes (§12.17).
ADHERENCE_GUARD_SEED = 98765


class GateExit(SystemExit):
    """A trainer gate halted the run for operator review.

    Subclasses SystemExit so the process exit code is the gate code and
    existing ``pytest.raises(SystemExit)`` handling keeps working. Codes:
    3 = adherence guard hard stop, 4 = boundary cert failure.
    """


def check_adherence_guard(
    training_agent: PPOAgent, args, episode: int, stop_ckpt: str, league
) -> None:
    """Convention adherence guard, two-tier (CE_Teacher_Design §3;
    §12.17/§12.21 protocol): the 200-300-game greedy probe cannot resolve
    sub-5-point convention regressions (it masked an 8-point partner-trump
    deficit for a full teacher run), so the guard reruns the probe at
    n=1000 on a FIXED seed (successive probes are paired). HARD tier
    (partner below the floor, or t0 trump-lead above the ceiling — the
    scramble signature) stops the run for operator review via GateExit(3),
    saving ``stop_ckpt`` first; the NOTIFY tier (partner below the notify
    line) only prints, because §12.20 showed partner dips during teaching
    can be oscillation with a restoring force, not collapse."""
    probe = greedy_health_probe(
        training_agent,
        n_games=int(getattr(args, "adherence_guard_games", 1000)),
        seed=ADHERENCE_GUARD_SEED,
    )
    print(
        f"🛡️ Adherence guard (n={probe['games']}): "
        f"called-suit {probe['called_suit_lead_rate']:.1f}% "
        f"t0-trump {probe['t0_trump_lead_rate']:.1f}% "
        f"partner-trump {probe['partner_trump_lead_rate']:.1f}%",
        flush=True,
    )
    violations = []
    partner_floor = getattr(args, "guard_partner_floor", None)
    if partner_floor is not None and probe["partner_trump_lead_rate"] < float(
        partner_floor
    ):
        violations.append(
            f"partner trump-lead {probe['partner_trump_lead_rate']:.1f}% "
            f"< hard floor {float(partner_floor):.1f}%"
        )
    t0_ceiling = getattr(args, "guard_t0_ceiling", None)
    if t0_ceiling is not None and probe["t0_trump_lead_rate"] > float(t0_ceiling):
        violations.append(
            f"t0 trump-lead {probe['t0_trump_lead_rate']:.1f}% "
            f"> ceiling {float(t0_ceiling):.1f}%"
        )
    partner_notify = getattr(args, "guard_partner_notify", None)
    if (
        not violations
        and partner_notify is not None
        and probe["partner_trump_lead_rate"] < float(partner_notify)
    ):
        print(
            f"🛡️⚠️ Adherence NOTIFY: partner trump-lead "
            f"{probe['partner_trump_lead_rate']:.1f}% < notify line "
            f"{float(partner_notify):.1f}% (hard floor "
            f"{float(partner_floor):.1f}%) — continuing",
            flush=True,
        )
    if violations:
        training_agent.save(stop_ckpt)
        league.save()
        print(
            "🚨 ADHERENCE GUARD STOP: "
            + "; ".join(violations)
            + f" — checkpoint saved to {stop_ckpt}; run halted for "
            "operator review",
            flush=True,
        )
        raise GateExit(3)


def run_boundary_cert(
    training_agent: PPOAgent, args, generation: int, checkpoint_dir: str
) -> dict:
    """Absolute-anchor expert-refresh cert (CE_Teacher_Design §3), run on the
    generation-boundary candidate before it may become the next generation's
    frozen expert.

    Two components, both against FIXED absolute bars (never relative to the
    previous generation — a relative cert lets a refresh chain ratchet drift
    into the certified regime):

    * n=--cert-games adherence battery at --cert-seeds distinct fixed deal
      seeds, judged on the ACROSS-SEED MEAN (single reads are luck-of-phase
      — the §12.22 lesson: consolidation called-suit swung 39.9-51.0 across
      reads): mean partner trump-lead >= --cert-partner-floor AND mean t0
      trump-lead <= --cert-t0-ceiling.
    * Paired CRN h2h vs the run's fixed cert anchor (--cert-anchor-ckpt,
      default the ORIGINAL expert checkpoint): edge must not be
      significantly negative (edge + 2*SE >= 0).

    The exploiter gate that follows every boundary is the third cert
    component and keeps its existing flow. Result is persisted to
    boundary_cert_gen<g>.json for the run record."""
    seeds = [ADHERENCE_GUARD_SEED + i for i in range(int(args.cert_seeds))]
    probes = [
        greedy_health_probe(
            training_agent, n_games=int(args.cert_games), seed=deal_seed
        )
        for deal_seed in seeds
    ]
    partner_mean = float(
        np.mean([probe["partner_trump_lead_rate"] for probe in probes])
    )
    t0_mean = float(np.mean([probe["t0_trump_lead_rate"] for probe in probes]))
    called_mean = float(np.mean([probe["called_suit_lead_rate"] for probe in probes]))

    anchor_path = args.cert_anchor_resolved
    anchor_agent = load_agent(anchor_path)
    saved_memories = training_agent.snapshot_player_memories()
    h2h = paired_edge(
        training_agent,
        anchor_agent,
        anchor_agent,
        n_deals=int(args.cert_h2h_deals),
        seed=LEAGUE_ANCHOR_EVAL_SEED,
        log_every=0,
    )
    training_agent.restore_player_memories(saved_memories)

    failures = []
    if partner_mean < float(args.cert_partner_floor):
        failures.append(
            f"partner trump-lead mean {partner_mean:.1f}% "
            f"< cert floor {float(args.cert_partner_floor):.1f}%"
        )
    if t0_mean > float(args.cert_t0_ceiling):
        failures.append(
            f"t0 trump-lead mean {t0_mean:.1f}% "
            f"> cert ceiling {float(args.cert_t0_ceiling):.1f}%"
        )
    if h2h["edge"] + 2.0 * h2h["se"] < 0.0:
        failures.append(
            f"h2h vs {os.path.basename(anchor_path)} significantly negative "
            f"({h2h['edge']:+.3f} ± {h2h['se']:.3f})"
        )
    result = {
        "generation": generation,
        "passed": not failures,
        "failures": failures,
        "adherence": {
            "seeds": seeds,
            "games_per_seed": int(args.cert_games),
            "partner_trump_by_seed": [
                probe["partner_trump_lead_rate"] for probe in probes
            ],
            "t0_trump_by_seed": [probe["t0_trump_lead_rate"] for probe in probes],
            "called_suit_by_seed": [probe["called_suit_lead_rate"] for probe in probes],
            "partner_trump_mean": partner_mean,
            "t0_trump_mean": t0_mean,
            "called_suit_mean": called_mean,
        },
        "h2h": {
            "anchor": anchor_path,
            "edge": h2h["edge"],
            "se": h2h["se"],
            "n_deals": h2h["n_deals"],
        },
    }
    cert_path = os.path.join(checkpoint_dir, f"boundary_cert_gen{generation}.json")
    with open(cert_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"📜 Boundary cert gen {generation}: partner {partner_mean:.1f}% "
        f"t0 {t0_mean:.1f}% called-suit {called_mean:.1f}% "
        f"(means over {len(seeds)} seeds x {int(args.cert_games)} games) | "
        f"h2h vs anchor {h2h['edge']:+.3f} ± {h2h['se']:.3f} -> "
        f"{'PASS' if result['passed'] else 'FAIL'}",
        flush=True,
    )
    return result
