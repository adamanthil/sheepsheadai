#!/usr/bin/env python3
"""Scan simulated games for defender trump-leads and emit reproducible cases.

A "defender trump-lead" is a spot where a player who is NOT the picker, NOT the
revealed partner, and NOT the secret partner (and the hand is not a leaster)
*leads* a trick with a trump card while at least one fail card was a legal lead.
This is contrary to standard human play and is the behavior we want to inspect
case-by-case in the web ``/analyze`` view.

The scanner drives the *exact* same deterministic simulation path that
``/analyze`` uses (``server.services.analyze.simulate_game``), so every
``(seed, partnerMode)`` it reports reproduces byte-for-byte when you type that
seed + partner mode into the Analyze page. The only deviation from calling the
service repeatedly is that the model is loaded once and cached (the analyze
service reloads it per request), which is safe because ``simulate_game`` resets
the agent's recurrent state at the start of every game.

Usage (from repo root):

    uv run python analysis/scan_defender_trump_leads.py \
        --num-seeds 500 --partner-mode 1 --out runs/defender_trump_leads.json

Then open the Analyze page, set the same Partner Mode, enter the reported
``seed`` (deterministic / greedy), and scroll the timeline to ``stepIndex`` to
inspect the action logits and full game state for that decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sheepshead import ACTION_LOOKUP, FAIL, TRUMP  # noqa: E402

TRUMP_SET = set(TRUMP)
FAIL_SET = set(FAIL)

DEFAULT_MODEL = str(REPO_ROOT / "final_pfsp_swish_ppo.pt")


# ---------------------------------------------------------------------------
# Load the model once and reuse it across seeds. ``simulate_game`` looks up
# ``load_agent`` from its own module namespace at call time, so patching the
# name there transparently swaps in our cached loader without touching any
# service code. ``simulate_game`` calls ``agent.reset_recurrent_state()`` at the
# start of every game, so a shared agent yields identical results to a fresh one.
# ---------------------------------------------------------------------------
import server.services.analyze as analyze_mod  # noqa: E402
from server.services.ai_loader import load_agent as _real_load_agent  # noqa: E402

_AGENT_CACHE: dict[str, object] = {}


def _cached_load_agent(model_path: str):
    agent = _AGENT_CACHE.get(model_path)
    if agent is None:
        agent = _real_load_agent(model_path)
        _AGENT_CACHE[model_path] = agent
    return agent


analyze_mod.load_agent = _cached_load_agent

from server.api.schemas import AnalyzeSimulateRequest  # noqa: E402
from server.services.analyze import simulate_game  # noqa: E402


@dataclass
class DefenderTrumpLeadCase:
    """One defender-leads-trump-with-fail-available decision."""

    seed: int
    partnerMode: int
    stepIndex: int
    trickIndex: int  # 0-based trick number
    seat: int
    seatName: str
    pickerSeat: int
    cardLed: str  # the trump card the defender led
    failOptions: List[str]  # fail cards that were legal to lead instead
    numFailOptions: int
    handTrumpCount: int
    handFailCount: int
    # Model preference signals at this decision (the "relative action logits"):
    chosenProb: Optional[float] = None
    chosenLogit: Optional[float] = None
    bestFailCard: Optional[str] = None
    bestFailProb: Optional[float] = None
    bestFailLogit: Optional[float] = None
    winProb: Optional[float] = None
    valueEstimate: Optional[float] = None


@dataclass
class ScanStats:
    seedsScanned: int = 0
    standardGames: int = 0
    leasterGames: int = 0
    defenderLeads: int = 0  # defender lead spots (non-leaster)
    defenderLeadsWithFail: int = 0  # of those, fail was a legal lead
    trumpLeadCases: int = 0  # of those, chose trump anyway

    @property
    def trumpLeadRate(self) -> float:
        if not self.defenderLeadsWithFail:
            return 0.0
        return self.trumpLeadCases / self.defenderLeadsWithFail


def _is_secret_partner(view: dict, partner_mode: int) -> bool:
    """Mirror Player.is_secret_partner using only the trace view + meta.

    JD mode: holds the Jack of Diamonds. Called-ace mode: holds the called
    card. False if the hand is alone or a leaster.
    """
    if view.get("alone") or view.get("is_leaster"):
        return False
    hand = view.get("hand") or []
    if partner_mode == 0:  # PARTNER_BY_JD
        return "JD" in hand
    called_card = view.get("called_card")
    return bool(called_card) and called_card in hand


def scan_game(
    resp, seed: int, partner_mode: int, stats: ScanStats
) -> List[DefenderTrumpLeadCase]:
    """Find all defender trump-leads in one simulated game's trace."""
    cases: List[DefenderTrumpLeadCase] = []

    is_leaster_game = bool(resp.final and resp.final.get("mode") == "leaster")
    if is_leaster_game:
        stats.leasterGames += 1
    else:
        stats.standardGames += 1

    for ad in resp.trace:
        # Must be a card play.
        if not ad.action.startswith("PLAY "):
            continue
        card = ad.action[5:]
        view = ad.view

        # Must be a *lead*: no card has been played into the current trick yet.
        current_trick = view.get("current_trick") or []
        if not all(c == "" for c in current_trick):
            continue

        # Exclude leasters entirely (no picker/partner -> no defender concept).
        if view.get("is_leaster"):
            continue

        seat = ad.seat
        picker = view.get("picker") or 0
        partner = view.get("partner") or 0
        is_picker = seat == picker
        is_partner = seat == partner  # 0 until revealed, so safe
        if is_picker or is_partner or _is_secret_partner(view, partner_mode):
            continue

        # This is a defender lead. Which fail cards were legal to lead?
        fail_options = sorted(
            (
                ACTION_LOOKUP[vid][5:]
                for vid in ad.validActionIds
                if ACTION_LOOKUP.get(vid, "").startswith("PLAY ")
                and ACTION_LOOKUP[vid][5:] in FAIL_SET
            ),
            key=FAIL.index,
        )
        stats.defenderLeads += 1
        if not fail_options:
            continue
        stats.defenderLeadsWithFail += 1

        # Only a case if the defender actually led trump (and not the UNDER token).
        if card not in TRUMP_SET:
            continue
        stats.trumpLeadCases += 1

        hand = view.get("hand") or []
        chosen = next((p for p in ad.probabilities if p.actionId == ad.actionId), None)
        # probabilities are sorted by prob desc, so first fail PLAY is the best.
        best_fail = next(
            (
                p
                for p in ad.probabilities
                if p.action.startswith("PLAY ") and p.action[5:] in FAIL_SET
            ),
            None,
        )

        cases.append(
            DefenderTrumpLeadCase(
                seed=seed,
                partnerMode=partner_mode,
                stepIndex=ad.stepIndex,
                trickIndex=int(view.get("current_trick_index", 0)),
                seat=seat,
                seatName=ad.seatName,
                pickerSeat=picker,
                cardLed=card,
                failOptions=fail_options,
                numFailOptions=len(fail_options),
                handTrumpCount=sum(1 for c in hand if c in TRUMP_SET),
                handFailCount=sum(1 for c in hand if c in FAIL_SET),
                chosenProb=chosen.prob if chosen else None,
                chosenLogit=chosen.logit if chosen else None,
                bestFailCard=best_fail.action[5:] if best_fail else None,
                bestFailProb=best_fail.prob if best_fail else None,
                bestFailLogit=best_fail.logit if best_fail else None,
                winProb=ad.winProb,
                valueEstimate=ad.valueEstimate,
            )
        )

    return cases


def format_case(c: DefenderTrumpLeadCase) -> str:
    fail_str = ", ".join(c.failOptions)
    pref = ""
    if c.chosenProb is not None and c.bestFailProb is not None:
        pref = (
            f"  | led {c.cardLed} p={c.chosenProb:.3f} (logit {c.chosenLogit:+.2f})"
            f"  vs best fail {c.bestFailCard} p={c.bestFailProb:.3f} (logit {c.bestFailLogit:+.2f})"
        )
    win = f"  winP={c.winProb:.2f}" if c.winProb is not None else ""
    return (
        f"seed={c.seed} pm={c.partnerMode} step={c.stepIndex} trick={c.trickIndex + 1} "
        f"{c.seatName}(seat {c.seat}) led {c.cardLed}  "
        f"fail avail [{fail_str}]{pref}{win}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model .pt")
    parser.add_argument(
        "--partner-mode",
        type=int,
        choices=[0, 1],
        default=1,
        help="0 = Jack of Diamonds, 1 = Called Ace (default, matches Analyze page)",
    )
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Stop after collecting this many cases.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write a JSON report (cases + stats).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case lines.")
    args = parser.parse_args()

    stats = ScanStats()
    all_cases: List[DefenderTrumpLeadCase] = []

    end_seed = args.start_seed + args.num_seeds
    for seed in range(args.start_seed, end_seed):
        req = AnalyzeSimulateRequest(
            seed=seed,
            partnerMode=args.partner_mode,
            deterministic=True,
            modelPath=args.model,
            maxSteps=args.max_steps,
        )
        resp = simulate_game(req)
        stats.seedsScanned += 1

        cases = scan_game(resp, seed, args.partner_mode, stats)
        for c in cases:
            all_cases.append(c)
            if not args.quiet:
                print(format_case(c))

        if args.limit_cases is not None and len(all_cases) >= args.limit_cases:
            break

    print("\n" + "=" * 72)
    print(
        f"Scanned {stats.seedsScanned} seeds "
        f"({stats.standardGames} standard, {stats.leasterGames} leaster), "
        f"partnerMode={args.partner_mode}"
    )
    print(
        f"Defender leads: {stats.defenderLeads}  "
        f"(with a fail lead available: {stats.defenderLeadsWithFail})"
    )
    print(
        f"Defender TRUMP-leads-with-fail-available: {stats.trumpLeadCases} "
        f"= {stats.trumpLeadRate:.1%} of defender leads with a fail option"
    )
    print(f"Cases collected: {len(all_cases)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "model": args.model,
                "partnerMode": args.partner_mode,
                "startSeed": args.start_seed,
                "numSeeds": args.num_seeds,
                "maxSteps": args.max_steps,
            },
            "stats": {**asdict(stats), "trumpLeadRate": stats.trumpLeadRate},
            "cases": [asdict(c) for c in all_cases],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {len(all_cases)} cases -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
