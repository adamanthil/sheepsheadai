#!/usr/bin/env python3
"""Scan simulated games for defender called-suit-lead adherence (convention C2).

Human convention (called-ace mode): a DEFENDER holding a called-suit fail leads
it while the called suit has not yet been led — the picker is rule-guaranteed to
hold the suit, the secret partner must surface the called card, so the lead
publicly identifies the partner and gives a void defender a shot at trumping
the 11-point ace. See notebooks/Convention_Optimality_202607.md (E1).

An ELIGIBLE node is a *lead* by a defender (not picker, not revealed partner,
not secret partner, non-leaster, non-alone) who holds at least one legal
called-suit fail lead AND at least one non-called-suit lead (a forced hand is
not a decision) while the called suit has never been led. The node is
ADHERENT when the card actually led is a called-suit fail.

Under-call hands (picker void in the called suit by rule) break the
"picker must follow" premise, so they are excluded from the primary rates and
tallied separately.

The scanner drives the same deterministic simulation path as ``/analyze``
(``server.services.analyze.simulate_game``), so every reported ``seed``
reproduces byte-for-byte on the Analyze page (Partner Mode = Called Ace).
Called-ace mode only: convention C2 does not exist in JD mode.

Usage (from repo root):

    uv run python -m sheepshead.analysis.scan_called_suit_leads \
        --num-seeds 500 --out runs/called_suit_leads.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Importing the trump-lead scanner installs the cached load_agent patch on the
# analyze service and provides set_scan_model + the secret-partner helper.
import sheepshead.analysis.scan_defender_trump_leads as scan  # noqa: E402
from server.api.schemas import AnalyzeSimulateRequest  # noqa: E402
from server.services.analyze import simulate_game  # noqa: E402
from sheepshead import ACTION_LOOKUP, FAIL, TRUMP_SET, UNDER_TOKEN  # noqa: E402

FAIL_SET = set(FAIL)

DEFAULT_MODEL = scan.DEFAULT_MODEL

PARTNER_MODE_CALLED_ACE = 1


@dataclass
class CalledSuitLeadNode:
    """One C2-eligible defender lead (adherent or not)."""

    seed: int
    partnerMode: int
    stepIndex: int
    trickIndex: int  # 0-based trick number
    seat: int
    seatName: str
    pickerSeat: int
    relPosFromPicker: int  # (seat - picker) % 5, in 1..4
    calledCard: str
    underCall: bool
    cardLed: str
    adhered: bool  # led a called-suit fail
    isFirstOpportunity: bool  # this seat's first eligible node in the game
    calledSuitOptions: List[str]  # called-suit fails that were legal leads
    numCalledSuitOptions: int
    handTrumpCount: int
    handFailCount: int
    # Model preference signals at this decision:
    chosenProb: Optional[float] = None
    chosenLogit: Optional[float] = None
    bestCalledCard: Optional[str] = None
    bestCalledProb: Optional[float] = None
    bestCalledLogit: Optional[float] = None
    winProb: Optional[float] = None
    valueEstimate: Optional[float] = None


@dataclass
class CalledSuitScanStats:
    seedsScanned: int = 0
    standardGames: int = 0
    leasterGames: int = 0
    aloneGames: int = 0
    underCallGames: int = 0
    defenderLeads: int = 0  # defender lead spots (non-leaster, non-alone)
    # Primary counters exclude under-call hands.
    eligible: int = 0
    adherent: int = 0
    eligibleTrick0: int = 0
    adherentTrick0: int = 0
    eligibleFirstOpp: int = 0
    adherentFirstOpp: int = 0
    # Under-call split (same eligibility test, picker void by rule).
    eligibleUnder: int = 0
    adherentUnder: int = 0
    # Positional split over primary eligible nodes: relPos -> [eligible, adherent]
    byRelPos: Dict[int, List[int]] = field(
        default_factory=lambda: {k: [0, 0] for k in (1, 2, 3, 4)}
    )

    @staticmethod
    def _rate(num: int, den: int) -> float:
        return num / den if den else 0.0

    @property
    def adherenceRate(self) -> float:
        return self._rate(self.adherent, self.eligible)

    @property
    def adherenceRateTrick0(self) -> float:
        return self._rate(self.adherentTrick0, self.eligibleTrick0)

    @property
    def adherenceRateFirstOpp(self) -> float:
        return self._rate(self.adherentFirstOpp, self.eligibleFirstOpp)

    @property
    def adherenceRateUnder(self) -> float:
        return self._rate(self.adherentUnder, self.eligibleUnder)


def _called_suit_fail(card: str, called_card: str) -> bool:
    """True when ``card`` is a fail of the called card's suit (QC/JC are trump,
    never called-suit; the suit letter is the last character for all fails)."""
    return card in FAIL_SET and card[-1] == called_card[-1]


def _called_suit_already_led(view: dict) -> bool:
    """Was any completed trick led in the called suit? Mirrors the engine's
    ``was_called_suit_played`` (which flips on trick completion when the led
    suit equals the called suit; an UNDER lead counts as the called suit)."""
    called = view.get("called_card")
    if not called:
        return False
    history = view.get("history") or []
    leaders = view.get("leaders") or []
    for t in range(int(view.get("current_trick_index", 0))):
        if t >= len(history) or t >= len(leaders) or not leaders[t]:
            continue
        lead = history[t][leaders[t] - 1]
        if not lead:
            continue
        if lead == UNDER_TOKEN or _called_suit_fail(lead, called):
            return True
    return False


def scan_game(
    resp, seed: int, stats: CalledSuitScanStats
) -> List[CalledSuitLeadNode]:
    """Find all C2-eligible defender leads in one simulated game's trace."""
    nodes: List[CalledSuitLeadNode] = []
    seats_with_opportunity: set[int] = set()

    is_leaster_game = bool(resp.final and resp.final.get("mode") == "leaster")
    if is_leaster_game:
        stats.leasterGames += 1
    else:
        stats.standardGames += 1

    counted_alone = False
    counted_under = False

    for ad in resp.trace:
        if not ad.action.startswith("PLAY "):
            continue
        card = ad.action[5:]
        view = ad.view

        if view.get("is_leaster"):
            continue
        if view.get("alone"):
            if not counted_alone:
                stats.aloneGames += 1
                counted_alone = True
            continue
        called = view.get("called_card")
        if not called:
            continue
        under_call = bool(view.get("called_under"))
        if under_call and not counted_under:
            stats.underCallGames += 1
            counted_under = True

        # Must be a *lead*: no card played into the current trick yet.
        current_trick = view.get("current_trick") or []
        if not all(c == "" for c in current_trick):
            continue

        seat = ad.seat
        picker = view.get("picker") or 0
        partner = view.get("partner") or 0
        if (
            seat == picker
            or seat == partner
            or scan._is_secret_partner(view, PARTNER_MODE_CALLED_ACE)
        ):
            continue
        stats.defenderLeads += 1

        if _called_suit_already_led(view):
            continue

        # Which called-suit fails were legal leads? Eligibility also requires a
        # non-called-suit alternative: a forced all-called-suit hand is not a
        # decision and would trivially inflate adherence.
        legal_leads = [
            ACTION_LOOKUP[vid][5:]
            for vid in ad.validActionIds
            if ACTION_LOOKUP.get(vid, "").startswith("PLAY ")
        ]
        called_options = sorted(
            (c for c in legal_leads if _called_suit_fail(c, called)),
            key=FAIL.index,
        )
        if not called_options or len(called_options) == len(legal_leads):
            continue

        # ELIGIBLE node.
        adhered = _called_suit_fail(card, called)
        trick_index = int(view.get("current_trick_index", 0))
        first_opp = seat not in seats_with_opportunity
        seats_with_opportunity.add(seat)

        if under_call:
            stats.eligibleUnder += 1
            stats.adherentUnder += int(adhered)
        else:
            stats.eligible += 1
            stats.adherent += int(adhered)
            if trick_index == 0:
                stats.eligibleTrick0 += 1
                stats.adherentTrick0 += int(adhered)
            if first_opp:
                stats.eligibleFirstOpp += 1
                stats.adherentFirstOpp += int(adhered)
            rel = (seat - picker) % 5
            stats.byRelPos[rel][0] += 1
            stats.byRelPos[rel][1] += int(adhered)

        hand = view.get("hand") or []
        chosen = next((p for p in ad.probabilities if p.actionId == ad.actionId), None)
        # probabilities are sorted by prob desc, so the first called-suit PLAY
        # is the policy's best called-suit lead.
        best_called = next(
            (
                p
                for p in ad.probabilities
                if p.action.startswith("PLAY ")
                and _called_suit_fail(p.action[5:], called)
            ),
            None,
        )

        nodes.append(
            CalledSuitLeadNode(
                seed=seed,
                partnerMode=PARTNER_MODE_CALLED_ACE,
                stepIndex=ad.stepIndex,
                trickIndex=trick_index,
                seat=seat,
                seatName=ad.seatName,
                pickerSeat=picker,
                relPosFromPicker=(seat - picker) % 5,
                calledCard=called,
                underCall=under_call,
                cardLed=card,
                adhered=adhered,
                isFirstOpportunity=first_opp,
                calledSuitOptions=called_options,
                numCalledSuitOptions=len(called_options),
                handTrumpCount=sum(1 for c in hand if c in TRUMP_SET),
                handFailCount=sum(1 for c in hand if c in FAIL_SET),
                chosenProb=chosen.prob if chosen else None,
                chosenLogit=chosen.logit if chosen else None,
                bestCalledCard=best_called.action[5:] if best_called else None,
                bestCalledProb=best_called.prob if best_called else None,
                bestCalledLogit=best_called.logit if best_called else None,
                winProb=ad.winProb,
                valueEstimate=ad.valueEstimate,
            )
        )

    return nodes


def format_node(n: CalledSuitLeadNode) -> str:
    opts = ", ".join(n.calledSuitOptions)
    verdict = "LED called suit" if n.adhered else f"led {n.cardLed} instead"
    pref = ""
    if n.chosenProb is not None and n.bestCalledProb is not None:
        pref = (
            f"  | chose p={n.chosenProb:.3f}"
            f"  vs best called {n.bestCalledCard} p={n.bestCalledProb:.3f}"
            f" (logit {n.bestCalledLogit:+.2f})"
        )
    under = " UNDER" if n.underCall else ""
    return (
        f"seed={n.seed} step={n.stepIndex} trick={n.trickIndex + 1} "
        f"{n.seatName}(seat {n.seat}, picker+{n.relPosFromPicker}) "
        f"called={n.calledCard}{under} [{opts}] -> {verdict}{pref}"
    )


def print_summary(stats: CalledSuitScanStats) -> None:
    print("\n" + "=" * 72)
    print(
        f"Scanned {stats.seedsScanned} seeds "
        f"({stats.standardGames} standard, {stats.leasterGames} leaster, "
        f"{stats.aloneGames} alone, {stats.underCallGames} under-call)"
    )
    print(f"Defender leads (called-ace, non-alone): {stats.defenderLeads}")
    print(
        f"C2-eligible nodes (holds called-suit fail, suit unled): {stats.eligible}"
        f"  -> adherent {stats.adherent} = {stats.adherenceRate:.1%}"
    )
    print(
        f"  at trick 0:         {stats.eligibleTrick0:4d} eligible"
        f"  -> {stats.adherenceRateTrick0:.1%} adherent"
    )
    print(
        f"  first opportunity:  {stats.eligibleFirstOpp:4d} eligible"
        f"  -> {stats.adherenceRateFirstOpp:.1%} adherent"
    )
    print(
        f"  under-call (excl.): {stats.eligibleUnder:4d} eligible"
        f"  -> {stats.adherenceRateUnder:.1%} adherent"
    )
    print("  by seat rel. picker (picker+k, primary nodes):")
    for rel in (1, 2, 3, 4):
        elig, adh = stats.byRelPos[rel]
        rate = f"{adh / elig:.1%}" if elig else "  n/a"
        print(f"    picker+{rel}: {elig:4d} eligible  -> {rate} adherent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model .pt")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--limit-nodes",
        type=int,
        default=None,
        help="Stop after collecting this many eligible nodes.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write a JSON report (nodes + stats).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-node lines."
    )
    args = parser.parse_args()

    scan.set_scan_model(args.model)
    stats = CalledSuitScanStats()
    all_nodes: List[CalledSuitLeadNode] = []

    end_seed = args.start_seed + args.num_seeds
    for seed in range(args.start_seed, end_seed):
        req = AnalyzeSimulateRequest(
            seed=seed,
            partnerMode=PARTNER_MODE_CALLED_ACE,
            deterministic=True,
            maxSteps=args.max_steps,
        )
        resp = simulate_game(req)
        stats.seedsScanned += 1

        nodes = scan_game(resp, seed, stats)
        for n in nodes:
            all_nodes.append(n)
            if not args.quiet:
                print(format_node(n))

        if args.limit_nodes is not None and len(all_nodes) >= args.limit_nodes:
            break

    print_summary(stats)
    print(f"Nodes collected: {len(all_nodes)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "model": args.model,
                "partnerMode": PARTNER_MODE_CALLED_ACE,
                "startSeed": args.start_seed,
                "numSeeds": args.num_seeds,
                "maxSteps": args.max_steps,
            },
            "stats": {
                **asdict(stats),
                "adherenceRate": stats.adherenceRate,
                "adherenceRateTrick0": stats.adherenceRateTrick0,
                "adherenceRateFirstOpp": stats.adherenceRateFirstOpp,
                "adherenceRateUnder": stats.adherenceRateUnder,
            },
            "nodes": [asdict(n) for n in all_nodes],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {len(all_nodes)} nodes -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
