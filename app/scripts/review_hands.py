#!/usr/bin/env python
"""Review every human decision in a database of finished hands against the
model, and report how the numbers look.

Read-only: it loads the hands, replays them through
server.services.hand_review and prints a summary. Nothing is written to the
database. With --out, every reviewed decision is also saved as JSON lines.

Run from the repo root:
    uv run python app/scripts/review_hands.py \
        --dsn postgres://sheepshead:sheepshead@localhost:5433/sheepshead_prod
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import asyncpg

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

from server.services.hand_review import (  # noqa: E402
    DecisionReview,
    HandRecord,
    Readout,
    ReplayError,
    recorded_actions,
    review_hand,
)
from sheepshead.agent import ppo  # noqa: E402
from sheepshead.game import ACTION_LOOKUP  # noqa: E402

DEFAULT_DSN = "postgres://sheepshead:sheepshead@localhost:5433/sheepshead_prod"
DEFAULT_MODEL = "models/sheepsheadai-1.1-202611.pt"

GAMES_SQL = """
SELECT g.game_id, g.time_closed, t.name AS table_name, t.is_doublers,
       g.is_called_partner, g.is_double_on_the_bump, g.is_alone, g.is_leaster,
       called.code AS called_card, under.code AS under_card,
       (SELECT array_agg(c.code) FROM cardset_card cc JOIN card c USING (card_id)
         WHERE cc.cardset_id = g.blind_id) AS blind,
       (SELECT array_agg(c.code) FROM cardset_card cc JOIN card c USING (card_id)
         WHERE cc.cardset_id = g.bury_id) AS bury
FROM game g
JOIN game_table t USING (game_table_id)
LEFT JOIN card called ON called.card_id = g.called_card_id
LEFT JOIN card under ON under.card_id = g.under_card_id
WHERE g.time_closed IS NOT NULL
ORDER BY g.time_closed
"""

SEATS_SQL = """
SELECT gp.game_id, gp.position, gp.player_id, gp.name, gp.is_picker, gp.score,
       gp.is_substituted_pick, gp.ai_actions,
       (SELECT array_agg(c.code) FROM cardset_card cc JOIN card c USING (card_id)
         WHERE cc.cardset_id = gp.starting_hand_id) AS hand
FROM game_player gp
"""

PLAYS_SQL = """
SELECT t.game_id, t.index AS trick, tc.index AS play, gp.position,
       c.code AS card, tc.is_substituted
FROM trick t
JOIN trick_card tc USING (trick_id)
JOIN game_player gp USING (game_player_id)
JOIN card c USING (card_id)
ORDER BY t.game_id, t.index, tc.index
"""


@dataclass
class StoredHand:
    game_id: str
    when: str
    table: str
    record: HandRecord
    # Human seats -> display name.
    humans: Dict[int, str]
    scores: Dict[int, Optional[int]]
    # (seat, trick, card) plays and pick decisions the AI made for a human.
    substituted_plays: Set[Tuple[int, int]]
    substituted_pick: Set[int]
    # Seats with AI decisions that no flag accounts for (a call or a bury).
    unflagged_ai: Set[int]


async def load_hands(dsn: str) -> List[StoredHand]:
    conn = await asyncpg.connect(dsn)
    try:
        games = await conn.fetch(GAMES_SQL)
        seats = await conn.fetch(SEATS_SQL)
        plays = await conn.fetch(PLAYS_SQL)
    finally:
        await conn.close()

    seats_by_game: Dict[Any, List[asyncpg.Record]] = {}
    for row in seats:
        seats_by_game.setdefault(row["game_id"], []).append(row)
    plays_by_game: Dict[Any, List[asyncpg.Record]] = {}
    for row in plays:
        plays_by_game.setdefault(row["game_id"], []).append(row)

    hands: List[StoredHand] = []
    for g in games:
        gid = g["game_id"]
        tricks: List[List[Tuple[int, str]]] = []
        substituted_plays: Set[Tuple[int, int]] = set()
        flagged: Dict[int, int] = {}
        for p in plays_by_game.get(gid, []):
            while len(tricks) <= p["trick"]:
                tricks.append([])
            tricks[p["trick"]].append((p["position"], p["card"]))
            if p["is_substituted"]:
                substituted_plays.add((p["position"], p["trick"]))
                flagged[p["position"]] = flagged.get(p["position"], 0) + 1
        picker: Optional[int] = None
        humans: Dict[int, str] = {}
        substituted_pick: Set[int] = set()
        unflagged_ai: Set[int] = set()
        for s in seats_by_game[gid]:
            seat = s["position"]
            if s["is_picker"]:
                picker = seat
            if s["player_id"] is None:
                continue
            humans[seat] = s["name"]
            if s["is_substituted_pick"]:
                substituted_pick.add(seat)
            accounted = flagged.get(seat, 0) + int(s["is_substituted_pick"])
            if s["ai_actions"] > accounted:
                unflagged_ai.add(seat)
        hands.append(
            StoredHand(
                game_id=str(gid),
                when=g["time_closed"].strftime("%b %d %H:%M"),
                table=g["table_name"],
                record=HandRecord(
                    partner_mode=1 if g["is_called_partner"] else 0,
                    double_on_the_bump=g["is_double_on_the_bump"],
                    doublers=g["is_doublers"],
                    hands={s["position"]: list(s["hand"]) for s in seats_by_game[gid]},
                    blind=list(g["blind"]),
                    picker=picker,
                    alone=bool(g["is_alone"]),
                    called_card=g["called_card"],
                    under_card=g["under_card"],
                    bury=list(g["bury"] or []),
                    tricks=tricks,
                ),
                humans=humans,
                scores={s["position"]: s["score"] for s in seats_by_game[gid]},
                substituted_plays=substituted_plays,
                substituted_pick=substituted_pick,
                unflagged_ai=unflagged_ai,
            )
        )
    return hands


def skipped_positions(hand: StoredHand) -> Set[Tuple[int, int]]:
    """(seat, position in the action sequence) of the decisions the AI made
    on a human seat's behalf."""
    skip: Set[Tuple[int, int]] = set()
    trick = 0
    plays_in_trick = 0
    for position, (seat, action_id) in enumerate(recorded_actions(hand.record)):
        action = ACTION_LOOKUP[action_id]
        if action in ("PICK", "PASS"):
            if seat in hand.substituted_pick:
                skip.add((seat, position))
        elif action.startswith("PLAY "):
            if (seat, trick) in hand.substituted_plays:
                skip.add((seat, position))
            plays_in_trick += 1
            if plays_in_trick == 5:
                trick, plays_in_trick = trick + 1, 0
    return skip


@dataclass
class Row:
    """One reviewed decision with the hand it came from."""

    game_id: str
    when: str
    table: str
    player: str
    final_score: Optional[int]
    hand: List[str]
    review: DecisionReview

    @property
    def cost_win(self) -> Optional[float]:
        r = self.review
        if r.agreed or r.chosen is None or r.model is None:
            return None
        return r.chosen.win_prob - r.model.win_prob

    @property
    def cost_score(self) -> Optional[float]:
        r = self.review
        if r.agreed or r.chosen is None or r.model is None:
            return None
        return r.chosen.score - r.model.score

    def label(self) -> str:
        r = self.review
        where = f"trick {r.trick + 1}" if r.trick is not None else r.kind
        prob = dict(r.policy)
        return (
            f"{self.when}  {self.player[:12]:<12} {where:<8} "
            f"{r.action:<12} model {r.model_action:<12} "
            f"(p={prob[r.model_action]:.2f} vs {prob[r.action]:.2f})"
        )


def pct(x: float) -> str:
    return f"{100 * x:+.0f}"


def quantiles(values: Sequence[float]) -> str:
    if len(values) < 2:
        return "n/a"
    qs = statistics.quantiles(values, n=20)
    return (
        f"p5 {qs[0]:+.2f}  p25 {qs[4]:+.2f}  median {qs[9]:+.2f}  "
        f"p75 {qs[14]:+.2f}  p95 {qs[18]:+.2f}"
    )


def report(rows: List[Row], top: int) -> None:
    open_rows = [r for r in rows if not r.review.forced]
    print(f"\nDecisions reviewed: {len(rows)}  ({len(rows) - len(open_rows)} forced)")
    print("\nAgreement with the model's first choice (open decisions)")
    for kind in ("pick", "call", "under", "bury", "play"):
        of_kind = [r for r in open_rows if r.review.kind == kind]
        if of_kind:
            agreed = sum(r.review.agreed for r in of_kind)
            print(
                f"  {kind:<6} {agreed:>4} / {len(of_kind):<4} {agreed / len(of_kind):.0%}"
            )
    agreed = sum(r.review.agreed for r in open_rows)
    print(
        f"  {'all':<6} {agreed:>4} / {len(open_rows):<4} {agreed / len(open_rows):.0%}"
    )

    costed = [r for r in open_rows if r.cost_win is not None]
    no_estimate = sum(
        1 for r in open_rows if not r.review.agreed and r.cost_win is None
    )
    wins = [r.cost_win for r in costed if r.cost_win is not None]
    scores = [r.cost_score for r in costed if r.cost_score is not None]
    print(f"\nDisagreements with a cost: {len(costed)}  (no estimate: {no_estimate})")
    print(f"  win estimate, yours minus model's:   {quantiles(wins)}")
    print(f"  expected score, yours minus model's: {quantiles(scores)}")
    worse = sum(1 for w in wins if w < 0)
    print(
        f"  model rates its own choice higher in {worse} of {len(wins)}"
        f" ({worse / max(1, len(wins)):.0%})"
    )
    terminal = sum(
        1
        for r in costed
        if (r.review.chosen and r.review.chosen.terminal)
        or (r.review.model and r.review.model.terminal)
    )
    print(f"  read from a finished hand rather than an estimate: {terminal}")
    for kind in ("pick", "call", "bury", "play"):
        ws = [r.cost_win for r in costed if r.review.kind == kind and r.cost_win]
        if ws:
            print(
                f"  {kind:<5} n={len(ws):<4} mean win cost {statistics.fmean(ws):+.3f}"
            )

    def show(title: str, chosen: List[Row]) -> None:
        print(f"\n{title}")
        for r in chosen:
            rv = r.review
            assert rv.chosen is not None and rv.model is not None
            print(
                f"  {pct(r.cost_win or 0):>4} win {r.cost_score or 0:+5.2f} score  "
                f"{r.label()}\n"
                f"        yours {rv.chosen.win_prob:.0%}/{rv.chosen.score:+.2f}"
                f"  model's {rv.model.win_prob:.0%}/{rv.model.score:+.2f}"
                f"  hand {' '.join(r.hand)}  result {r.final_score}"
            )

    by_cost = sorted(costed, key=lambda r: r.cost_win or 0.0)
    show(f"Costliest decisions (top {top})", by_cost[:top])
    show(
        f"Model rates your choice above its own (top {top})",
        [r for r in reversed(by_cost[-top:]) if (r.cost_win or 0) > 0],
    )

    picks = [r for r in rows if r.review.kind == "pick"]
    differ = [r for r in picks if not r.review.agreed]
    print(f"\nPick decisions: {len(picks)}  (disagreements: {len(differ)})")
    print(
        f"  picked, model passes: {sum(r.review.action == 'PICK' for r in differ)}"
        f"   passed, model picks: {sum(r.review.action == 'PASS' for r in differ)}"
    )
    compared = [r for r in picks if r.review.pick_all_blinds is not None]
    if compared:
        print(
            "\n  PICK read over all 325 blinds vs with the blind that was there"
            " (win% / score), and PASS"
        )

        def cell(x: Optional[Readout]) -> str:
            return f"{x.win_prob:>4.0%}/{x.score:+5.2f}" if x else "    n/a    "

        for r in compared:
            rv = r.review
            prob = dict(rv.policy)
            print(
                f"  {r.when} {r.player[:10]:<10} seat {rv.seat} did {rv.action:<4} "
                f"p(pick)={prob['PICK']:.2f}  pick {cell(rv.pick_all_blinds)}  "
                f"actual-blind {cell(rv.pick_actual_blind)}  "
                f"pass {cell(rv.pass_actual_deal)}  {' '.join(r.hand)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="review only the first N hands")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--every-pick",
        action="store_true",
        help="compare PICK with PASS at every pick decision, not only disagreements",
    )
    parser.add_argument("--out", help="write every reviewed decision as JSON lines")
    args = parser.parse_args()

    hands = asyncio.run(load_hands(args.dsn))
    if args.limit:
        hands = hands[: args.limit]
    agent = ppo.load_agent(args.model)

    rows: List[Row] = []
    skipped: Dict[str, int] = {}
    evaluations = 0
    reviewed = 0
    started = time.perf_counter()
    slowest = 0.0
    for n, hand in enumerate(hands, 1):
        seats = set(hand.humans) - hand.unflagged_ai
        if not seats:
            reason = "AI made an unflagged call or bury" if hand.humans else "no human"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        t0 = time.perf_counter()
        try:
            result = review_hand(
                agent,
                hand.record,
                seats,
                skip=skipped_positions(hand),
                every_pick=args.every_pick,
            )
        except ReplayError as err:
            skipped[f"does not replay: {err}"] = (
                skipped.get(f"does not replay: {err}", 0) + 1
            )
            continue
        slowest = max(slowest, time.perf_counter() - t0)
        reviewed += 1
        evaluations += result.evaluations
        for review in result.decisions:
            rows.append(
                Row(
                    game_id=hand.game_id,
                    when=hand.when,
                    table=hand.table,
                    player=hand.humans[review.seat],
                    final_score=hand.scores[review.seat],
                    hand=hand.record.hands[review.seat],
                    review=review,
                )
            )
        if n % 25 == 0:
            print(f"  {n}/{len(hands)} hands", file=sys.stderr)
    elapsed = time.perf_counter() - started

    print(f"Hands: {len(hands)}  reviewed: {reviewed}")
    for reason, count in sorted(skipped.items()):
        print(f"  skipped ({reason}): {count}")
    print(
        f"Model evaluations: {evaluations}  in {elapsed:.1f}s "
        f"({1000 * elapsed / max(1, evaluations):.1f} ms each, "
        f"{elapsed / max(1, reviewed):.2f}s per hand, slowest hand {slowest:.1f}s) "
        f"on {ppo.device}"
    )
    report(rows, args.top)

    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {
                            "game_id": r.game_id,
                            "when": r.when,
                            "table": r.table,
                            "player": r.player,
                            "final_score": r.final_score,
                            "hand": r.hand,
                            **asdict(r.review),
                        }
                    )
                    + "\n"
                )
        print(f"\nWrote {len(rows)} decisions to {args.out}")


if __name__ == "__main__":
    main()
