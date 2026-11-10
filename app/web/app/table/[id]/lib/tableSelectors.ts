import type {
  FinalState,
  GameView,
  HandResult,
  ScoreSegment,
  TableStateMsg,
  TableView,
} from "../../../../lib/types";
import { parseCard } from "../../../../lib/ds";
import { gameModeLabel, scoringModeLabel } from "../../../../lib/rules";
import { accountForSeat, nameForSeat, isAiSeat } from "../utils/seatMath";
import { relSeat } from "./seatLayout";
import { getSeatRole, type InterludeMode, type TablePhase } from "./phase";
import type { CallOption, SeatView } from "../components/stage/types";

export type PhaseLabelKind =
  "pick" | "bury" | "call" | "under" | "setup" | "play" | "done";

const SUIT_NAME: Record<string, string> = {
  C: "Clubs",
  S: "Spades",
  H: "Hearts",
  D: "Diamonds",
};

export function rulesBadgeText(
  rules: Record<string, unknown> | undefined,
): string | null {
  if (!rules) return null;
  return `${gameModeLabel(rules)} · ${scoringModeLabel(rules)}`;
}

/** Label for the stake the hand in progress is being played for, or null at
 * the base stake — the overwhelmingly common case, which needs no badge. */
export function stakeBadgeText(multiplier: number | undefined): string | null {
  return multiplier && multiplier > 1 ? `${multiplier}× Doubler` : null;
}

export function buildCallOptions(
  validActions: number[],
  actionLookup: Record<string, string>,
): CallOption[] {
  const out: CallOption[] = [];
  for (const aid of validActions) {
    const label = actionLookup[String(aid)];
    if (!label) continue;
    if (label.startsWith("CALL ")) {
      const rest = label.slice(5); // e.g. "AC" or "AC UNDER"
      const [code, ...mods] = rest.split(" ");
      const { rank, suit } = parseCard(code);
      const under = mods.includes("UNDER");
      const display = suit
        ? `${rank === "A" ? "Ace" : rank} of ${SUIT_NAME[suit] ?? suit}${under ? " (under)" : ""}`
        : label;
      out.push({ actionId: aid, label, code, display });
    } else if (label === "ALONE") {
      out.push({ actionId: aid, label, code: null, display: "Go alone" });
    } else if (label.startsWith("JD")) {
      // Drawn as the jack itself; a null code would draw the Alone panel.
      out.push({
        actionId: aid,
        label,
        code: "JD",
        display: "Jack of Diamonds",
      });
    }
  }
  return out;
}

export function computeIsYourTurn(lastState: TableStateMsg): boolean {
  return (
    lastState.yourSeat !== null && lastState.actorSeat === lastState.yourSeat
  );
}

export function computeIsHost(lastState: TableStateMsg): boolean {
  return lastState.isHost ?? false;
}

export function computeValidActionStrings(
  lastState: TableStateMsg | null,
  actionLookup: Record<string, string>,
): Set<string> {
  const s = new Set<string>();
  if (!lastState) return s;
  for (const id of lastState.valid_actions) {
    const label = actionLookup[String(id)];
    if (label) s.add(label);
  }
  return s;
}

export function computeActionIdByString(
  actionLookup: Record<string, string>,
): Record<string, number> {
  const m: Record<string, number> = {};
  Object.entries(actionLookup).forEach(([id, label]) => {
    const n = Number(id);
    if (Number.isFinite(n)) m[label] = n;
  });
  return m;
}

/** The "kind" used for labels/helper text. */
export function computeKind(
  phase: TablePhase,
  yourMode: InterludeMode,
): PhaseLabelKind {
  if (phase === "pick") return "pick";
  if (phase === "play") return "play";
  if (phase === "done") return "done";
  if (yourMode === "bury") return "bury";
  if (yourMode === "call") return "call";
  if (yourMode === "under") return "under";
  return "setup";
}

// Seat 1 always picks first and leads the first trick, so the dealer — the
// player to its right — is always seat 5. The deal still passes left:
// each new hand rotates every occupant down a seat, seat 1's into seat 5.
export const DEALER_SEAT = 5;

export function buildSeats(
  lastState: TableStateMsg,
  table: TableView,
  yourSeat: number | null,
  started: boolean,
): SeatView[] {
  // A spectator sees the ring from seat 1, with no seat marked as theirs.
  const pov = yourSeat ?? 1;
  return [1, 2, 3, 4, 5].map((absSeat) => ({
    absSeat,
    rel: relSeat(absSeat, pov),
    name: nameForSeat(absSeat, table),
    isAI: isAiSeat(absSeat, table),
    account: accountForSeat(absSeat, table),
    role: getSeatRole(lastState, absSeat, started),
    you: absSeat === yourSeat,
    leaving: !!table.seatLeavingAfterHand[String(absSeat)],
    dealer: absSeat === DEALER_SEAT,
  }));
}

export function computeDisplayCards(
  showPrev: boolean,
  view: GameView,
): string[] {
  return showPrev && view.last_trick ? view.last_trick : view.current_trick;
}

export function computeWinnerSeat(
  showPrev: boolean,
  view: GameView,
): number | null {
  return showPrev ? view.last_trick_winner : null;
}

export interface ScoreBlockRow {
  hand: number;
  multiplier: number;
  /** The picking team lost: the picker's score is negative. */
  set: boolean;
  /** Per column: the total through this hand, and this hand's own score. */
  cells: Array<{ total: number; delta: number }>;
}

/** Hands played by one set of players, with running totals from 0. */
export interface ScoreBlock {
  columns: ScoreSegment["columns"];
  rows: ScoreBlockRow[];
}

/** The scores card's blocks, oldest first, each starting over at the deal
 * a new player was dealt in. Blocks with no finished hand are left out. */
export function scoreBlocks(table: TableView): ScoreBlock[] {
  const blocks: ScoreBlock[] = (table.scoreSegments ?? []).map((s) => ({
    columns: s.columns,
    rows: [],
  }));
  const history: HandResult[] = table.resultsHistory ?? [];
  history.forEach((h, idx) => {
    const block = blocks[h.segment ?? 0];
    if (!block) return;
    const entries = Object.values(h.bySeat ?? {});
    const picker = entries.find((e) => h.picker != null && e.id === h.picker);
    const prev = block.rows.at(-1)?.cells;
    block.rows.push({
      hand: h.hand || idx + 1,
      multiplier: h.multiplier || 1,
      set: !!picker && picker.score < 0,
      cells: block.columns.map((c, i) => {
        const delta = entries
          .filter((e) => e.column === c.id)
          .reduce((sum, e) => sum + (e.score || 0), 0);
        return { delta, total: (prev?.[i].total ?? 0) + delta };
      }),
    });
  });
  return blocks.filter((b) => b.rows.length > 0);
}

/** Finished hands in the current block, the one the running scores count. */
export function handsInCurrentBlock(table: TableView): number {
  const current = (table.scoreSegments?.length ?? 0) - 1;
  return (table.resultsHistory ?? []).filter(
    (h: HandResult) => (h.segment ?? 0) === current,
  ).length;
}

export function computeHandNumber(table: TableView): number {
  return (table.resultsHistory?.length ?? 0) + 1;
}

export function computeHasLastTrick(view: GameView): boolean {
  return view.last_trick?.length === 5;
}

/** One piece of the context line: plain text, or a card drawn in its suit's
 * colour. Kept apart so a player's name is never read as a card code. */
export type ContextPart = string | { card: string };

export interface HandContext {
  parts: ContextPart[];
  /** The partner is known, so the call is only a reference now. */
  faded: boolean;
}

const ORDINALS = ["1st", "2nd", "3rd", "4th", "5th"];

/**
 * The line under the table: what the hand was declared as, kept in view
 * as a reference, e.g. "A♣ called", "A♥ called under", "J♦ partner",
 * "Kyle goes alone", "Leaster", followed by the doubler when one is on.
 * It doesn't change as the hand is played; once the partner is revealed
 * it fades. While the player has the last trick open it says only that;
 * a trick held on the table after it is taken keeps the usual line.
 */
export function computeContext(
  lastState: TableStateMsg,
  phase: TablePhase,
  lastTrickOpen: boolean,
): HandContext {
  if (lastTrickOpen) return { parts: ["Last trick"], faded: false };
  const { view, table, yourSeat } = lastState;
  const picker = view.picker || 0;
  const who = (seat: number) =>
    seat === yourSeat ? "You" : nameForSeat(seat, table);
  const jdMode = table.rules?.partnerMode === 0;

  const facts: ContextPart[][] = [];
  let hasPartner = false;
  if (phase === "pick" && yourSeat !== null) {
    // Seats are numbered in pick order from the dealer's left.
    facts.push([`${ORDINALS[yourSeat - 1]} Position`]);
  } else if (view.is_leaster) {
    facts.push(["Leaster"]);
  } else if (picker) {
    if (view.alone) {
      facts.push([
        `${who(picker)} ${picker === yourSeat ? "go" : "goes"} alone`,
      ]);
    } else if (view.called_card) {
      facts.push([
        { card: view.called_card },
        view.called_under ? " called under" : " called",
      ]);
      hasPartner = true;
    } else if (jdMode && phase !== "interlude") {
      facts.push([{ card: "JD" }, " partner"]);
      hasPartner = true;
    } else {
      facts.push([`${who(picker)} picked`]);
    }
  }
  const stake = stakeBadgeText(table.scoreMultiplier);
  if (stake) facts.push([stake]);

  return {
    parts: facts.flatMap((fact, i) => (i === 0 ? fact : [" · ", ...fact])),
    faded: hasPartner && !!view.partner,
  };
}

/** The finished hand in one line, for the mobile table while the chat is
 * in use: the outcome, and your own score when you played it. */
export function resultLine(
  final: FinalState,
  table: TableView,
  yourSeat: number | null,
): string {
  const signed = (n: number) => (n > 0 ? `+${n}` : String(n));
  const parts: string[] = [];
  if (final.mode === "leaster") {
    const winner = final.winner ?? 0;
    parts.push(
      `${nameForSeat(winner, table)} wins the leaster with ${final.points_taken?.[winner - 1] ?? 0}`,
    );
  } else {
    parts.push(
      `Picker ${final.picker_score} – ${final.defender_score} Defenders`,
    );
  }
  if (yourSeat !== null) {
    parts.push(`You ${signed(final.scores?.[yourSeat - 1] ?? 0)}`);
  }
  return parts.join(" · ");
}

/** The turn clock as minutes and seconds, e.g. "2:00", "0:05". */
export function formatClock(secondsLeft: number): string {
  const minutes = Math.floor(secondsLeft / 60);
  const seconds = secondsLeft % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}
