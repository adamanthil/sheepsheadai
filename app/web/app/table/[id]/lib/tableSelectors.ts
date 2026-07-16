import type { ChatMessage, GameView, TableStateMsg, TableView } from "../../../../lib/types";
import { parseCard } from "../../../../lib/ds";
import { nameForSeat, isAiSeat } from "../utils/seatMath";
import { relSeat } from "./seatLayout";
import { getSeatRole, type InterludeMode, type TablePhase } from "./phase";
import type { CallOption, SeatView } from "../components/stage/types";

export type PhaseLabelKind =
  | "pick"
  | "bury"
  | "call"
  | "setup"
  | "play"
  | "done";

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
  const partner = rules.partnerMode === 0 ? "Jack of Diamonds" : "Called Ace";
  const scoring = rules.doubleOnTheBump ? "Double on Bump" : "Symmetric";
  return `${partner} · ${scoring}`;
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
      out.push({
        actionId: aid,
        label,
        code: null,
        display: "Jack of Diamonds",
      });
    }
  }
  return out;
}

export function computeIsYourTurn(lastState: TableStateMsg): boolean {
  return lastState.actorSeat === lastState.yourSeat;
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
  return "setup";
}

export function buildSeats(
  lastState: TableStateMsg,
  table: TableView,
  yourSeat: number,
  started: boolean,
): SeatView[] {
  return [1, 2, 3, 4, 5].map((absSeat) => ({
    absSeat,
    rel: relSeat(absSeat, yourSeat),
    name: nameForSeat(absSeat, table),
    isAI: isAiSeat(absSeat, table),
    role: getSeatRole(lastState, absSeat, started),
    you: absSeat === yourSeat,
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

export function computeHandNumber(table: TableView): number {
  return (table.resultsHistory?.length ?? 0) + 1;
}

export function computeHasLastTrick(view: GameView): boolean {
  return view.last_trick?.length === 5;
}

export function computePrevText(
  showPrev: boolean,
  hasLastTrick: boolean,
  view: GameView,
  table: TableView,
): string | null {
  if (!(showPrev && hasLastTrick)) return null;
  return `Trick to ${nameForSeat(view.last_trick_winner, table)} · ${view.last_trick_points ?? 0} pts`;
}

export function computeLastMessage(chatMessages: ChatMessage[]): string {
  return chatMessages.length
    ? chatMessages[chatMessages.length - 1].body
    : "Hand in progress";
}
