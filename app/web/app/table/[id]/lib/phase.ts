import type { TableStateMsg } from "../../../../lib/types";

export type TablePhase = "pick" | "interlude" | "play" | "done";

// Role shown on a seat. PENDING = it is this seat's turn to pick/pass.
export type SeatRole = "PICKER" | "PARTNER" | "PASS" | "PENDING" | null;

// Your sub-mode during the interlude (between picking and the first trick).
// Derived from the set of valid action labels, since those tell us exactly
// what the server is asking you to do right now.
export type InterludeMode = "bury" | "call" | "waiting";

/**
 * Has the play (trick-taking) phase started? Mirrors the server's
 * `play_started` flag with fallbacks for older state shapes.
 */
export function playStarted(msg: TableStateMsg): boolean {
  if (msg.state?.play_started === 1) return true;
  if (msg.view.current_trick_index > 0) return true;
  const picker = msg.view.picker || 0;
  if (picker > 0) {
    const ct = msg.view.current_trick as string[] | undefined;
    if (ct && ct.some((c) => c !== "")) return true;
  }
  return false;
}

/** The single source of truth for "what moment is this hand in?". */
export function derivePhase(msg: TableStateMsg): {
  phase: TablePhase;
  isLeaster: boolean;
} {
  const isLeaster = !!msg.view.is_leaster;
  if (msg.view.is_done) return { phase: "done", isLeaster };
  if (playStarted(msg)) return { phase: "play", isLeaster };
  const picker = msg.view.picker || 0;
  if (!isLeaster && picker === 0) return { phase: "pick", isLeaster };
  return { phase: "interlude", isLeaster };
}

/**
 * Role badge for a seat. Extracted verbatim from the table page's original
 * getPlayerStatus memo, with PICK/PICKER collapsed to a single PICKER label.
 */
export function getSeatRole(
  msg: TableStateMsg,
  absSeat: number,
  started: boolean,
): SeatRole {
  const picker = msg.view.picker || 0;
  const partner = msg.view.partner || 0;
  const actorSeat = msg.actorSeat;
  const inPickDecision = !msg.view.is_leaster && picker === 0;

  if (started) {
    if (absSeat === picker) return "PICKER";
    if (absSeat === partner) return "PARTNER";
    return null;
  }

  if (picker > 0) {
    if (absSeat === picker) return "PICKER";
    if (absSeat < picker) return "PASS";
    return null;
  }

  if (!inPickDecision) return null;
  if (actorSeat && absSeat === actorSeat) return "PENDING";
  if (actorSeat && absSeat < actorSeat) return "PASS";
  return null;
}

/** Your interlude sub-mode, derived from the valid action labels. */
export function interludeMode(validActionStrings: Set<string>): InterludeMode {
  for (const label of validActionStrings) {
    if (label.startsWith("BURY")) return "bury";
  }
  for (const label of validActionStrings) {
    if (label.startsWith("CALL") || label === "ALONE" || label.startsWith("JD"))
      return "call";
  }
  return "waiting";
}

export const PHASE_HELP: Record<TablePhase, string> = {
  pick: "Pick or pass",
  interlude: "Set up the hand",
  play: "Play a card",
  done: "Hand complete",
};
