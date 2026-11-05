// Moves drawn before the server confirms them. The server's state is never
// edited: the table is drawn from it with the pending move laid over the
// top, so a refused move goes back to exactly what the server last said.
import type { TableStateMsg } from "./types";

/** A move this player sent that the table has not shown back yet. */
export interface PendingMove {
  /** The action's label, e.g. "PLAY QC" or "PASS". */
  label: string;
  /** moveSeq of the state the move was made from. Any state with another
   * moveSeq has moved on, and supersedes the prediction. */
  baseSeq: number;
}

/** Whether a move is drawn before the server confirms it. Bury and under
 * cards are staged in the center and confirmed by the player already, so
 * they wait for the server as before. */
export function isPredicted(label: string): boolean {
  return !label.startsWith("BURY ") && !label.startsWith("UNDER ");
}

const SEATS = 5;

/**
 * The table as it will be once ``label`` lands: your controls gone and,
 * where it can be known without the server, the move itself shown.
 *
 * - PLAY <card>: the card leaves your hand for your place in the trick, and
 *   the turn passes to the next seat, or to no one when your card completes
 *   the trick (its winner is the server's to say).
 * - PASS: the turn passes to the next seat (bidding runs from seat 1 to 5);
 *   the last seat's pass waits, as a leaster or a redeal follows.
 * - PICK: you are the picker; the blind arrives with the server's state.
 * - Anything else (a call, going alone, PLAY UNDER): the controls go.
 */
export function applyPendingMove(
  state: TableStateMsg,
  label: string,
): TableStateMsg {
  const seat = state.yourSeat;
  if (seat === null) return state;
  const sent = { ...state, valid_actions: [] };

  if (label.startsWith("PLAY ") && label !== "PLAY UNDER") {
    const card = label.slice("PLAY ".length);
    const trick = [...state.view.current_trick];
    trick[seat - 1] = card;
    const complete = trick.every((c) => c !== "");
    return {
      ...sent,
      actorSeat: complete ? null : (seat % SEATS) + 1,
      view: {
        ...state.view,
        hand: state.view.hand.filter((c) => c !== card),
        current_trick: trick,
      },
    };
  }
  if (label === "PASS") {
    return { ...sent, actorSeat: seat < SEATS ? seat + 1 : seat };
  }
  if (label === "PICK") {
    return { ...sent, actorSeat: null, view: { ...state.view, picker: seat } };
  }
  return { ...sent, actorSeat: null };
}
