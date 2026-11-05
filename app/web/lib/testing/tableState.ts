import type { GameView, TableStateMsg, TableView } from "../types";

type Overrides = Partial<Omit<TableStateMsg, "view" | "table">> & {
  view?: Partial<GameView>;
  table?: Partial<TableView>;
};

/**
 * A table state message for unit tests: seat 1's view of a called-ace
 * leasters table, at the first pick decision, with the overrides applied.
 */
export function tableState(overrides: Overrides = {}): TableStateMsg {
  const { view, table, ...rest } = overrides;
  return {
    type: "state",
    yourSeat: 1,
    actorSeat: 1,
    moveSeq: 0,
    isHost: true,
    state: {},
    valid_actions: [],
    ...rest,
    table: {
      id: "t1",
      status: "playing",
      name: "Test table",
      host: "c1",
      initialNames: {},
      initialSeatOrder: [],
      resultsHistory: [],
      rules: { allPassMode: "leasters", doubleOnTheBump: true, partnerMode: 1 },
      scoreMultiplier: 1,
      passOutCount: 0,
      runningBySeat: {},
      seatIsAI: {},
      seatAccount: {},
      seatTakeable: {},
      seatLeavingAfterHand: {},
      closingAfterHand: false,
      seatOccupants: {},
      seats: { "1": "You", "2": "Ann", "3": "Bo", "4": "Cy", "5": "Di" },
      ...table,
    },
    view: {
      hand: ["QC", "JD", "AH", "10S", "9C", "7D"],
      picker: 0,
      partner: 0,
      is_leaster: false,
      current_trick: ["", "", "", "", ""],
      current_trick_index: 0,
      last_trick: null,
      last_trick_winner: null,
      was_trick_just_completed: false,
      is_done: false,
      final: null,
      called_card: null,
      called_card_display: null,
      called_under: false,
      alone: false,
      bury: [],
      under_card: null,
      ...view,
    },
  };
}
