import type { components } from "./api.gen";

/** REST shapes are generated from the server's OpenAPI schema — regenerate
 * with `npm run gen:api` after changing server/api/schemas.py. */
export type TablePublic = components["schemas"]["TablePublic"];
export type Rules = components["schemas"]["RulesInput"];
export type JoinTableResponse = components["schemas"]["JoinTableResponse"];
export type PlayerPublic = components["schemas"]["PlayerPublic"];

export type TableSummary = TablePublic;

export type HandResult = {
  hand?: number;
  bySeat?: Record<string, { id: string; score: number }>;
  sum?: number;
};

/** A single chat log entry, as sent over the table WebSocket's
 * chat:init / chat:append messages. */
export interface ChatMessage {
  id: string;
  table_id: string;
  type: "player" | "system";
  author: string | null;
  body: string;
  timestamp: number;
}

/** The table object inside WebSocket state messages is the same
 * to_public_dict() payload the REST endpoints return. */
export type TableView = TablePublic;

export type FinalState = {
  mode?: "leaster" | "standard";
  winner?: number;
  picker?: number;
  partner?: number;
  picker_score?: number;
  defender_score?: number;
  scores?: number[];
  points_taken?: number[];
};

/** The per-client game view received in WebSocket state messages. */
export type GameView = {
  hand: string[];
  picker: number;
  partner: number;
  is_leaster: boolean;
  current_trick: string[];
  current_trick_index: number;
  last_trick: string[] | null;
  last_trick_winner: number | null;
  last_trick_points?: number;
  was_trick_just_completed: boolean;
  is_done: boolean;
  final?: FinalState;
  called_card: string | null;
  called_card_display: string | null;
  called_under: boolean;
  alone: boolean;
};

export type TableStateMsg = {
  type: "state";
  table: TableView;
  yourSeat: number;
  actorSeat: number | null;
  isHost: boolean;
  state: {
    play_started?: number;
    is_leaster?: number;
    current_trick?: number;
    picker_rel?: number;
    partner_rel?: number;
    [key: string]: unknown;
  };
  view: GameView;
  valid_actions: number[];
};
