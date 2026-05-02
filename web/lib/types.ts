export type TableSummary = {
  id: string;
  name: string;
  status: 'open' | 'playing' | 'finished';
  rules: Record<string, unknown>;
  fillWithAI: boolean;
  seats: Record<string, string | null>;
  seatIsAI?: Record<string, boolean>;
  host: string | null;
  runningBySeat?: Record<string, number>;
};

export type HandResult = {
  hand?: number;
  bySeat?: Record<string, { id: string; score: number }>;
  sum?: number;
};

/** The table object received inside WebSocket state messages. */
export type TableView = {
  id: string;
  name: string;
  status: 'open' | 'playing' | 'finished';
  seats: Record<string, string | null>;
  seatIsAI?: Record<string, boolean>;
  seatOccupants?: Record<string, string>;
  initialSeatOrder?: string[];
  resultsHistory?: HandResult[];
  host: string | null;
  rules?: Record<string, unknown>;
  runningBySeat?: Record<string, number>;
};

export type FinalState = {
  mode?: 'leaster' | 'normal';
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
  type: 'state';
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

export type GameOverMsg = { type: 'game_over'; table: TableView };

export type TableClosedMsg = { type: 'table_closed'; reason?: string; tableId?: string };
