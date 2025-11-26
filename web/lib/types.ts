export type TableSummary = {
  id: string;
  name: string;
  status: 'open' | 'playing' | 'finished';
  rules: Record<string, any>;
  fillWithAI: boolean;
  seats: Record<string, string | null>;
  seatIsAI?: Record<string, boolean>;
  host: string | null;
  hostId?: string | null;
  runningBySeat?: Record<string, number>;
};

export type TableStateMsg = {
  type: 'state';
  table: any;
  yourSeat: number;
  actorSeat: number | null;
  state: {
    play_started?: number;
    is_leaster?: number;
    current_trick?: number;
    picker_rel?: number;
    partner_rel?: number;
    [key: string]: unknown;
  };
  view: any;
  valid_actions: number[];
};

export type GameOverMsg = { type: 'game_over'; table: any };

export type TableClosedMsg = { type: 'table_closed'; reason?: string; tableId?: string };
