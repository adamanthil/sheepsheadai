export type TableSummary = {
  id: string;
  name: string;
  status: 'open' | 'playing' | 'finished';
  rules: Record<string, any>;
  fillWithAI: boolean;
  seats: Record<string, string | null>;
  host: string | null;
  hostId?: string | null;
  runningBySeat?: Record<string, number>;
};

export type TableStateMsg = {
  type: 'state';
  table: any;
  yourSeat: number;
  actorSeat: number | null;
  state: number[];
  view: any;
  valid_actions: number[];
};

export type GameOverMsg = { type: 'game_over'; table: any };


