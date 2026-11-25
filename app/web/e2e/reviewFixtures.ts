import type {
  AccountMe,
  AccountReview,
  HandHistory,
  HandHistoryRow,
  HandReview,
  ReviewDifference,
  ReviewQueue,
} from "../lib/types";

// What the account API serves for one reviewed hand, typed against the
// API's own shapes so a change to them fails the type-check here. The hand
// is a picker's: the pick and call agree with the AI, the first lead cost
// a full point, and the second trick's play gained one.

export const GAME_ID = "11111111-2222-4333-8444-555555555555";

export function me(isAdmin: boolean): AccountMe {
  return {
    player_id: "99999999-2222-4333-8444-555555555555",
    name: "Tester",
    account: {
      username: "tester",
      email: "tester@example.com",
      email_verified: true,
      is_admin: isAdmin,
    },
  };
}

const row: HandHistoryRow = {
  game_id: GAME_ID,
  time_closed: "2026-11-16T20:46:16Z",
  table_name: "Fixture table",
  role: "picker",
  alone: false,
  multiplier: 1,
  score: 2,
  counted_score: 2,
  points_taken: 69,
  points_scope: "team",
  ai_assisted: false,
  abandoned: false,
  opponents: { humans: 0, ai: 4 },
  review: { status: "reviewed", cost: 1.2, gained: 1.4, notable: false },
};

export const hands: HandHistory = { hands: [row], next_cursor: null };

export const handReview: HandReview = {
  hand: row,
  seat: 1,
  called_card: "AH",
  called_under: false,
  dealt_hand: ["QC", "QD", "AD", "AC", "10C", "9H"],
  blind: ["JD", "9C"],
  partner_mode: 1,
  seats: [
    { seat: 1, is_ai: false, is_you: true, name: "Tester", role: "picker" },
    { seat: 2, is_ai: true, is_you: false, name: null, role: "partner" },
    { seat: 3, is_ai: true, is_you: false, name: null, role: "defender" },
    { seat: 4, is_ai: true, is_you: false, name: null, role: "defender" },
    { seat: 5, is_ai: true, is_you: false, name: null, role: "defender" },
  ],
  summary: {
    decisions: 4,
    agreed: 2,
    even: 0,
    slight: 0,
    costly: 1,
    better: 1,
    cost: 1.2,
    gained: 1.4,
    headline_index: 2,
    notable: false,
  },
  decisions: [
    {
      index: 0,
      kind: "pick",
      trick: null,
      hand: ["QC", "QD", "AD", "AC", "10C", "9H"],
      table: [],
      action: "PICK",
      ai_action: "PICK",
      policy: [
        ["PICK", 0.995],
        ["PASS", 0.005],
      ],
      standing: "agreed",
      expected_score: 1.1,
      oracle_score: 3.0,
      chosen_score: null,
      ai_score: null,
      difference: null,
      high_trump: { card: "QC", held: true, unseen: 11 },
      partner_known: false,
    },
    {
      index: 1,
      kind: "call",
      trick: null,
      hand: ["QC", "QD", "AD", "AC", "10C", "9H", "JD", "9C"],
      table: [],
      action: "CALL AH",
      ai_action: "CALL AH",
      policy: [
        ["CALL AH", 1.0],
        ["ALONE", 0.0],
      ],
      standing: "agreed",
      expected_score: 0.5,
      oracle_score: 3.4,
      chosen_score: null,
      ai_score: null,
      difference: null,
      high_trump: { card: "QC", held: true, unseen: 10 },
      partner_known: false,
    },
    {
      index: 2,
      kind: "play",
      trick: 0,
      hand: ["QC", "QD", "AD", "9H", "JD", "9C"],
      table: [],
      action: "PLAY QC",
      ai_action: "PLAY JD",
      policy: [
        ["PLAY JD", 0.99],
        ["PLAY QC", 0.01],
      ],
      standing: "costly",
      expected_score: 0.7,
      oracle_score: 2.5,
      chosen_score: 0.2,
      ai_score: 1.4,
      difference: -1.2,
      high_trump: { card: "QC", held: true, unseen: 10 },
      partner_known: false,
    },
    {
      index: 3,
      kind: "play",
      trick: 1,
      hand: ["QD", "AD", "9H", "JD", "9C"],
      table: [
        [4, "QH"],
        [5, "10D"],
      ],
      action: "PLAY AD",
      ai_action: "PLAY QD",
      policy: [
        ["PLAY QD", 0.8],
        ["PLAY AD", 0.2],
      ],
      standing: "better",
      expected_score: 0.2,
      oracle_score: -1.0,
      chosen_score: 1.6,
      ai_score: 0.2,
      difference: 1.4,
      high_trump: { card: "QD", held: true, unseen: 6 },
      partner_known: false,
    },
  ],
  ai_model: "fixture-model",
  newer_game_id: null,
  older_game_id: null,
};

const costliest: ReviewDifference = {
  game_id: GAME_ID,
  index: 2,
  kind: "play",
  trick: 0,
  led: true,
  action: "PLAY QC",
  ai_action: "PLAY JD",
  ai_probability: 0.99,
  difference: -1.2,
  standing: "costly",
  time_closed: row.time_closed,
  table_name: row.table_name,
};

const best: ReviewDifference = {
  game_id: GAME_ID,
  index: 3,
  kind: "play",
  trick: 1,
  led: false,
  action: "PLAY AD",
  ai_action: "PLAY QD",
  ai_probability: 0.8,
  difference: 1.4,
  standing: "better",
  time_closed: row.time_closed,
  table_name: row.table_name,
};

const noDifferences = { all: [], pick: [], call: [], bury: [], play: [] };

export const totals: AccountReview = {
  hands_reviewed: 1,
  hands_waiting: 0,
  decisions: 4,
  agreed: 2,
  even: 0,
  slight: 0,
  costly: 1,
  better: 1,
  net: 0.2,
  by_kind: [
    { kind: "pick", decisions: 1, agreed: 1, costly: 0, better: 0, net: 0 },
    { kind: "call", decisions: 1, agreed: 1, costly: 0, better: 0, net: 0 },
    { kind: "play", decisions: 2, agreed: 0, costly: 1, better: 1, net: 0.2 },
  ],
  cost_most: { ...noDifferences, all: [costliest], play: [costliest] },
  gained_most: { ...noDifferences, all: [best], play: [best] },
  pick_decisions: 1,
  pick_agreed: 1,
  picked_ai_passes: { count: 0, score: 0, picks: [] },
  passed_ai_picks: { count: 0, score: 0, picks: [] },
};

export const queue: ReviewQueue = {
  paused: false,
  tables_in_play: 0,
  waiting: 3,
  oldest_queued: "2026-11-16T20:40:00Z",
  reviewed_24h: 12,
  average_ms_24h: 1100,
  slowest_ms_24h: 7400,
  reviewed: 160,
  skipped: 2,
  failed: 0,
  total: 165,
  last_reviewed: "2026-11-16T20:46:30Z",
  ai_model: "fixture-model",
  other_model_reviews: 0,
  failures: [],
};
