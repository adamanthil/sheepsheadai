// Runtime validation for inbound websocket messages. The schemas are
// deliberately loose (unknown extra fields pass through) — the goal is to
// reject malformed frames at the single parse point instead of crashing a
// component deep in the tree, while staying forward-compatible with new
// fields. Shapes here mirror the static types in lib/types.ts field-for-
// field, so a validated message is already structurally assignable to those
// types — no unsafe cast is needed to hand it off.
import { z } from "zod";

const rules = z.looseObject({
  doubleOnTheBump: z.boolean(),
  partnerMode: z.union([z.literal(0), z.literal(1)]),
});

const looseTable = z.looseObject({
  id: z.string(),
  status: z.enum(["open", "playing", "finished"]),
  name: z.string(),
  host: z.string().nullable(),
  fillWithAI: z.boolean(),
  initialNames: z.record(z.string(), z.string()),
  initialSeatOrder: z.array(z.string()),
  resultsHistory: z.array(z.looseObject({})),
  rules,
  runningBySeat: z.record(z.string(), z.number()),
  seatIsAI: z.record(z.string(), z.boolean()),
  seatOccupants: z.record(z.string(), z.string().nullable()),
  seats: z.record(z.string(), z.string().nullable()),
});

const chatMessage = z.looseObject({
  id: z.string(),
  table_id: z.string(),
  type: z.enum(["player", "system"]),
  author: z.string().nullable(),
  body: z.string(),
  timestamp: z.number(),
});

const finalState = z.looseObject({
  mode: z.enum(["leaster", "standard"]).optional(),
  winner: z.number().optional(),
  picker: z.number().optional(),
  partner: z.number().optional(),
  picker_score: z.number().optional(),
  defender_score: z.number().optional(),
  scores: z.array(z.number()).optional(),
  points_taken: z.array(z.number()).optional(),
});

const gameView = z.looseObject({
  hand: z.array(z.string()),
  picker: z.number(),
  partner: z.number(),
  is_leaster: z.boolean(),
  current_trick: z.array(z.string()),
  current_trick_index: z.number(),
  last_trick: z.array(z.string()).nullable(),
  last_trick_winner: z.number().nullable(),
  last_trick_points: z.number().optional(),
  was_trick_just_completed: z.boolean(),
  is_done: z.boolean(),
  final: finalState.nullable().optional(),
  called_card: z.string().nullable(),
  called_card_display: z.string().nullable(),
  called_under: z.boolean(),
  alone: z.boolean(),
});

const tableState = z.looseObject({
  play_started: z.number().optional(),
  is_leaster: z.number().optional(),
  current_trick: z.number().optional(),
  picker_rel: z.number().optional(),
  partner_rel: z.number().optional(),
});

export const wsMessageSchema = z.discriminatedUnion("type", [
  z.looseObject({
    type: z.literal("state"),
    table: looseTable,
    yourSeat: z.number(),
    actorSeat: z.number().nullable(),
    isHost: z.boolean(),
    state: tableState,
    view: gameView,
    valid_actions: z.array(z.number()),
  }),
  z.looseObject({
    type: z.literal("table_update"),
    table: looseTable,
    isHost: z.boolean().optional(),
  }),
  z.looseObject({
    type: z.literal("lobby_event"),
    message: z.string(),
    table: looseTable.optional(),
  }),
  z.looseObject({
    type: z.literal("table_closed"),
    reason: z.string().optional(),
    tableId: z.string().optional(),
  }),
  z.looseObject({
    type: z.literal("chat:init"),
    messages: z.array(chatMessage),
  }),
  z.looseObject({
    type: z.literal("chat:append"),
    message: chatMessage,
  }),
  z.looseObject({ type: z.literal("server_restart") }),
]);

export type WsMessage = z.infer<typeof wsMessageSchema>;

/** Parse a raw frame. Returns null (and warns) for malformed or unknown
 * messages — the game must never crash on a bad frame. */
export function parseWsMessage(raw: string): WsMessage | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    console.warn("ws: non-JSON frame dropped");
    return null;
  }
  const result = wsMessageSchema.safeParse(data);
  if (!result.success) {
    console.warn("ws: frame failed validation, dropped", {
      type: (data as { type?: string })?.type,
      issues: result.error.issues.slice(0, 3),
    });
    return null;
  }
  return result.data;
}
