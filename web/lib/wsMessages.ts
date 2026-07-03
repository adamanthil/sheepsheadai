// Runtime validation for inbound websocket messages. The schemas are
// deliberately loose (unknown extra fields pass through) — the goal is to
// reject malformed frames at the single parse point instead of crashing a
// component deep in the tree, while staying forward-compatible with new
// fields. Static shapes live in lib/types.ts; a message that validates here
// is cast to those types.
import { z } from "zod";

const looseTable = z.looseObject({
  id: z.string(),
  status: z.string(),
  seats: z.record(z.string(), z.string().nullable()),
});

const chatMessage = z.looseObject({
  id: z.string(),
  type: z.string(),
  body: z.string(),
  author: z.string().nullable().optional(),
  timestamp: z.number(),
});

export const wsMessageSchema = z.discriminatedUnion("type", [
  z.looseObject({
    type: z.literal("state"),
    table: looseTable,
    yourSeat: z.number(),
    actorSeat: z.number().nullable(),
    isHost: z.boolean(),
    state: z.record(z.string(), z.unknown()),
    view: z.looseObject({}),
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
