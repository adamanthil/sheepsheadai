import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseWsMessage } from "./wsMessages";
import { tableState } from "./testing/tableState";
import type { TableStateMsg } from "./types";

describe("parseWsMessage", () => {
  beforeEach(() => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("accepts a state frame, moveSeq and all", () => {
    const frame = tableState({ moveSeq: 7 });
    expect(parseWsMessage(JSON.stringify(frame))).toEqual(frame);
  });

  it("lets unknown fields through, for forward compatibility", () => {
    const frame = { ...tableState(), somethingNew: 1 };
    expect(parseWsMessage(JSON.stringify(frame))).toMatchObject({
      somethingNew: 1,
    });
  });

  it("accepts a state frame from a server without moveSeq", () => {
    const frame: Partial<TableStateMsg> = tableState();
    delete frame.moveSeq;
    expect(parseWsMessage(JSON.stringify(frame))?.type).toBe("state");
  });

  it("drops a state frame missing a required field", () => {
    const frame: Partial<TableStateMsg> = tableState();
    delete frame.valid_actions;
    expect(parseWsMessage(JSON.stringify(frame))).toBeNull();
  });

  it("drops non-JSON and unknown message types", () => {
    expect(parseWsMessage("not json")).toBeNull();
    expect(parseWsMessage(JSON.stringify({ type: "mystery" }))).toBeNull();
  });

  it("parses a turn timer", () => {
    expect(
      parseWsMessage(
        JSON.stringify({ type: "turn_timer", seat: 2, secondsLeft: 30 }),
      ),
    ).toEqual({ type: "turn_timer", seat: 2, secondsLeft: 30 });
  });
});
