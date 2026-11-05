import { describe, expect, it } from "vitest";
import { applyPendingMove, isPredicted } from "./optimistic";
import { tableState } from "./testing/tableState";

const playing = (seat: number, trick: string[]) =>
  tableState({
    yourSeat: seat,
    actorSeat: seat,
    valid_actions: [1, 2, 3],
    view: { picker: 3, current_trick: trick, current_trick_index: 2 },
  });

describe("isPredicted", () => {
  it.each([
    "PLAY QC",
    "PLAY UNDER",
    "PASS",
    "PICK",
    "CALL AC",
    "CALL AH UNDER",
    "ALONE",
    "JD PARTNER",
  ])("draws %s before the server", (label) =>
    expect(isPredicted(label)).toBe(true),
  );

  it.each(["BURY QC", "UNDER 7D"])("leaves staged %s to the server", (label) =>
    expect(isPredicted(label)).toBe(false),
  );
});

describe("applyPendingMove", () => {
  it("moves a played card from the hand to your place in the trick", () => {
    const before = playing(2, ["QC", "", "", "", ""]);
    const after = applyPendingMove(before, "PLAY AH");

    expect(after.view.hand).not.toContain("AH");
    expect(after.view.hand).toHaveLength(5);
    expect(after.view.current_trick).toEqual(["QC", "AH", "", "", ""]);
    expect(after.actorSeat).toBe(3);
    expect(after.valid_actions).toEqual([]);
  });

  it("leaves the server's state untouched", () => {
    const before = playing(2, ["QC", "", "", "", ""]);
    const snapshot = structuredClone(before);
    applyPendingMove(before, "PLAY AH");
    expect(before).toEqual(snapshot);
  });

  it("passes the turn from seat 5 round to seat 1", () => {
    const after = applyPendingMove(
      playing(5, ["", "", "QC", "AC", ""]),
      "PLAY AH",
    );
    expect(after.actorSeat).toBe(1);
  });

  it("names no next seat when the card completes the trick", () => {
    const after = applyPendingMove(
      playing(4, ["QC", "AC", "KC", "", "9C"]),
      "PLAY AH",
    );
    expect(after.view.current_trick).toEqual(["QC", "AC", "KC", "AH", "9C"]);
    expect(after.actorSeat).toBeNull();
  });

  it("passes the bid to the next seat", () => {
    const after = applyPendingMove(
      tableState({ yourSeat: 2, actorSeat: 2, valid_actions: [1, 2] }),
      "PASS",
    );
    expect(after.actorSeat).toBe(3);
    expect(after.valid_actions).toEqual([]);
  });

  it("keeps the last seat's pass on that seat until the server answers", () => {
    const after = applyPendingMove(
      tableState({ yourSeat: 5, actorSeat: 5, valid_actions: [1, 2] }),
      "PASS",
    );
    expect(after.actorSeat).toBe(5);
  });

  it("makes you the picker on PICK", () => {
    const after = applyPendingMove(
      tableState({ yourSeat: 2, actorSeat: 2, valid_actions: [1, 2] }),
      "PICK",
    );
    expect(after.view.picker).toBe(2);
    expect(after.actorSeat).toBeNull();
    expect(after.view.hand).toHaveLength(6);
  });

  it("only clears the controls for a call", () => {
    const before = tableState({ view: { picker: 1 }, valid_actions: [7, 8] });
    const after = applyPendingMove(before, "CALL AC");
    expect(after.valid_actions).toEqual([]);
    expect(after.actorSeat).toBeNull();
    expect(after.view).toBe(before.view);
  });

  it("keeps the under card in hand for PLAY UNDER", () => {
    const before = playing(1, ["", "", "", "", ""]);
    const after = applyPendingMove(before, "PLAY UNDER");
    expect(after.view).toBe(before.view);
    expect(after.actorSeat).toBeNull();
  });

  it("changes nothing for a spectator", () => {
    const before = tableState({ yourSeat: null });
    expect(applyPendingMove(before, "PASS")).toBe(before);
  });
});
