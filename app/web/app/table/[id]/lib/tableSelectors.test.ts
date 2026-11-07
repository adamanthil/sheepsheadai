import { describe, expect, it } from "vitest";
import { tableState } from "../../../../lib/testing/tableState";
import {
  buildCallOptions,
  computeContext,
  computeIsYourTurn,
  computeValidActionStrings,
  formatClock,
  stakeBadgeText,
} from "./tableSelectors";

const lookup: Record<string, string> = {
  "1": "PICK",
  "2": "PASS",
  "10": "CALL AC",
  "11": "CALL AH UNDER",
  "12": "ALONE",
  "13": "JD PARTNER",
};

describe("buildCallOptions", () => {
  it("describes each call the picker can make", () => {
    expect(buildCallOptions([10, 11, 12, 13, 1], lookup)).toEqual([
      { actionId: 10, label: "CALL AC", code: "AC", display: "Ace of Clubs" },
      {
        actionId: 11,
        label: "CALL AH UNDER",
        code: "AH",
        display: "Ace of Hearts (under)",
      },
      { actionId: 12, label: "ALONE", code: null, display: "Go alone" },
      {
        actionId: 13,
        label: "JD PARTNER",
        code: "JD",
        display: "Jack of Diamonds",
      },
    ]);
  });
});

describe("computeIsYourTurn", () => {
  it("is your turn when you are the actor", () => {
    expect(computeIsYourTurn(tableState({ actorSeat: 1 }))).toBe(true);
    expect(computeIsYourTurn(tableState({ actorSeat: 2 }))).toBe(false);
  });

  it("is never a spectator's turn", () => {
    expect(
      computeIsYourTurn(tableState({ yourSeat: null, actorSeat: null })),
    ).toBe(false);
  });
});

describe("computeValidActionStrings", () => {
  it("maps action ids to their labels", () => {
    const valid = computeValidActionStrings(
      tableState({ valid_actions: [1, 2, 99] }),
      lookup,
    );
    expect([...valid]).toEqual(["PICK", "PASS"]);
  });
});

describe("computeContext", () => {
  const text = (parts: ReturnType<typeof computeContext>["parts"]) =>
    parts.map((p) => (typeof p === "string" ? p : `[${p.card}]`)).join("");

  it("names the called card, faded once the partner is known", () => {
    const hidden = computeContext(
      tableState({
        view: { picker: 2, called_card: "AH", called_under: true },
      }),
      "play",
      false,
    );
    expect(text(hidden.parts)).toBe("[AH] called under");
    expect(hidden.faded).toBe(false);

    const revealed = computeContext(
      tableState({ view: { picker: 2, called_card: "AH", partner: 4 } }),
      "play",
      false,
    );
    expect(revealed.faded).toBe(true);
  });

  it("says who went alone, in the second person for you", () => {
    const you = computeContext(
      tableState({ view: { picker: 1, alone: true } }),
      "play",
      false,
    );
    expect(text(you.parts)).toBe("You go alone");
    const them = computeContext(
      tableState({ view: { picker: 3, alone: true } }),
      "play",
      false,
    );
    expect(text(them.parts)).toBe("Bo goes alone");
  });

  it("adds the stake after a doubler", () => {
    const ctx = computeContext(
      tableState({ view: { is_leaster: true }, table: { scoreMultiplier: 4 } }),
      "play",
      false,
    );
    expect(text(ctx.parts)).toBe("Leaster · 4× Doubler");
  });

  it("gives your pick position from the dealer while picking", () => {
    const first = computeContext(tableState({ yourSeat: 1 }), "pick", false);
    expect(text(first.parts)).toBe("1st Position");
    const dealer = computeContext(
      tableState({ yourSeat: 5, table: { scoreMultiplier: 2 } }),
      "pick",
      false,
    );
    expect(text(dealer.parts)).toBe("5th Position · 2× Doubler");
  });

  it("says only that while the last trick is open", () => {
    expect(computeContext(tableState(), "play", true).parts).toEqual([
      "Last trick",
    ]);
  });
});

describe("badges and clock", () => {
  it("badges only a raised stake", () => {
    expect(stakeBadgeText(1)).toBeNull();
    expect(stakeBadgeText(2)).toBe("2× Doubler");
  });

  it("formats the turn clock", () => {
    expect(formatClock(120)).toBe("2:00");
    expect(formatClock(5)).toBe("0:05");
  });
});
