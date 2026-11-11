import { describe, expect, it } from "vitest";
import { tableState } from "../../../../lib/testing/tableState";
import {
  buildCallOptions,
  computeContext,
  computeIsYourTurn,
  computeValidActionStrings,
  formatClock,
  handsInCurrentBlock,
  resultLine,
  scoreBlocks,
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
      tableState({ view: { picker: 3 }, table: { scoreMultiplier: 4 } }),
      "play",
      false,
    );
    expect(text(ctx.parts)).toBe("Bo picked · 4× Doubler");
  });

  it("says where a leaster's blind goes until that trick is cleared", () => {
    const first = computeContext(
      tableState({ view: { is_leaster: true, current_trick_index: 0 } }),
      "play",
      false,
    );
    expect(text(first.parts)).toBe("Leaster · Blind to 1st trick");
    const held = computeContext(
      tableState({ view: { is_leaster: true, current_trick_index: 1 } }),
      "play",
      false,
      true,
    );
    expect(text(held.parts)).toBe("Leaster · Blind to 1st trick");
    const later = computeContext(
      tableState({ view: { is_leaster: true, current_trick_index: 1 } }),
      "play",
      false,
    );
    expect(text(later.parts)).toBe("Leaster");
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

describe("scoreBlocks", () => {
  const cols = (...ids: string[]) => ids.map((id) => ({ id, name: id }));
  // Two-seat hands keep the fixtures short; the columns are what count.
  const hand = (
    segment: number,
    picker: string | null,
    scores: Array<[id: string, column: string, score: number]>,
  ) => ({
    segment,
    picker,
    bySeat: Object.fromEntries(
      scores.map(([id, column, score], i) => [i + 1, { id, column, score }]),
    ),
  });
  const table = tableState({
    table: {
      scoreSegments: [
        { columns: cols("ann", "ai1") },
        { columns: cols("bob", "ann") },
      ],
      resultsHistory: [
        hand(0, "ann", [
          ["ann", "ann", 2],
          ["ai1", "ai1", -2],
        ]),
        // The AI stood in for Ann, on her column, and was set.
        hand(0, "stand-in", [
          ["stand-in", "ann", -4],
          ["ai1", "ai1", 4],
        ]),
        // Bob was dealt in: a new block from 0.
        hand(1, null, [
          ["bob", "bob", 3],
          ["ann", "ann", -3],
        ]),
      ],
    },
  }).table;

  it("restarts running totals in each block of players", () => {
    const blocks = scoreBlocks(table);

    expect(blocks.map((b) => b.columns.map((c) => c.id))).toEqual([
      ["ann", "ai1"],
      ["bob", "ann"],
    ]);
    expect(blocks.map((b) => b.rows.map((r) => r.cells))).toEqual([
      [
        [
          { delta: 2, total: 2 },
          { delta: -2, total: -2 },
        ],
        [
          { delta: -4, total: -2 },
          { delta: 4, total: 2 },
        ],
      ],
      [
        [
          { delta: 3, total: 3 },
          { delta: -3, total: -3 },
        ],
      ],
    ]);
  });

  it("marks the hands the picker was set", () => {
    expect(scoreBlocks(table).flatMap((b) => b.rows.map((r) => r.set))).toEqual(
      [false, true, false],
    );
  });

  it("counts only the current block's hands for the running scores", () => {
    expect(handsInCurrentBlock(table)).toBe(1);
  });
});

describe("resultLine", () => {
  const table = tableState({
    table: {
      seats: { "1": "Ann", "2": "Bob", "3": "Cy", "4": "Di", "5": "Ed" },
    },
  }).table;

  it("gives the team split and your score", () => {
    const final = {
      mode: "standard" as const,
      picker: 2,
      picker_score: 58,
      defender_score: 62,
      scores: [1, -4, 1, 1, 1],
    };
    expect(resultLine(final, table, 1)).toBe(
      "Picker 58 – 62 Defenders · You +1",
    );
  });

  it("names a leaster's winner and leaves a spectator's score out", () => {
    const final = {
      mode: "leaster" as const,
      winner: 3,
      points_taken: [30, 41, 0, 25, 24],
      scores: [-1, -1, 4, -1, -1],
    };
    expect(resultLine(final, table, null)).toBe("Cy wins the leaster with 0");
  });
});
