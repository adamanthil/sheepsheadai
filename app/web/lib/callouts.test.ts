import { describe, expect, it } from "vitest";
import { diffCallouts } from "./callouts";
import { tableState } from "./testing/tableState";

const started = { picker: 2, current_trick_index: 1 };

describe("diffCallouts: trick complete", () => {
  it("fires as a trick is taken, with its cards and winner", () => {
    const prev = tableState({ view: started });
    const next = tableState({
      view: {
        ...started,
        current_trick_index: 2,
        was_trick_just_completed: true,
        last_trick: ["QC", "AC", "KC", "AH", "9C"],
        last_trick_winner: 1,
      },
    });
    expect(diffCallouts(prev, next).trickComplete).toEqual({
      cards: ["QC", "AC", "KC", "AH", "9C"],
      winner: 1,
    });
  });

  it("fires once, not on every state while the flag stays up", () => {
    const view = {
      ...started,
      was_trick_just_completed: true,
      last_trick: ["QC", "AC", "KC", "AH", "9C"],
      last_trick_winner: 1,
    };
    const diff = diffCallouts(tableState({ view }), tableState({ view }));
    expect(diff.trickComplete).toBeNull();
  });
});

describe("diffCallouts: phase", () => {
  it("announces the picker by seat name", () => {
    const diff = diffCallouts(
      tableState(),
      tableState({ view: { picker: 3 } }),
    );
    expect(diff.phase).toEqual({ kind: "picker", pickerName: "Bo" });
  });

  it("announces a leaster", () => {
    const diff = diffCallouts(
      tableState(),
      tableState({ view: { is_leaster: true } }),
    );
    expect(diff.phase).toEqual({ kind: "leaster" });
  });

  it("announces a doublers redeal by the rising pass-out count", () => {
    const diff = diffCallouts(
      tableState({ table: { passOutCount: 0, scoreMultiplier: 1 } }),
      tableState({ table: { passOutCount: 1, scoreMultiplier: 2 } }),
    );
    expect(diff.phase).toEqual({
      kind: "doubler",
      multiplier: 2,
      capped: false,
    });
  });

  it("says a redeal at the stake cap is capped", () => {
    const diff = diffCallouts(
      tableState({ table: { passOutCount: 3, scoreMultiplier: 8 } }),
      tableState({ table: { passOutCount: 4, scoreMultiplier: 8 } }),
    );
    expect(diff.phase).toEqual({
      kind: "doubler",
      multiplier: 8,
      capped: true,
    });
  });

  it("announces going alone", () => {
    const diff = diffCallouts(
      tableState({ view: { picker: 2 } }),
      tableState({ view: { picker: 2, alone: true } }),
    );
    expect(diff.phase).toEqual({ kind: "alone", pickerName: "Ann" });
  });

  it("announces the called card", () => {
    const diff = diffCallouts(
      tableState({ view: { picker: 2 } }),
      tableState({
        view: {
          picker: 2,
          called_card: "AH",
          called_card_display: "Ace of Hearts",
          called_under: true,
        },
      }),
    );
    expect(diff.phase).toEqual({
      kind: "call",
      pickerName: "Ann",
      cardDisplay: "Ace of Hearts",
      under: true,
    });
  });

  it("is quiet when nothing changed", () => {
    const state = tableState({ view: { picker: 2 } });
    expect(diffCallouts(state, state)).toEqual({
      trickComplete: null,
      phase: null,
    });
  });
});
