import { describe, expect, it } from "vitest";
import { tableState } from "../../../../lib/testing/tableState";
import {
  derivePhase,
  getSeatRole,
  getYourRole,
  interludeMode,
  playStarted,
} from "./phase";

describe("derivePhase", () => {
  it("is the pick before anyone picks", () => {
    expect(derivePhase(tableState()).phase).toBe("pick");
  });

  it("is the interlude between the pick and the first card", () => {
    expect(derivePhase(tableState({ view: { picker: 2 } })).phase).toBe(
      "interlude",
    );
  });

  it("is the interlude for a leaster before play", () => {
    const { phase, isLeaster } = derivePhase(
      tableState({ view: { is_leaster: true } }),
    );
    expect(phase).toBe("interlude");
    expect(isLeaster).toBe(true);
  });

  it("is play once the server says so or a card is down", () => {
    expect(derivePhase(tableState({ state: { play_started: 1 } })).phase).toBe(
      "play",
    );
    expect(
      derivePhase(
        tableState({
          view: { picker: 2, current_trick: ["QC", "", "", "", ""] },
        }),
      ).phase,
    ).toBe("play");
  });

  it("is done when the hand is", () => {
    expect(derivePhase(tableState({ view: { is_done: true } })).phase).toBe(
      "done",
    );
  });
});

describe("playStarted", () => {
  it("counts a later trick as started", () => {
    expect(playStarted(tableState({ view: { current_trick_index: 3 } }))).toBe(
      true,
    );
  });

  it("is false during the pick", () => {
    expect(playStarted(tableState())).toBe(false);
  });
});

describe("getSeatRole", () => {
  it("marks the seat deciding and the seats that passed during the pick", () => {
    const state = tableState({ actorSeat: 3 });
    expect([1, 2, 3, 4, 5].map((s) => getSeatRole(state, s, false))).toEqual([
      "PASS",
      "PASS",
      "PENDING",
      null,
      null,
    ]);
  });

  it("marks the picker and the passes before it once picked", () => {
    const state = tableState({ view: { picker: 3 }, actorSeat: 3 });
    expect([1, 2, 3, 4, 5].map((s) => getSeatRole(state, s, false))).toEqual([
      "PASS",
      "PASS",
      "PICKER",
      null,
      null,
    ]);
  });

  it("marks only picker and partner in play", () => {
    const state = tableState({ view: { picker: 3, partner: 5 } });
    expect([1, 2, 3, 4, 5].map((s) => getSeatRole(state, s, true))).toEqual([
      null,
      null,
      "PICKER",
      null,
      "PARTNER",
    ]);
  });
});

describe("interludeMode", () => {
  const mode = (...labels: string[]) => interludeMode(new Set(labels));

  it("reads the mode off the valid actions", () => {
    expect(mode("BURY QC", "BURY JD")).toBe("bury");
    expect(mode("UNDER 7D", "UNDER 9C")).toBe("under");
    expect(mode("CALL AC", "ALONE")).toBe("call");
    expect(mode("JD PARTNER")).toBe("call");
  });

  it("does not take a called-under ace or a played under for under mode", () => {
    expect(mode("CALL AH UNDER")).toBe("call");
    expect(mode("PLAY UNDER")).toBe("waiting");
  });

  it("is waiting with nothing to do", () => {
    expect(mode()).toBe("waiting");
  });
});

describe("getYourRole", () => {
  it("is picker for the picker", () => {
    expect(getYourRole(tableState({ view: { picker: 1 } }))).toBe("PICKER");
  });

  it("knows you are partner from the called card in hand", () => {
    const state = tableState({
      view: { picker: 2, called_card: "AH", hand: ["AH", "QC"] },
    });
    expect(getYourRole(state)).toBe("PARTNER");
  });

  it("knows you are partner from the jack of diamonds in JD mode", () => {
    const state = tableState({
      view: { picker: 2, hand: ["JD", "QC"] },
      table: {
        rules: {
          allPassMode: "leasters",
          doubleOnTheBump: true,
          partnerMode: 0,
        },
      },
    });
    expect(getYourRole(state)).toBe("PARTNER");
  });

  it("is nothing for a defender, or in a leaster", () => {
    expect(
      getYourRole(
        tableState({ view: { picker: 2, called_card: "AH", hand: ["QC"] } }),
      ),
    ).toBeNull();
    expect(getYourRole(tableState({ view: { is_leaster: true } }))).toBeNull();
  });
});
