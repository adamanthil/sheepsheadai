import { describe, expect, it } from "vitest";
import {
  actionCodes,
  decisionCounts,
  decisionHeadline,
  decisionKicker,
  decisionLabel,
  decisionNote,
  differenceLines,
  handHeadline,
  net,
  netTone,
  partnerLine,
  shortAction,
  signed,
  tone,
} from "./review";
import type { DecisionReview, HandReview } from "./types";

function decision(over: Partial<DecisionReview>): DecisionReview {
  return {
    index: 0,
    kind: "play",
    trick: 3,
    hand: ["JC", "10D", "KC"],
    table: [
      [5, "7H"],
      [1, "JD"],
      [2, "7S"],
    ],
    action: "PLAY JC",
    ai_action: "PLAY KC",
    policy: [
      ["PLAY KC", 0.99],
      ["PLAY JC", 0.01],
    ],
    standing: "costly",
    expected_score: 3.0,
    oracle_score: 1.9,
    chosen_score: 1.8,
    ai_score: 3.7,
    difference: -2.0,
    high_trump: { card: "JC", held: true, unseen: 2 },
    partner_known: false,
    ...over,
  };
}

function review(
  decisions: DecisionReview[],
  summary: Partial<NonNullable<HandReview["summary"]>>,
  status: "queued" | "reviewed" | "skipped" | "failed" = "reviewed",
): HandReview {
  return {
    hand: { review: { status, cost: 0, gained: 0, notable: false } },
    summary:
      status === "reviewed"
        ? {
            decisions: decisions.length,
            agreed: 0,
            even: 0,
            slight: 0,
            costly: 0,
            better: 0,
            cost: 0,
            gained: 0,
            headline_index: null,
            notable: false,
            ...summary,
          }
        : null,
    decisions,
  } as HandReview;
}

describe("signed", () => {
  it("uses a real minus sign and never shows a signed zero", () => {
    expect(signed(1.24)).toBe("+1.2");
    expect(signed(-0.7)).toBe("−0.7");
    expect(signed(0)).toBe("0.0");
    expect(signed(-0.04)).toBe("0.0");
  });
});

describe("actionCodes", () => {
  it("keeps card codes for colouring", () => {
    expect(actionCodes("PLAY JH")).toBe("JH");
    expect(actionCodes("BURY 8S AH")).toBe("8S AH");
    expect(actionCodes("CALL AS UNDER")).toBe("AS under");
    expect(actionCodes("JD PARTNER")).toBe("JD partner");
    expect(actionCodes("PASS")).toBe("Pass");
  });
});

describe("shortAction", () => {
  it("reduces an action to its cards", () => {
    expect(shortAction("PICK")).toBe("Pick");
    expect(shortAction("PLAY 10D")).toBe("10♦");
    expect(shortAction("BURY 8S AH")).toBe("8♠ A♥");
    expect(shortAction("CALL AC")).toBe("A♣");
    expect(shortAction("CALL AS UNDER")).toBe("A♠ under");
    expect(shortAction("JD PARTNER")).toBe("J♦ partner");
  });
});

describe("decisionHeadline", () => {
  it("says what each did, in their own tense", () => {
    expect(decisionHeadline(decision({}))).toBe(
      "You played the J♣. The AI plays the K♣.",
    );
  });

  it("says led for the first card of a trick", () => {
    expect(
      decisionHeadline(
        decision({ table: [], action: "PLAY 8D", ai_action: "PLAY JD" }),
      ),
    ).toBe("You led the 8♦. The AI leads the J♦.");
  });

  it("covers the choices before the first card", () => {
    expect(
      decisionHeadline(
        decision({
          kind: "pick",
          table: [],
          action: "PICK",
          ai_action: "PASS",
        }),
      ),
    ).toBe("You picked. The AI passes.");
    expect(
      decisionHeadline(
        decision({
          kind: "bury",
          table: [],
          action: "BURY 8S AH",
          ai_action: "BURY 10C AH",
        }),
      ),
    ).toBe("You buried 8♠ and A♥. The AI buries 10♣ and A♥.");
    expect(
      decisionHeadline(
        decision({
          kind: "call",
          table: [],
          action: "CALL AC",
          ai_action: "CALL AS",
        }),
      ),
    ).toBe("You called the A♣. The AI calls the A♠.");
  });

  it("says so when they agreed", () => {
    expect(
      decisionHeadline(
        decision({
          kind: "pick",
          table: [],
          action: "PICK",
          ai_action: "PICK",
        }),
      ),
    ).toBe("You picked. So does the AI.");
  });
});

describe("labels and notes", () => {
  it("numbers tricks from one", () => {
    expect(decisionLabel(decision({}))).toBe("Trick 4");
    expect(decisionLabel(decision({}), true)).toBe("T4");
    expect(decisionLabel(decision({ kind: "bury", trick: null }))).toBe("Bury");
  });

  it("places the decision in the hand", () => {
    expect(decisionKicker(decision({}), 3)).toBe(
      "Trick 4 · 7♥ led · fourth to play",
    );
    expect(decisionKicker(decision({ table: [] }), 3)).toBe(
      "Trick 4 · your lead",
    );
    expect(decisionKicker(decision({ kind: "pick", table: [] }), 3)).toBe(
      "Pick or pass · seat 3 · 2 passed before you",
    );
    expect(decisionKicker(decision({ kind: "pick", table: [] }), 1)).toBe(
      "Pick or pass · seat 1 · first to speak",
    );
  });

  it("names the AI's choice only where it differs", () => {
    expect(decisionNote(decision({}))).toBe("AI: K♣");
    expect(decisionNote(decision({ standing: "even" }))).toBe("Even: K♣");
    expect(decisionNote(decision({ standing: "agreed" }))).toBe("Agreed");
    expect(decisionNote(decision({ standing: "forced" }))).toBe("Forced");
  });

  it("words a difference for a list", () => {
    expect(
      differenceLines({ action: "PLAY 8D", ai_action: "PLAY JD", led: true }),
    ).toEqual({ yours: "Led the 8♦", ai: "AI leads the J♦" });
  });
});

describe("handHeadline", () => {
  it("names the costliest decision", () => {
    const hand = review([decision({ index: 6 })], {
      costly: 1,
      headline_index: 6,
    });
    expect(handHeadline(hand)).toEqual({
      kicker: "Costliest decision · Trick 4 · −2.0 points of expected score",
      text: "You played the J♣. The AI plays the K♣.",
      index: 6,
    });
  });

  it("names the best decision when nothing cost a full point", () => {
    const hand = review(
      [decision({ index: 7, trick: 4, standing: "better", difference: 1.4 })],
      { better: 1, headline_index: 7 },
    );
    expect(handHeadline(hand).kicker).toBe(
      "Best decision · Trick 5 · +1.4 points of expected score",
    );
  });

  it("says so plainly when every difference was small", () => {
    expect(handHeadline(review([], { even: 3 })).text).toBe(
      "Three different choices, all about even.",
    );
    expect(handHeadline(review([], { even: 1 })).text).toBe(
      "One different choice, about even.",
    );
  });

  it("does not call a small difference even", () => {
    expect(handHeadline(review([], { even: 2, slight: 1 })).text).toBe(
      "Three different choices, none worth a full point.",
    );
    expect(handHeadline(review([], { slight: 1 })).text).toBe(
      "One different choice, worth less than a point.",
    );
  });

  it("says so when they agreed throughout", () => {
    expect(handHeadline(review([], { agreed: 4 }))).toEqual({
      kicker: null,
      text: "You and the AI agreed on every decision.",
      index: null,
    });
  });

  it("explains a hand with no review", () => {
    expect(handHeadline(review([], {}, "queued")).text).toBe(
      "This hand is waiting for its review.",
    );
    expect(handHeadline(review([], {}, "failed")).text).toBe(
      "This hand could not be reviewed.",
    );
  });
});

describe("totals", () => {
  it("counts decisions by how they compare", () => {
    expect(
      decisionCounts({
        decisions: 8,
        agreed: 4,
        even: 1,
        slight: 1,
        costly: 2,
        better: 1,
        cost: 2.7,
        gained: 0.9,
        headline_index: 6,
        notable: false,
      }),
    ).toBe(
      "8 decisions: 4 agreed, 1 about even, 1 small difference, 2 cost you, 1 gained",
    );
  });

  it("nets cost against gain to one decimal", () => {
    expect(net(2.7, 0.9)).toBe(-1.8);
    expect(net(0.1, 0.1)).toBe(0);
  });

  it("colours a net only for a notable hand", () => {
    expect(netTone(-1.3, true)).toBe("cost");
    expect(netTone(2.1, true)).toBe("gain");
    expect(netTone(-1.4, false)).toBe("plain");
    expect(netTone(0.5, false)).toBe("plain");
  });

  it("colours only a full point or more", () => {
    expect(tone(-1)).toBe("cost");
    expect(tone(1)).toBe("gain");
    expect(tone(-0.9)).toBe("plain");
    expect(tone(null)).toBe("plain");
  });
});

describe("partnerLine", () => {
  const seats = (you: HandReview["seats"][number]["role"]) =>
    [1, 2, 3, 4, 5].map((seat) => ({
      seat,
      is_ai: seat !== 3,
      is_you: seat === 3,
      name: null,
      role:
        seat === 3
          ? you
          : seat === 1 && you !== "leaster"
            ? "picker"
            : "defender",
    })) as HandReview["seats"];
  const hand = (alone = false) => ({ alone }) as HandReview["hand"];
  const called = {
    hand: hand(),
    seat: 3,
    seats: seats("defender"),
    called_card: "AH",
    called_under: false,
    partner_mode: 1,
  };

  it("is absent before the call and in a leaster", () => {
    expect(partnerLine(called, decision({ kind: "pick" }))).toBeNull();
    expect(partnerLine(called, decision({ kind: "call" }))).toBeNull();
    expect(
      partnerLine({ ...called, seats: seats("leaster") }, decision({})),
    ).toBeNull();
  });

  it("names the called ace and whether the partner is known", () => {
    expect(partnerLine(called, decision({}))).toEqual({
      label: "Called",
      card: "AH",
      under: false,
      held: false,
      note: "partner not yet known",
    });
    expect(
      partnerLine(
        { ...called, called_under: true },
        decision({ partner_known: true }),
      ),
    ).toMatchObject({ under: true, note: "partner known" });
  });

  it("tells the partner it is in their hand, then that they played it", () => {
    const partner = { ...called, seats: seats("partner") };
    expect(
      partnerLine(partner, decision({ hand: ["AH", "QC"] })),
    ).toMatchObject({ held: true, note: "in your hand" });
    expect(
      partnerLine(partner, decision({ partner_known: true })),
    ).toMatchObject({ held: false, note: "you played it" });
  });

  it("shows the jack of diamonds, or that the picker is alone", () => {
    expect(
      partnerLine(
        { ...called, partner_mode: 0, called_card: null },
        decision({}),
      ),
    ).toMatchObject({ label: "Partner", card: "JD" });
    expect(
      partnerLine({ ...called, hand: hand(true) }, decision({})),
    ).toMatchObject({ card: null, note: "none, the picker is alone" });
  });
});
