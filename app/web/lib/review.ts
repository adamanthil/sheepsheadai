/** Wording for the AI review pages: how a decision, and a whole hand, is
 * put into words. Pure functions over the review API's shapes.
 *
 * Vocabulary, the same on every screen: a decision is "agreed", "about
 * even", a "small difference", "cost you" or "gained"; the other party is
 * "the AI". */
import { parseCard, suitSymbol } from "./ds/cardUtils";
import { signed } from "./format";
import type { DecisionReview, HandReview } from "./types";

/** "J♣" for "JC". */
export function cardLabel(code: string): string {
  const { rank, suit } = parseCard(code);
  return `${rank}${suitSymbol(suit)}`;
}

function cardsOf(action: string): string[] {
  return action
    .split(" ")
    .filter((word) => /^(10|[AKQJ987])[CSHD]$/.test(word));
}

/** The action alone, as short as it goes, with its cards still as codes
 * ("Pick", "AC", "8S AH") for CardText to draw in their suit colours. */
export function actionCodes(action: string): string {
  if (action === "PICK") return "Pick";
  if (action === "PASS") return "Pass";
  if (action === "ALONE") return "Alone";
  if (action === "JD PARTNER") return "JD partner";
  if (action === "PLAY UNDER") return "Under";
  const cards = cardsOf(action).join(" ");
  return action.startsWith("CALL ") && action.endsWith(" UNDER")
    ? `${cards} under`
    : cards;
}

/** The same in plain text: "Pick", "A♣", "8♠ A♥". */
export function shortAction(action: string): string {
  return actionCodes(action)
    .split(" ")
    .map((word) => (cardsOf(word).length ? cardLabel(word) : word))
    .join(" ");
}

function list(cards: string[]): string {
  return cards.map(cardLabel).join(" and ");
}

/** What was done, as a sentence without its subject: "played the J♣" for
 * the player ("you"), "plays the K♣" for the AI. */
function verbPhrase(action: string, led: boolean, ai: boolean): string {
  const cards = cardsOf(action);
  if (action === "PICK") return ai ? "picks" : "picked";
  if (action === "PASS") return ai ? "passes" : "passed";
  if (action === "ALONE") return ai ? "goes alone" : "went alone";
  if (action === "JD PARTNER")
    return ai ? "takes the J♦ as partner" : "took the J♦ as partner";
  if (action.startsWith("CALL "))
    return `${ai ? "calls" : "called"} the ${list(cards)}`;
  if (action.startsWith("UNDER "))
    return `${ai ? "puts" : "put"} the ${list(cards)} under`;
  if (action.startsWith("BURY "))
    return `${ai ? "buries" : "buried"} ${list(cards)}`;
  if (action === "PLAY UNDER")
    return ai ? "plays the under" : "played the under";
  if (led) return `${ai ? "leads" : "led"} the ${list(cards)}`;
  return `${ai ? "plays" : "played"} the ${list(cards)}`;
}

function led(decision: Pick<DecisionReview, "kind" | "table">): boolean {
  return decision.kind === "play" && decision.table.length === 0;
}

/** "You played the J♣. The AI plays the K♣.", or "You picked. So does the
 * AI." where they agreed. */
export function decisionHeadline(
  decision: Pick<DecisionReview, "kind" | "table" | "action" | "ai_action">,
): string {
  const first = led(decision);
  const you = `You ${verbPhrase(decision.action, first, false)}.`;
  if (decision.action === decision.ai_action) return `${you} So does the AI.`;
  return `${you} The AI ${verbPhrase(decision.ai_action, first, true)}.`;
}

/** "Led the 8♦" and "AI leads the J♦", for a list of differences. */
export function differenceLines(item: {
  action: string;
  ai_action: string;
  led: boolean;
}): { yours: string; ai: string } {
  const yours = verbPhrase(item.action, item.led, false);
  return {
    yours: yours.charAt(0).toUpperCase() + yours.slice(1),
    ai: `AI ${verbPhrase(item.ai_action, item.led, true)}`,
  };
}

/** "Pick", "Call", "Bury", "Trick 4"; "T4" in brief. */
export function decisionLabel(
  decision: Pick<DecisionReview, "kind" | "trick">,
  brief = false,
): string {
  if (decision.kind === "play") {
    const number = (decision.trick ?? 0) + 1;
    return brief ? `T${number}` : `Trick ${number}`;
  }
  return { pick: "Pick", call: "Call", under: "Under", bury: "Bury" }[
    decision.kind
  ];
}

const ORDINAL = ["first", "second", "third", "fourth", "fifth"];

/** The line above a decision's headline: where in the hand it was made. */
export function decisionKicker(decision: DecisionReview, seat: number): string {
  if (decision.kind === "pick") {
    const before = seat - 1;
    const passed =
      before === 0 ? "first to speak" : `${before} passed before you`;
    return `Pick or pass · seat ${seat} · ${passed}`;
  }
  if (decision.kind === "call") return "Partner call";
  if (decision.kind === "under") return "Under card";
  if (decision.kind === "bury") return "Bury · two cards, chosen together";
  const trick = `Trick ${(decision.trick ?? 0) + 1}`;
  if (decision.table.length === 0) return `${trick} · your lead`;
  const lead = cardLabel(decision.table[0][1]);
  return `${trick} · ${lead} led · ${ORDINAL[decision.table.length]} to play`;
}

/** Beside a decision in a list: "Agreed", "Forced", "Even: A♠", "AI: K♣". */
export function decisionNote(
  decision: Pick<DecisionReview, "standing" | "ai_action">,
): string {
  if (decision.standing === "forced") return "Forced";
  if (decision.standing === "agreed") return "Agreed";
  const ai = shortAction(decision.ai_action);
  return decision.standing === "even" ? `Even: ${ai}` : `AI: ${ai}`;
}

/** Under the difference in a decision's detail. */
export function decisionVerdict(decision: DecisionReview): string {
  switch (decision.standing) {
    case "forced":
      return "forced";
    case "agreed":
      return "agreed";
    case "even":
      return "points · about even";
    case "slight":
      return "points · a small difference";
    default:
      return "points";
  }
}

/** Who the picker's partner is, as the table knew it at a decision. */
export type PartnerLine = {
  /** "Called" for a called ace, "Partner" otherwise. */
  label: string;
  /** The card that names the partner; null when the picker went alone. */
  card: string | null;
  under: boolean;
  /** The deciding seat holds the card: they are the partner. */
  held: boolean;
  note: string;
};

/** The partner call as it stood at a decision: what was called and, from
 * the deciding seat's view, whether the partner is known yet. Null before
 * the call is made (the pick and the call itself) and in a leaster. */
export function partnerLine(
  review: Pick<
    HandReview,
    "hand" | "seat" | "seats" | "called_card" | "called_under" | "partner_mode"
  >,
  decision: Pick<DecisionReview, "kind" | "hand" | "partner_known">,
): PartnerLine | null {
  if (decision.kind === "pick" || decision.kind === "call") return null;
  const picker = review.seats.find((s) => s.role === "picker")?.seat;
  if (picker == null) return null;
  const youPicked = picker === review.seat;
  if (review.hand.alone) {
    return {
      label: "Partner",
      card: null,
      under: false,
      held: false,
      note: youPicked ? "none, you went alone" : "none, the picker is alone",
    };
  }
  const card = review.partner_mode === 1 ? review.called_card : "JD";
  if (!card) return null;
  const held = decision.hand.includes(card);
  const note = held
    ? "in your hand"
    : !decision.partner_known
      ? "partner not yet known"
      : review.seats.find((s) => s.seat === review.seat)?.role === "partner"
        ? "you played it"
        : "partner known";
  return {
    label: review.partner_mode === 1 ? "Called" : "Partner",
    card,
    under: review.called_under,
    held,
    note,
  };
}

/** How the two expected scores were arrived at, in a sentence. */
export function decisionMethod(decision: DecisionReview): string {
  if (decision.standing === "forced")
    return "One legal choice, so there is nothing to compare.";
  if (decision.standing === "agreed")
    return "You and the AI make the same choice here.";
  if (decision.kind === "pick")
    return "Picking is averaged over every blind you could have found; passing is played forward with the cards as they were dealt, the AI playing the other seats.";
  return "Each choice is played forward to your next decision with the cards as they were dealt, the AI playing the other seats.";
}

export interface Headline {
  /** Small line above; null when there is nothing to say. */
  kicker: string | null;
  text: string;
  /** The decision the headline is about, if it is about one. */
  index: number | null;
}

function count(n: number, one: string, many: string): string {
  const words = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven"];
  return `${words[n] ?? n} ${n === 1 ? one : many}`;
}

/** A reviewed hand in one line. It names a single decision where one
 * stands out (the costliest, else the best gain) and otherwise says
 * plainly that nothing did. */
export function handHeadline(review: HandReview): Headline {
  const status = review.hand.review?.status;
  const summary = review.summary;
  if (status !== "reviewed" || !summary) {
    const text =
      status === "queued"
        ? "This hand is waiting for its review."
        : status === "skipped"
          ? "The AI made some of your decisions in this hand, so it is not reviewed."
          : "This hand could not be reviewed.";
    return { kicker: null, text, index: null };
  }
  const decision = review.decisions.find(
    (d) => d.index === summary.headline_index,
  );
  if (decision && decision.difference != null) {
    const costly = decision.difference < 0;
    return {
      kicker: `${costly ? "Costliest" : "Best"} decision · ${decisionLabel(decision)} · ${signed(decision.difference, 1)} points of expected score`,
      text: decisionHeadline(decision),
      index: decision.index,
    };
  }
  const different = summary.even + summary.slight;
  if (different > 0) {
    const choices = count(different, "different choice", "different choices");
    // Nothing reached a full point: say whether any of it was more than
    // noise.
    const size =
      summary.slight === 0
        ? different === 1
          ? "about even"
          : "all about even"
        : different === 1
          ? "worth less than a point"
          : "none worth a full point";
    return { kicker: null, text: `${choices}, ${size}.`, index: null };
  }
  return {
    kicker: null,
    text: "You and the AI agreed on every decision.",
    index: null,
  };
}

/** "8 decisions: 4 agreed, 1 about even, 2 cost you, 1 gained". */
export function decisionCounts(
  summary: NonNullable<HandReview["summary"]>,
): string {
  const parts = [
    [summary.agreed, "agreed"],
    [summary.even, "about even"],
    [
      summary.slight,
      summary.slight === 1 ? "small difference" : "small differences",
    ],
    [summary.costly, "cost you"],
    [summary.better, "gained"],
  ]
    .filter(([n]) => (n as number) > 0)
    .map(([n, label]) => `${n} ${label}`);
  const total = `${summary.decisions} ${summary.decisions === 1 ? "decision" : "decisions"}`;
  return parts.length ? `${total}: ${parts.join(", ")}` : total;
}

/** The two tables of the hand history page. */
export type HistoryView = "results" | "ai";

/** The hand history page opened on ``view``. */
export function historyHref(view: HistoryView): string {
  return view === "ai" ? "/account/history?view=ai" : "/account/history";
}

/** A hand's net against the AI from its two totals. */
export function net(cost: number, gained: number): number {
  return Math.round((gained - cost) * 10) / 10;
}

/** How a figure is coloured. Put on an element as data-tone, which
 * globals.css colours; "plain" leaves it as it was. */
export type Tone = "cost" | "gain" | "plain";

/** The colour of a hand's net: red or green by its sign for a notable
 * hand (a decision cost or gained a full point, and the net came out a
 * full point the same way), and plain for every other. */
export function netTone(value: number, notable: boolean): Tone {
  if (!notable) return "plain";
  return value < 0 ? "cost" : "gain";
}

/** The colour of a decision: red where it cost a full point against the
 * AI's choice, green where it gained one. The server bands the decision
 * (its standing); nothing here knows the thresholds. */
export function tone(standing: DecisionReview["standing"]): Tone {
  if (standing === "costly") return "cost";
  if (standing === "better") return "gain";
  return "plain";
}

/** A total over many hands is coloured once it is a point either way. A
 * rule of its own, for sums: it says nothing about any one decision. */
const TOTAL_TONE_FROM = 1;

export function totalTone(total: number): Tone {
  if (total <= -TOTAL_TONE_FROM) return "cost";
  if (total >= TOTAL_TONE_FROM) return "gain";
  return "plain";
}
