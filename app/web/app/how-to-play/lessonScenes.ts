// The How to Play lesson: chapters of beats, each beat a caption plus a
// scene that places every card it shows on the stage.
//
// Stage coordinates are in units of 1% of the stage width (the stage is a
// size container, so the player renders a unit as 1cqw). x/y name a card's
// centre. The stage is 100 wide and STAGE_H[narrow] tall; narrow is the
// portrait stage used on phones, which gets its own placements rather than a
// squeezed copy of the wide ones.

export type CardTone = "hi" | "trump" | "dim";

export interface CardPlacement {
  x: number;
  y: number;
  /** Rotation in degrees. */
  r?: number;
  faceDown?: boolean;
  tone?: CardTone;
}

export type LabelKind = "overline" | "tag" | "tagTrump" | "tagRed" | "big";

export interface StageLabel {
  id: string;
  text: string;
  x: number;
  y: number;
  kind: LabelKind;
  /** Which edge of the label sits at x (default centre). */
  align?: "start" | "center";
}

/** Cards a scene leaves out fade away where they last stood. */
export interface Scene {
  cards: Record<string, CardPlacement>;
  labels: StageLabel[];
  /** The closing "start a game" button, centred at x/y. */
  cta?: { text: string; x: number; y: number };
}

export interface Beat {
  caption: string;
  scene: (narrow: boolean) => Scene;
}

export interface Chapter {
  id: string;
  title: string;
  beats: Beat[];
}

export const STAGE_H = { wide: 54, narrow: 128 } as const;
export const CARD_W = { wide: 6.6, narrow: 12.4 } as const;

// ---- The deck ----

const PRINTED_RANKS = ["A", "K", "Q", "J", "10", "9", "8", "7"];
const SUITS = ["C", "S", "H", "D"] as const;

/** All 32 cards, in printed suit order — also the fan's order. */
export const DECK = SUITS.flatMap((s) => PRINTED_RANKS.map((r) => r + s));

const TRUMP = [
  "QC", "QS", "QH", "QD", "JC", "JS", "JH", "JD",
  "AD", "10D", "KD", "9D", "8D", "7D",
]; // prettier-ignore
const FAIL_RANKS = ["A", "10", "K", "9", "8", "7"];
const FAIL_SUITS = [
  { suit: "C", name: "Clubs ♣" },
  { suit: "S", name: "Spades ♠" },
  { suit: "H", name: "Hearts ♥" },
];

const isTrump = (c: string) => TRUMP.includes(c);
const rankOf = (c: string) => c.slice(0, -1);

// One deal, played through from "you"'s seat: you pick, so "you" is the
// picker in every chapter. Your six hold four trump and a Queen (the picking
// tip); the blind brings two hearts, so you can bury your own clubs and still
// call hearts; the partner holds the A♥; and the player on your left has no
// fail hearts, only the J♥, so they trump in when hearts are led.
const DEAL = {
  you: ["QC", "JS", "AD", "9D", "10C", "KC"],
  left: ["QS", "JH", "10D", "9C", "8C", "8S"],
  farLeft: ["AS", "QH", "KD", "AC", "10H", "7S"],
  farRight: ["AH", "KS", "QD", "JC", "8D", "7C"],
  right: ["10S", "9S", "7H", "JD", "7D", "KH"],
};
type Seat = keyof typeof DEAL;
const BLIND = ["9H", "8H"];
const BURY = ["10C", "KC"];
/** Your hand after picking and burying. */
const PICKER_HAND = ["QC", "JS", "AD", "9D", "9H", "8H"];

// ---- Layout helpers ----

type Cards = Record<string, CardPlacement>;

/** A centred row of cards; `extra` adds per-card fields. */
function row(
  cards: string[],
  cx: number,
  y: number,
  step: number,
  extra: (c: string, i: number) => Partial<CardPlacement> = () => ({}),
): Cards {
  const x0 = cx - ((cards.length - 1) * step) / 2;
  return Object.fromEntries(
    cards.map((c, i) => [c, { x: x0 + i * step, y, ...extra(c, i) }]),
  );
}

/** Cards on an arc around a pivot below them, like a fanned hand. */
function fan(
  cards: string[],
  cx: number,
  pivotY: number,
  radius: number,
  spreadDeg: number,
): Cards {
  return Object.fromEntries(
    cards.map((c, i) => {
      const a = (i / (cards.length - 1) - 0.5) * spreadDeg;
      const rad = (a * Math.PI) / 180;
      return [
        c,
        {
          x: cx + radius * Math.sin(rad),
          y: pivotY - radius * Math.cos(rad),
          r: a,
        },
      ];
    }),
  );
}

/** Deterministic jitter in [0, 1) so piles look tossed but never reshuffle. */
const jitter = (i: number) => {
  const v = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
  return v - Math.floor(v);
};

/** A loose won-tricks pile. */
function pile(cards: string[], cx: number, cy: number, spread: number): Cards {
  return Object.fromEntries(
    cards.map((c, i) => [
      c,
      {
        x: cx + (jitter(i) - 0.5) * spread,
        y: cy + (jitter(i + 40) - 0.5) * spread * 0.6,
        r: (jitter(i + 80) - 0.5) * 34,
        faceDown: false,
      },
    ]),
  );
}

function withTone(cards: Cards, codes: string[], tone: CardTone): Cards {
  const out = { ...cards };
  for (const c of codes) if (out[c]) out[c] = { ...out[c], tone };
  return out;
}

const SEATS: Record<
  "wide" | "narrow",
  Record<Seat, { x: number; y: number }>
> = {
  wide: {
    you: { x: 50, y: 44 },
    left: { x: 13, y: 33 },
    farLeft: { x: 22, y: 10 },
    farRight: { x: 78, y: 10 },
    right: { x: 87, y: 33 },
  },
  narrow: {
    you: { x: 50, y: 110 },
    left: { x: 25, y: 48 },
    farLeft: { x: 25, y: 14 },
    farRight: { x: 75, y: 14 },
    right: { x: 75, y: 48 },
  },
};

const mode = (narrow: boolean) => (narrow ? "narrow" : "wide");

/** One seat's hand: face down and tight, or face up and readable. */
function seatHand(
  narrow: boolean,
  seat: Seat,
  cards: string[],
  faceUp = false,
): Cards {
  const { x, y } = SEATS[mode(narrow)][seat];
  const step = faceUp ? (narrow ? 10.8 : 7) : narrow ? 4.2 : 2.6;
  return row(cards, x, y, step, () => ({ faceDown: !faceUp }));
}

function seatLabel(
  narrow: boolean,
  seat: Seat,
  text: string,
  kind: LabelKind,
): StageLabel {
  const { x, y } = SEATS[mode(narrow)][seat];
  return { id: `seat-${seat}`, text, x, y: y + (narrow ? 13 : 8), kind };
}

const blindSpot = (narrow: boolean) =>
  narrow ? { x: 50, y: 74, step: 14 } : { x: 50, y: 23, step: 7.6 };

/** The whole table: every seat's hand plus whatever sits in the middle. */
function table(
  narrow: boolean,
  hands: Partial<Record<Seat, string[]>> = {},
  youFaceUp = false,
): Cards {
  let out: Cards = {};
  for (const seat of Object.keys(DEAL) as Seat[]) {
    out = {
      ...out,
      ...seatHand(
        narrow,
        seat,
        hands[seat] ?? DEAL[seat],
        seat === "you" && youFaceUp,
      ),
    };
  }
  return out;
}

// A lone card in the middle of the table, for the tips.
const trickSlots = (narrow: boolean) =>
  narrow ? { y: 44, tagY: 57 } : { y: 22, tagY: 29.5 };

// The lesson's one played trick: a defender's called-suit lead from your
// right, your heart, the J♥ trumping in, a 10♥ thrown onto the winner, and
// the partner forced to play the called Ace. Play moves to the left.
const TRICK: [Seat, string][] = [
  ["right", "KH"],
  ["you", "8H"],
  ["left", "JH"],
  ["farLeft", "10H"],
  ["farRight", "AH"],
];

// Where each seat's card lands, between its hand and the middle of the
// table. On the narrow stage the other hands draw in tight against the edges
// to leave the middle for the trick.
const PLAYED: Record<
  "wide" | "narrow",
  Record<Seat, { x: number; y: number }>
> = {
  wide: {
    you: { x: 50, y: 32 },
    left: { x: 35.2, y: 27.6 },
    farLeft: { x: 38.8, y: 18.4 },
    farRight: { x: 61.2, y: 18.4 },
    right: { x: 64.8, y: 27.6 },
  },
  narrow: {
    you: { x: 50, y: 78 },
    left: { x: 35, y: 54 },
    farLeft: { x: 39, y: 30 },
    farRight: { x: 61, y: 30 },
    right: { x: 65, y: 54 },
  },
};
const TRICK_SEATS_NARROW: Record<Seat, { x: number; y: number }> = {
  ...SEATS.narrow,
  left: { x: 16, y: 48 },
  farLeft: { x: 16, y: 14 },
  farRight: { x: 84, y: 14 },
  right: { x: 84, y: 48 },
};
const trickSeat = (narrow: boolean, seat: Seat) =>
  narrow ? TRICK_SEATS_NARROW[seat] : SEATS.wide[seat];

/** The table once the first `n` cards of the trick are played: what's left
 * of each hand, plus the played cards in front of their seats. */
function trickTable(narrow: boolean, n: number): Cards {
  const played = TRICK.slice(0, n);
  const playedCards = played.map(([, c]) => c);
  let out: Cards = {};
  for (const seat of Object.keys(DEAL) as Seat[]) {
    const you = seat === "you";
    const hand = (you ? PICKER_HAND : DEAL[seat]).filter(
      (c) => !playedCards.includes(c),
    );
    const { x, y } = trickSeat(narrow, seat);
    const step = you ? (narrow ? 10.8 : 7) : narrow ? 2.4 : 2.6;
    out = { ...out, ...row(hand, x, y, step, () => ({ faceDown: !you })) };
  }
  for (const [seat, c] of played) out[c] = { ...PLAYED[mode(narrow)][seat] };
  return out;
}

/** A tag just under a seat's played card. */
function playedTag(
  narrow: boolean,
  seat: Seat,
  text: string,
  kind: LabelKind,
): StageLabel {
  const { x, y } = PLAYED[mode(narrow)][seat];
  return { id: `played-${seat}`, text, x, y: y + (narrow ? 12.5 : 7.5), kind };
}

/** A label under a seat's hand in the trick layout. */
function trickSeatLabel(
  narrow: boolean,
  seat: Seat,
  text: string,
  kind: LabelKind,
): StageLabel {
  const { x, y } = trickSeat(narrow, seat);
  return { id: `seat-${seat}`, text, x, y: y + (narrow ? 13 : 8), kind };
}

/** A big display numeral with a small caption under it. */
function figure(
  id: string,
  big: string,
  caption: string,
  x: number,
  y: number,
  narrow: boolean,
): StageLabel[] {
  return [
    { id: `${id}-big`, text: big, x, y, kind: "big" },
    {
      id: `${id}-cap`,
      text: caption,
      x,
      y: y + (narrow ? 10 : 7.5),
      kind: "overline",
    },
  ];
}

function deckFan(narrow: boolean): Cards {
  return narrow ? fan(DECK, 50, 170, 92, 56) : fan(DECK, 50, 98, 66, 84);
}

// ---- Chapters ----

// Points live with the cards; the lesson's two piles reuse these splits.
const PICKER_61 = ["AC", "AS", "AH", "AD", "10C", "KC", "QC"];
const PICKER_74 = [...PICKER_61, "10S", "QS"];

const POINT_GROUPS = [
  { ranks: ["A"], points: "11", name: "Aces" },
  { ranks: ["10"], points: "10", name: "Tens" },
  { ranks: ["K"], points: "4", name: "Kings" },
  { ranks: ["Q"], points: "3", name: "Queens" },
  { ranks: ["J"], points: "2", name: "Jacks" },
  { ranks: ["9", "8", "7"], points: "0", name: "9 · 8 · 7" },
];

export const CHAPTERS: Chapter[] = [
  {
    id: "deck",
    title: "The deck",
    beats: [
      {
        caption:
          "Sheepshead uses a short deck of 32 cards: the sevens through the aces of all four suits.",
        scene: (narrow) => ({
          cards: deckFan(narrow),
          labels: figure(
            "deck",
            "32",
            "cards · four suits · seven to ace",
            50,
            narrow ? 30 : 9,
            narrow,
          ),
        }),
      },
      {
        caption:
          "Five players are dealt six cards each. The last two go face down in the middle: the blind.",
        scene: (narrow) => {
          const b = blindSpot(narrow);
          return {
            cards: {
              ...table(narrow),
              ...row(BLIND, b.x, b.y, b.step, () => ({ faceDown: true })),
            },
            labels: [
              {
                id: "blind",
                text: "The blind",
                x: b.x,
                y: b.y + (narrow ? 13 : 8),
                kind: "tag",
              },
            ],
          };
        },
      },
    ],
  },
  {
    id: "trump",
    title: "Trump & fail",
    beats: [
      {
        caption:
          "Sorted by suit, it looks like any other deck: clubs, spades, hearts and diamonds, ace down to seven.",
        scene: (narrow) => suitRows(narrow, false),
      },
      {
        caption:
          "But in Sheepshead every Queen, every Jack and every Diamond is trump: fourteen cards that beat everything else.",
        scene: (narrow) => suitRows(narrow, true),
      },
      {
        caption:
          "Trump runs from the Queen of Clubs down to the 7 of Diamonds. The other 18 cards are fail. In all suits the 10 outranks the King.",
        scene: (narrow) => trumpAndFail(narrow),
      },
    ],
  },
  {
    id: "points",
    title: "Points",
    beats: [
      {
        caption:
          "Cards also carry points. The Queens and Jacks are the strongest cards, but the aces and tens hold most of the points.",
        scene: (narrow) => pointGroups(narrow),
      },
      {
        caption:
          "That makes 120 points in the deck. To win, a team needs more than half of them: 61.",
        scene: (narrow) => {
          const rest = DECK.filter((c) => !PICKER_61.includes(c));
          const [lx, rx, py, spread] = narrow
            ? [27, 73, 70, 14]
            : [22, 78, 33, 10];
          return {
            cards: {
              ...pile(PICKER_61, lx, py, spread),
              ...pile(rest, rx, py, spread),
            },
            labels: [
              ...figure(
                "total",
                "120",
                "points in the deck",
                50,
                narrow ? 16 : 26,
                narrow,
              ),
              ...figure("win", "61", "wins", lx, narrow ? 42 : 10, narrow),
              ...figure("lose", "59", "loses", rx, narrow ? 42 : 10, narrow),
            ],
          };
        },
      },
    ],
  },
  {
    id: "picking",
    title: "Picking",
    beats: [
      {
        caption:
          "Starting to the dealer's left, each player may pick up the blind or pass. The first to pick becomes the picker.",
        scene: (narrow) => {
          const b = blindSpot(narrow);
          return {
            cards: {
              ...table(narrow, {}, true),
              ...row(BLIND, b.x, b.y, b.step, () => ({ faceDown: true })),
            },
            labels: [seatLabel(narrow, "you", "You", "overline")],
          };
        },
      },
      {
        caption: "The picker adds the two blind cards to their hand…",
        scene: (narrow) => ({
          cards: withTone(
            table(narrow, { you: [...DEAL.you, ...BLIND] }, true),
            BLIND,
            "hi",
          ),
          labels: [seatLabel(narrow, "you", "Picker", "tagTrump")],
        }),
      },
      {
        caption:
          "…then buries two cards face down. Their points still count for the picker.",
        scene: (narrow) => {
          const b = blindSpot(narrow);
          return {
            cards: {
              ...table(narrow, { you: PICKER_HAND }, true),
              ...row(BURY, b.x, b.y, b.step, () => ({ faceDown: true })),
            },
            labels: [
              seatLabel(narrow, "you", "Picker", "tagTrump"),
              {
                id: "blind",
                text: "Buried",
                x: b.x,
                y: b.y + (narrow ? 13 : 8),
                kind: "tag",
              },
            ],
          };
        },
      },
      {
        caption:
          "Then the picker calls the Ace of a fail suit they hold, here hearts. Whoever has the A♥ is the secret partner: two against three.",
        scene: (narrow) => {
          const b = blindSpot(narrow);
          // Drawn last and raised out of the partner's hand so the viewer
          // sees it.
          const others = table(narrow, { you: PICKER_HAND }, true);
          delete others.AH;
          const cards = withTone(
            {
              ...others,
              AH: {
                ...seatHand(narrow, "farRight", DEAL.farRight).AH,
                y: SEATS[mode(narrow)].farRight.y - (narrow ? 6 : 3.5),
                faceDown: false,
              },
            },
            ["9H", "8H", "AH"],
            "hi",
          );
          return {
            cards,
            labels: [
              ...figure(
                "call",
                "A♥",
                "called",
                b.x,
                b.y - (narrow ? 4 : 2),
                narrow,
              ),
              seatLabel(narrow, "you", "Picker", "tagTrump"),
              seatLabel(narrow, "farRight", "Partner · secret", "tagTrump"),
              seatLabel(narrow, "left", "Defender", "tag"),
              seatLabel(narrow, "farLeft", "Defender", "tag"),
              seatLabel(narrow, "right", "Defender", "tag"),
            ],
          };
        },
      },
    ],
  },
  {
    id: "following",
    title: "Following suit",
    beats: [
      {
        caption:
          "Play moves to the left. The player on your right leads the K♥, and everyone must follow the led suit if they can, so you must play a heart.",
        scene: (narrow) => ({
          cards: withTone(
            withTone(trickTable(narrow, 1), ["9H", "8H"], "hi"),
            ["QC", "JS", "AD", "9D"],
            "dim",
          ),
          labels: [
            playedTag(narrow, "right", "Led", "tag"),
            trickSeatLabel(narrow, "you", "You · picker", "tagTrump"),
          ],
        }),
      },
      {
        caption:
          "The next player has no hearts, so they may play anything. Their J♥ is trump, not a heart, and trump beats every fail card.",
        scene: (narrow) => ({
          cards: withTone(trickTable(narrow, 3), ["JH"], "hi"),
          labels: [
            playedTag(narrow, "right", "Led", "tag"),
            playedTag(narrow, "left", "Trump", "tagTrump"),
            trickSeatLabel(narrow, "you", "You · picker", "tagTrump"),
          ],
        }),
      },
      {
        caption:
          "Hearts is the called suit, so whoever holds the A♥ must play it. Now everyone knows who your partner is.",
        scene: (narrow) => ({
          cards: withTone(trickTable(narrow, 5), ["AH"], "hi"),
          labels: [
            playedTag(narrow, "right", "Led", "tag"),
            playedTag(narrow, "left", "Trump", "tagTrump"),
            playedTag(narrow, "farRight", "Called ace", "tagRed"),
            trickSeatLabel(narrow, "you", "You · picker", "tagTrump"),
            trickSeatLabel(narrow, "farRight", "Partner", "tagTrump"),
          ],
        }),
      },
      {
        caption:
          "The highest trump wins; if nobody trumps, the highest card of the led suit does. The J♥ takes all 27 points for the defenders, and its player leads next.",
        scene: (narrow) => {
          const trickCards = TRICK.map(([, c]) => c);
          const [px, py, spread] = narrow ? [24, 80, 6] : [30, 30, 5];
          return {
            cards: {
              ...trickTable(narrow, 5),
              ...withTone(pile(trickCards, px, py, spread), ["JH"], "hi"),
            },
            labels: [
              ...figure(
                "trick",
                "27",
                "points to the defenders",
                narrow ? 66 : 54,
                narrow ? 74 : 21,
                narrow,
              ),
              trickSeatLabel(narrow, "you", "You · picker", "tagTrump"),
              trickSeatLabel(narrow, "farRight", "Partner", "tagTrump"),
            ],
          };
        },
      },
    ],
  },
  {
    id: "scoring",
    title: "Scoring",
    beats: [
      {
        caption:
          "After six tricks, each team counts its points. With 61 or more, the picker and partner win; otherwise the defenders do. The app keeps score for you.",
        scene: (narrow) => {
          const rest = DECK.filter((c) => !PICKER_74.includes(c));
          const [lx, rx, py, spread] = narrow
            ? [27, 73, 70, 14]
            : [22, 78, 33, 10];
          return {
            cards: {
              ...pile(PICKER_74, lx, py, spread),
              ...pile(rest, rx, py, spread),
            },
            labels: [
              ...figure(
                "win",
                "74",
                "picker + partner",
                lx,
                narrow ? 42 : 10,
                narrow,
              ),
              ...figure(
                "lose",
                "46",
                "defenders",
                rx,
                narrow ? 42 : 10,
                narrow,
              ),
              {
                id: "result",
                text: "Picker wins",
                x: 50,
                y: narrow ? 20 : 30,
                kind: "tagTrump",
              },
            ],
          };
        },
      },
    ],
  },
  {
    id: "tips",
    title: "Tips",
    beats: [
      {
        caption:
          "When to pick: four or more trump, including at least one Queen, like the hand you picked with here.",
        scene: (narrow) => {
          const hand = DEAL.you;
          const step = narrow ? 14 : 8.4;
          const y = narrow ? 64 : 27;
          const trumpMid = 50 - step;
          return {
            cards: withTone(
              withTone(row(hand, 50, y, step), ["JS", "AD", "9D"], "trump"),
              ["QC"],
              "hi",
            ),
            labels: [
              tipHeading(narrow, "Tip 1 · When to pick"),
              {
                id: "tip-a",
                text: "4 trump",
                x: trumpMid,
                y: y + (narrow ? 15 : 9),
                kind: "tagTrump",
              },
              {
                id: "tip-b",
                text: "a Queen",
                x: 50 - 2.5 * step,
                y: y - (narrow ? 15 : 9),
                kind: "tagRed",
              },
            ],
          };
        },
      },
      {
        caption:
          "On the picker's team, lead trump. Drawing out the defenders' trump keeps it from taking your fail and your points.",
        scene: (narrow) =>
          tipLead(
            narrow,
            "Tip 2 · Picker & partner",
            "QC",
            PICKER_HAND,
            "Lead trump",
          ),
      },
      {
        caption:
          "On defense, lead the called suit or another fail, and save your trump. Leading the called suit makes the partner give up their Ace.",
        scene: (narrow) =>
          tipLead(
            narrow,
            "Tip 3 · Defenders",
            "10H",
            DEAL.farLeft,
            "Called suit",
          ),
      },
      {
        caption:
          "That's everything you need for your first hand. At the table, the cards you're allowed to play are highlighted for you.",
        scene: (narrow) => ({
          cards: deckFan(narrow),
          labels: [
            {
              id: "end",
              text: "That's the game",
              x: 50,
              y: narrow ? 24 : 5,
              kind: "overline",
            },
          ],
          cta: { text: "Deal me in", x: 50, y: narrow ? 36 : 12.5 },
        }),
      },
    ],
  },
];

// ---- Scene builders used above ----

function suitRows(narrow: boolean, showTrump: boolean): Scene {
  const names = ["Clubs ♣", "Spades ♠", "Hearts ♥", "Diamonds ♦"];
  let cards: Cards = {};
  const labels: StageLabel[] = [];
  SUITS.forEach((s, i) => {
    const codes = PRINTED_RANKS.map((r) => r + s);
    const y = narrow ? 20 + i * 29 : 8 + i * 12.6;
    const lift = (c: string): Partial<CardPlacement> =>
      showTrump && isTrump(c)
        ? { tone: "trump", y: y - (narrow ? 2 : 1.6) }
        : showTrump
          ? { tone: "dim" }
          : {};
    cards = {
      ...cards,
      ...(narrow ? row(codes, 50, y, 11, lift) : row(codes, 58, y, 8, lift)),
    };
    labels.push(
      narrow
        ? {
            id: `suit-${s}`,
            text: names[i],
            x: 6,
            y: y - 14.5,
            kind: "overline",
            align: "start",
          }
        : {
            id: `suit-${s}`,
            text: names[i],
            x: 4,
            y,
            kind: "overline",
            align: "start",
          },
    );
  });
  return { cards, labels };
}

/** Trump in power order over the three fail suits, ace down to seven: the
 * lesson's last trump beat, and the table cheat sheet's Power page. */
export function trumpAndFail(narrow: boolean): Scene {
  let cards: Cards;
  const labels: StageLabel[] = [];
  if (narrow) {
    cards = {
      ...row(TRUMP.slice(0, 7), 50, 20, 12.8, () => ({ tone: "trump" })),
      ...row(TRUMP.slice(7), 50, 41, 12.8, () => ({ tone: "trump" })),
    };
    FAIL_SUITS.forEach(({ suit }, i) => {
      cards = {
        ...cards,
        ...row(
          FAIL_RANKS.map((r) => r + suit),
          50,
          70 + i * 21,
          13,
        ),
      };
    });
    labels.push(
      {
        id: "trump",
        text: "Trump · highest → lowest",
        x: 6,
        y: 6.5,
        kind: "tagTrump",
        align: "start",
      },
      { id: "fail", text: "Fail", x: 6, y: 56, kind: "tag", align: "start" },
    );
  } else {
    cards = row(TRUMP, 59, 11, 5.6, () => ({ tone: "trump" }));
    FAIL_SUITS.forEach(({ suit, name }, i) => {
      const y = 25.5 + i * 11.6;
      cards = {
        ...cards,
        ...row(
          FAIL_RANKS.map((r) => r + suit),
          40.5,
          y,
          7.4,
        ),
      };
      labels.push({
        id: `fail-${suit}`,
        text: name,
        x: 64,
        y,
        kind: "overline",
        align: "start",
      });
    });
    labels.push(
      {
        id: "trump",
        text: "Trump",
        x: 3,
        y: 11,
        kind: "tagTrump",
        align: "start",
      },
      {
        id: "trump-order",
        text: "high → low",
        x: 3,
        y: 16.5,
        kind: "overline",
        align: "start",
      },
      { id: "fail", text: "Fail", x: 3, y: 37.1, kind: "tag", align: "start" },
    );
  }
  return { cards, labels };
}

/** The deck in columns by point value: the lesson's first points beat, and
 * the table cheat sheet's Points page. */
export function pointGroups(narrow: boolean): Scene {
  const cards: Cards = {};
  const labels: StageLabel[] = [];
  POINT_GROUPS.forEach((g, i) => {
    const codes = DECK.filter((c) => g.ranks.includes(rankOf(c)));
    const zero = g.points === "0";
    const [x, top] = narrow
      ? [17 + (i % 3) * 33, i < 3 ? 26 : 82]
      : [9 + i * 16.4, 22];
    const dy = zero ? (narrow ? 2 : 1.4) : narrow ? 4.5 : 3.4;
    codes.forEach((c, j) => {
      cards[c] = { x, y: top + j * dy, tone: zero ? "dim" : undefined };
    });
    const bottom = top + (codes.length - 1) * dy;
    labels.push(
      {
        id: `pts-${i}`,
        text: g.points,
        x,
        y: top - (narrow ? 15 : 11),
        kind: "big",
      },
      {
        id: `pts-name-${i}`,
        text: g.name,
        x,
        y: bottom + (narrow ? 13 : 8.5),
        kind: "overline",
      },
    );
  });
  return { cards, labels };
}

function tipHeading(narrow: boolean, text: string): StageLabel {
  return {
    id: "tip-heading",
    text,
    x: 50,
    y: narrow ? 10 : 5,
    kind: "overline",
  };
}

function tipLead(
  narrow: boolean,
  heading: string,
  lead: string,
  hand: string[],
  tag: string,
): Scene {
  const rest = hand.filter((c) => c !== lead);
  const restCards = row(
    rest,
    50,
    narrow ? 100 : 44,
    narrow ? 13 : 7.6,
    (c) => ({
      tone: isTrump(c) === isTrump(lead) ? undefined : "dim",
    }),
  );
  const { y, tagY } = trickSlots(narrow);
  return {
    cards: { [lead]: { x: 50, y, tone: "hi" }, ...restCards },
    labels: [
      tipHeading(narrow, heading),
      {
        id: "tip-tag",
        text: tag,
        x: 50,
        y: tagY,
        kind: isTrump(lead) ? "tagTrump" : "tagRed",
      },
    ],
  };
}
