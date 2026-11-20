import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";

// The link-preview card: the landing page's wordmark and lede on the left, a
// picker's hand fanned on the right. Rendered once at build time (nothing
// here reads the request), so the fonts are read from disk at build only.
// Colours track globals.css; Satori can't read CSS variables.

export const alt =
  "Sheepshead AI: play live with friends, strangers, and a deep neural network.";

// Designed at 1200×630, the size link previews are specified in, and
// rendered at twice that so the card stays sharp on high-density screens.
// Every length goes through px() rather than a scale() transform on the
// whole layout: Satori snaps boxes to whole layout pixels, and a transform
// would double that snap, enough to set a corner's suit visibly off-centre
// under its rank.
const SCALE = 2;
const px = (n: number) => n * SCALE;
export const size = { width: px(1200), height: px(630) };
export const contentType = "image/png";

const PAPER = "#f3f1ea";
const INK = "#0e0d0a";
const INK_SOFT = "#2c2b27";
const MUTED = "#6e6c63";
const RULE_STRONG = "#a3a094";
const ACCENT = "#1f3b2f";
const CARD_RED = "#b3331e";
const CARD_EDGE = "#d5d2c4";

const font = (file: string) =>
  readFile(join(process.cwd(), "assets/fonts", file));

type Suit = "C" | "S" | "H" | "D";

// The suit glyphs come from Sheepshead Suits, the four-glyph font the site
// itself uses for them (see app/layout.tsx), so the CSS from
// PlayingCard.module.css below lays the card out as the site does. Satori
// draws text as vector outlines, so the pips stay as sharp as the ranks
// through the cards' rotation.
const SUIT_GLYPH: Record<Suit, string> = {
  C: "♣",
  S: "♠",
  H: "♥",
  D: "♦",
};

// The suit glyphs' ink sits below their line-box centre, and their pointed
// bottoms make them read lower still, so the centre pip is lifted, the heart
// and diamond (drawn higher in the em-box) less than the others. As in
// PlayingCard.module.css, in em.
const CENTER_LIFT: Record<Suit, number> = {
  C: 0.12,
  S: 0.12,
  H: 0.03,
  D: 0.05,
};

// How far each rank's ink sits right of its advance box's centre in
// Instrument Serif, in em (the J's hook swings left, the 1 of 10 carries a
// wide left bearing). The site centres a corner's suit on the rank's box;
// fractions of a pixel on its cards, but at this size a visible lean, so
// here the suit is centred on the rank's ink instead.
const RANK_INK_OFFSET: Record<string, number> = {
  "7": 0.0069,
  "8": 0,
  "9": -0.005,
  "10": -0.0155,
  J: -0.0375,
  Q: 0.02,
  K: 0.0195,
  A: 0,
};

// Sizes are PlayingCard.module.css's fractions of the card width.
const CARD_W = px(150);
const CARD_H = CARD_W * 1.45;
const RANK_SIZE = CARD_W * 0.18;
const CORNER_SUIT_SIZE = CARD_W * 0.14;
const CENTER_SUIT_SIZE = CARD_W * 0.5;
const CORNER_INSET = CARD_W * 0.06;

function Card({ code, angle }: { code: string; angle: number }) {
  const rank = code.slice(0, -1);
  const suit = code.slice(-1) as Suit;
  const glyph = SUIT_GLYPH[suit];
  const color = suit === "H" || suit === "D" ? CARD_RED : INK;
  const corner = (flip: boolean) => (
    <div
      style={{
        position: "absolute",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: px(1),
        ...(flip
          ? {
              right: CORNER_INSET,
              bottom: CORNER_INSET,
              transform: "rotate(180deg)",
            }
          : { left: CORNER_INSET, top: CORNER_INSET }),
      }}
    >
      <div style={{ fontSize: RANK_SIZE, lineHeight: 0.9 }}>{rank}</div>
      <div
        style={{
          fontFamily: "Sheepshead Suits",
          fontSize: CORNER_SUIT_SIZE,
          lineHeight: 1,
          transform: `translateX(${RANK_INK_OFFSET[rank] * RANK_SIZE}px)`,
        }}
      >
        {glyph}
      </div>
    </div>
  );
  return (
    <div
      style={{
        position: "absolute",
        left: "50%",
        top: "50%",
        marginLeft: -CARD_W / 2,
        marginTop: -CARD_H / 2,
        width: CARD_W,
        height: CARD_H,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#ffffff",
        border: `${px(3)}px solid ${CARD_EDGE}`,
        borderRadius: px(7),
        boxShadow: `0 ${px(6)}px ${px(18)}px rgba(15, 15, 12, 0.12)`,
        color,
        fontFamily: "Instrument Serif",
        transform: `rotate(${angle}deg)`,
        transformOrigin: "50% 160%",
      }}
    >
      {corner(false)}
      <div
        style={{
          fontFamily: "Sheepshead Suits",
          fontSize: CENTER_SUIT_SIZE,
          lineHeight: 1,
          opacity: 0.92,
          transform: `translateY(${-CENTER_LIFT[suit] * CENTER_SUIT_SIZE}px)`,
        }}
      >
        {glyph}
      </div>
      {corner(true)}
    </div>
  );
}

// Q♣ and J♠ are the top trumps, J♦ the usual partner card.
const HAND = ["QC", "JS", "JD", "AH", "10H", "KC"];
const SPREAD = 9;

export default async function Image() {
  const [serif, serifItalic, geist, geistMedium, suits] = await Promise.all([
    font("InstrumentSerif-Regular.ttf"),
    font("InstrumentSerif-Italic.ttf"),
    font("Geist-Regular.ttf"),
    font("Geist-Medium.ttf"),
    font("SheepsheadSuits-Regular.ttf"),
  ]);

  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
        padding: `${px(56)}px ${px(72)}px`,
        background: PAPER,
        color: INK,
      }}
    >
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <div
          style={{
            fontFamily: "Geist",
            fontWeight: 500,
            fontSize: px(15),
            letterSpacing: px(2.7),
            color: MUTED,
            marginBottom: px(26),
          }}
        >
          THE CARD ROOM · WISCONSIN
        </div>
        <div
          style={{
            fontFamily: "Instrument Serif",
            fontSize: px(150),
            lineHeight: 0.92,
            letterSpacing: px(-3),
          }}
        >
          Sheepshead
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: px(22),
            marginTop: px(8),
          }}
        >
          <div
            style={{
              fontFamily: "Instrument Serif",
              fontStyle: "italic",
              fontSize: px(88),
              lineHeight: 0.9,
              color: ACCENT,
            }}
          >
            AI
          </div>
          <div
            style={{
              flex: 1,
              height: px(1.5),
              background: RULE_STRONG,
              marginTop: px(14),
            }}
          />
        </div>
        <div
          style={{
            fontFamily: "Geist",
            fontSize: px(26),
            lineHeight: 1.4,
            color: INK_SOFT,
            marginTop: px(34),
            maxWidth: px(560),
            textWrap: "balance",
          }}
        >
          Play live with friends, strangers, and a deep neural network.
        </div>
      </div>
      <div
        style={{
          width: px(470),
          height: px(520),
          position: "relative",
          display: "flex",
        }}
      >
        {HAND.map((code, i) => (
          <Card
            key={code}
            code={code}
            angle={(i - (HAND.length - 1) / 2) * SPREAD}
          />
        ))}
      </div>
    </div>,
    {
      ...size,
      fonts: [
        { name: "Instrument Serif", data: serif, style: "normal", weight: 400 },
        {
          name: "Instrument Serif",
          data: serifItalic,
          style: "italic",
          weight: 400,
        },
        { name: "Geist", data: geist, style: "normal", weight: 400 },
        { name: "Geist", data: geistMedium, style: "normal", weight: 500 },
        {
          name: "Sheepshead Suits",
          data: suits,
          style: "normal",
          weight: 400,
        },
      ],
    },
  );
}
