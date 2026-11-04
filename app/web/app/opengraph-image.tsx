import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";

// The link-preview card: the landing page's wordmark and lede on the left, a
// picker's hand fanned on the right. Rendered once at build time (nothing
// here reads the request), so the fonts are read from disk at build only.
// Colours track globals.css; Satori can't read CSS variables.

export const alt =
  "Sheepshead AI: play live with friends, strangers, and a deep neural network.";
export const size = { width: 1200, height: 630 };
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

// Neither face has the suit glyphs (the site's cards borrow them from system
// fallback fonts), so the pips are drawn. 100×100 box, pip centred.
const SUIT_PATHS: Record<Suit, string> = {
  D: "M50 2 L86 50 L50 98 L14 50 Z",
  H: "M50 94 C20 72 4 52 4 31 C4 15 16 5 29 5 C39 5 46 11 50 21 C54 11 61 5 71 5 C84 5 96 15 96 31 C96 52 80 72 50 94 Z",
  S: "M50 3 C78 28 96 44 96 61 C96 75 85 83 73 83 C64 83 57 79 53 72 C54 83 58 91 66 97 L34 97 C42 91 46 83 47 72 C43 79 36 83 27 83 C15 83 4 75 4 61 C4 44 22 28 50 3 Z",
  C: "M45 56 C45 78 41 90 33 97 L67 97 C59 90 55 78 55 56 Z",
};

// The club's three leaves, drawn over its stem.
const CLUB_LEAVES = [
  [50, 26],
  [27, 57],
  [73, 57],
] as const;

function Pip({ suit, px }: { suit: Suit; px: number }) {
  const red = suit === "H" || suit === "D";
  return (
    <svg width={px} height={px} viewBox="0 0 100 100">
      <path d={SUIT_PATHS[suit]} fill={red ? CARD_RED : INK} />
      {suit === "C" &&
        CLUB_LEAVES.map(([cx, cy]) => (
          <circle key={cx} cx={cx} cy={cy} r={22} fill={INK} />
        ))}
    </svg>
  );
}

const CARD_W = 150;
const CARD_H = CARD_W * 1.45;

function Card({ code, angle }: { code: string; angle: number }) {
  const rank = code.slice(0, -1);
  const suit = code.slice(-1) as Suit;
  const color = suit === "H" || suit === "D" ? CARD_RED : INK;
  const corner = (flip: boolean) => (
    <div
      style={{
        position: "absolute",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 3,
        ...(flip
          ? { right: 11, bottom: 9, transform: "rotate(180deg)" }
          : { left: 11, top: 9 }),
      }}
    >
      <div style={{ fontSize: 32, lineHeight: 0.9 }}>{rank}</div>
      <Pip suit={suit} px={20} />
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
        border: `3px solid ${CARD_EDGE}`,
        borderRadius: 7,
        boxShadow: "0 6px 18px rgba(15, 15, 12, 0.12)",
        color,
        fontFamily: "Instrument Serif",
        transform: `rotate(${angle}deg)`,
        transformOrigin: "50% 160%",
      }}
    >
      {corner(false)}
      <Pip suit={suit} px={62} />
      {corner(true)}
    </div>
  );
}

// Q♣ and J♠ are the top trumps, J♦ the usual partner card.
const HAND = ["QC", "JS", "JD", "AH", "10H", "KC"];
const SPREAD = 9;

export default async function Image() {
  const [serif, serifItalic, geist, geistMedium] = await Promise.all([
    font("InstrumentSerif-Regular.ttf"),
    font("InstrumentSerif-Italic.ttf"),
    font("Geist-Regular.ttf"),
    font("Geist-Medium.ttf"),
  ]);

  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
        padding: "56px 72px",
        background: PAPER,
        color: INK,
      }}
    >
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <div
          style={{
            fontFamily: "Geist",
            fontWeight: 500,
            fontSize: 15,
            letterSpacing: 2.7,
            color: MUTED,
            marginBottom: 26,
          }}
        >
          THE CARD ROOM · WISCONSIN
        </div>
        <div
          style={{
            fontFamily: "Instrument Serif",
            fontSize: 150,
            lineHeight: 0.92,
            letterSpacing: -3,
          }}
        >
          Sheepshead
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 22,
            marginTop: 8,
          }}
        >
          <div
            style={{
              fontFamily: "Instrument Serif",
              fontStyle: "italic",
              fontSize: 88,
              lineHeight: 0.9,
              color: ACCENT,
            }}
          >
            AI
          </div>
          <div
            style={{
              flex: 1,
              height: 1.5,
              background: RULE_STRONG,
              marginTop: 14,
            }}
          />
        </div>
        <div
          style={{
            fontFamily: "Geist",
            fontSize: 26,
            lineHeight: 1.4,
            color: INK_SOFT,
            marginTop: 34,
            maxWidth: 560,
            textWrap: "balance",
          }}
        >
          Play live with friends, strangers, and a deep neural network.
        </div>
      </div>
      <div
        style={{
          width: 470,
          height: 520,
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
      ],
    },
  );
}
