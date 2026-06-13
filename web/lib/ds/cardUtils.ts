// Card parsing/formatting shared across the design system.

export type Suit = "C" | "S" | "H" | "D";

export interface ParsedCard {
  rank: string;
  suit: Suit | null;
  faceDown: boolean;
  special: "UNDER" | null;
}

/** Parse a card code like "QC", "10H", "AD", "__", "UNDER". */
export function parseCard(code: string): ParsedCard {
  if (!code || code === "__")
    return { rank: "", suit: null, faceDown: true, special: null };
  if (code === "UNDER")
    return { rank: "", suit: null, faceDown: false, special: "UNDER" };
  if (code.startsWith("10"))
    return {
      rank: "10",
      suit: code.slice(2) as Suit,
      faceDown: false,
      special: null,
    };
  return {
    rank: code[0],
    suit: code[1] as Suit,
    faceDown: false,
    special: null,
  };
}

export function suitSymbol(s: Suit | null): string {
  switch (s) {
    case "C":
      return "♣";
    case "S":
      return "♠";
    case "H":
      return "♥";
    case "D":
      return "♦";
    default:
      return "";
  }
}

export function isRedSuit(s: Suit | null): boolean {
  return s === "H" || s === "D";
}
