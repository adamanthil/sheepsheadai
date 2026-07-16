// Deck constants for the scenario builder, mirroring sheepshead/game.py
// (TRUMP order = power order; FAIL grouped by suit).

export const TRUMP = [
  "QC",
  "QS",
  "QH",
  "QD",
  "JC",
  "JS",
  "JH",
  "JD",
  "AD",
  "10D",
  "KD",
  "9D",
  "8D",
  "7D",
] as const;

export const FAIL_BY_SUIT: Record<string, string[]> = {
  Clubs: ["AC", "10C", "KC", "9C", "8C", "7C"],
  Spades: ["AS", "10S", "KS", "9S", "8S", "7S"],
  Hearts: ["AH", "10H", "KH", "9H", "8H", "7H"],
};

export const DECK: string[] = [
  ...TRUMP,
  ...FAIL_BY_SUIT.Clubs,
  ...FAIL_BY_SUIT.Spades,
  ...FAIL_BY_SUIT.Hearts,
];

/** Sort card codes into DECK (power) order. */
export function sortByDeckOrder(cards: string[]): string[] {
  return [...cards].sort((a, b) => DECK.indexOf(a) - DECK.indexOf(b));
}
