// Card utility functions for parsing and displaying cards

export function parseCard(card: string): { rank: string; suit: 'C'|'S'|'H'|'D'|null } {
  if (!card || card === '__' || card === 'UNDER') return { rank: card, suit: null };
  if (card.startsWith('10')) return { rank: '10', suit: card.slice(2) as any };
  return { rank: card[0], suit: card[1] as any };
}

export function suitSymbol(s: 'C'|'S'|'H'|'D'|null) {
  if (s === 'C') return '♣';
  if (s === 'S') return '♠';
  if (s === 'H') return '♥';
  if (s === 'D') return '♦';
  return '';
}
