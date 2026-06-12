import React from 'react';
import { parseCard, suitSymbol, isRedSuit } from './cardUtils';

interface CardTextProps {
  children: string;
  className?: string;
}

/**
 * Replaces card codes inside a text string with formatted suit symbols and
 * theme colors. Drop-in for action labels: <CardText>PLAY QC</CardText> → "PLAY Q♣".
 */
export default function CardText({ children, className }: CardTextProps) {
  const cardPattern = /\b(?:10[CSHD]|[AKQJ987][CSHD])\b/g;
  const parts = children.split(cardPattern);
  const matches = children.match(cardPattern) ?? [];

  const elements: React.ReactNode[] = [];
  for (let i = 0; i < parts.length; i++) {
    if (parts[i]) elements.push(parts[i]);
    if (i < matches.length) {
      const { rank, suit } = parseCard(matches[i]);
      elements.push(
        <span
          key={`card-${i}`}
          style={{ color: isRedSuit(suit) ? 'var(--card-red)' : 'var(--ink)', fontWeight: 500 }}
        >
          {rank}{suitSymbol(suit)}
        </span>
      );
    }
  }

  return <span className={className}>{elements}</span>;
}
