import React from 'react';
import { parseCard, suitSymbol } from './cardUtils';

interface CardTextProps {
  children: string;
  className?: string;
}

/**
 * CardText component replaces card strings in text with proper suit symbols and colors.
 * It's a drop-in replacement for plain text that automatically formats cards like "QC" → "Q♣"
 *
 * Usage: <CardText>PLAY QC</CardText> → "PLAY Q♣" (with proper red/black coloring)
 */
export default function CardText({ children, className }: CardTextProps) {
  // Regex to match card patterns: 10[CSHD] or [AKQJ987][CSHD]
  const cardPattern = /\b(?:10[CSHD]|[AKQJ987][CSHD])\b/g;

  // Split text by cards and process each part
  const parts = children.split(cardPattern);
  const matches = children.match(cardPattern) || [];

  const elements: React.ReactNode[] = [];

  for (let i = 0; i < parts.length; i++) {
    // Add the text part
    if (parts[i]) {
      elements.push(parts[i]);
    }

    // Add the formatted card if there's a match
    if (i < matches.length) {
      const cardStr = matches[i];
      const { rank, suit } = parseCard(cardStr);
      const isRed = suit === 'H' || suit === 'D';

      elements.push(
        <span
          key={`card-${i}`}
          style={{
            color: isRed ? '#dc2626' : '#1f2937', // red for hearts/diamonds, dark gray for clubs/spades
            fontWeight: '500'
          }}
        >
          {rank}{suitSymbol(suit)}
        </span>
      );
    }
  }

  return <span className={className}>{elements}</span>;
}
