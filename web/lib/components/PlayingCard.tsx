import React from 'react';
import { parseCard, suitSymbol } from './cardUtils';
import cardStyles from './PlayingCard.module.css';

export interface PlayingCardProps {
  label: string;
  small?: boolean;
  highlight?: boolean;
  width?: number;
  height?: number;
  bigMarks?: boolean;
}

export default function PlayingCard({
  label,
  small,
  highlight,
  width,
  height,
  bigMarks
}: PlayingCardProps) {
  const blank = label === '__' || !label;
  const { rank, suit } = parseCard(label);
  const w = width ?? (small ? 48 : 76);
  const h = height ?? (small ? 64 : 108);
  const red = suit === 'H' || suit === 'D';
  const sizeClass = bigMarks ? 'big' : (small ? 'small' : 'normal');
  const classNames = [
    cardStyles.card,
    highlight ? cardStyles.highlight : '',
    blank ? cardStyles.blank : '',
    red ? cardStyles.redText : ''
  ].filter(Boolean).join(' ');

  return (
    <div
      className={classNames}
      style={{
        ['--w' as any]: `${w}px`,
        ['--h' as any]: `${h}px`,
        ['--pad' as any]: small ? '4px' : '8px'
      }}
    >
      <div className={cardStyles.topSection}>
        <div className={`${cardStyles.rankTop} ${cardStyles[sizeClass as 'small'|'normal'|'big']}`}>
          {rank}
        </div>
        <div className={`${cardStyles.suitTopLeft} ${cardStyles[sizeClass as 'small'|'normal'|'big']}`}>
          {suitSymbol(suit)}
        </div>
      </div>
      <div className={`${cardStyles.suitCenter} ${cardStyles[sizeClass as 'small'|'normal'|'big']}`}>
        {suitSymbol(suit)}
      </div>
      <div className={`${cardStyles.rankBottom} ${cardStyles[sizeClass as 'small'|'normal'|'big']}`}>
        {rank}
      </div>
    </div>
  );
}
