import React from 'react';

const CARDS = [
  { rank: 'Q', sym: '♣', red: false, rot: -12 },
  { rank: 'J', sym: '♦', red: true,  rot: 0 },
  { rank: 'A', sym: '♥', red: true,  rot: 12 },
];

/**
 * Brand mark — three fanned miniature cards (Q♣ · J♦ · A♥), the highest trump,
 * the partner card, and a fail ace. Fanned like a held hand so the collapsed
 * headers stay cohesive with the home masthead and the table.
 */
export default function MiniCardMark({ h = 24 }: { h?: number }) {
  const w = Math.max(12, Math.round(h / 1.45));
  const overlap = Math.round(w * 0.55);
  return (
    <span aria-hidden="true" style={{ display: 'inline-flex', flexShrink: 0 }}>
      {CARDS.map((c, i) => (
        <span key={c.sym} style={{
          width: w, height: h,
          marginLeft: i === 0 ? 0 : -overlap,
          background: 'var(--card-paper)',
          border: '1px solid var(--card-edge)',
          borderRadius: Math.max(2, Math.round(h * 0.09)),
          boxShadow: 'var(--shadow-1)',
          transform: `rotate(${c.rot}deg)`,
          transformOrigin: '50% 120%',
          display: 'inline-flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          lineHeight: 1, fontFamily: 'var(--font-display)',
          color: c.red ? 'var(--card-red)' : 'var(--card-black)',
        }}>
          <span style={{ fontSize: Math.round(h * 0.34), lineHeight: 1 }}>{c.rank}</span>
          <span style={{ fontSize: Math.round(h * 0.3), lineHeight: 1, marginTop: 1 }}>{c.sym}</span>
        </span>
      ))}
    </span>
  );
}
