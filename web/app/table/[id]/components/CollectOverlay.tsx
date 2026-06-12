import React, { useRef, useEffect } from 'react';
import { PlayingCard } from '../../../../lib/ds';
import { collectAnchorPct, relSeat } from '../lib/seatLayout';
import styles from './Stage.module.css';

interface CollectOverlayProps {
  containerRef: React.RefObject<HTMLDivElement | null>;
  yourSeat: number;
  winner: number;
  cards: string[];
  cardW: number;
}

/**
 * Animates the just-completed trick's cards flying to the winner's seat.
 * Seat anchors come from seatLayout.collectAnchorPct so they track the stage.
 */
export default function CollectOverlay({ containerRef, yourSeat, winner, cards, cardW }: CollectOverlayProps) {
  const cw = containerRef.current?.clientWidth ?? 1000;
  const ch = containerRef.current?.clientHeight ?? 400;
  const refs = useRef<Array<HTMLDivElement | null>>([]);
  if (refs.current.length !== cards.length) {
    refs.current = Array(cards.length).fill(null);
  }

  const pToPx = (pct: { left: number; top: number }) => ({
    x: (pct.left / 100) * cw,
    y: (pct.top / 100) * ch,
  });

  useEffect(() => {
    const start = performance.now();
    const dur = 1100;
    const bezier = (t: number, p0: number, c1: number, c2: number, p3: number) => {
      const it = 1 - t;
      return it * it * it * p0 + 3 * it * it * t * c1 + 3 * it * t * t * c2 + t * t * t * p3;
    };
    const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

    const fromPts = cards.map((_, idx) => pToPx(collectAnchorPct(relSeat(idx + 1, yourSeat))));
    const toPt = pToPx(collectAnchorPct(relSeat(winner, yourSeat)));
    const ctrl = fromPts.map((p0) => ({
      cx1: (p0.x + cw / 2) / 2,
      cy1: Math.min(p0.y, toPt.y) - 0.15 * ch,
      cx2: (toPt.x + cw / 2) / 2,
      cy2: Math.min(p0.y, toPt.y) - 0.15 * ch,
    }));

    let raf = 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / dur);
      const et = easeInOut(t);
      cards.forEach((_, idx) => {
        const el = refs.current[idx];
        if (!el) return;
        const p0 = fromPts[idx];
        const c = ctrl[idx];
        const x = bezier(et, p0.x, c.cx1, c.cx2, toPt.x);
        const y = bezier(et, p0.y, c.cy1, c.cy2, toPt.y);
        el.style.left = `${Math.round(x)}px`;
        el.style.top = `${Math.round(y)}px`;
        el.style.transform = `translate(-50%, -50%) scale(${1 - 0.3 * et})`;
        el.style.opacity = String(1 - 0.9 * et);
      });
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [cards, yourSeat, winner, cw, ch]);

  return (
    <div className={styles.collectOverlay}>
      {cards.map((c, idx) => (
        <div key={idx} ref={(el) => { refs.current[idx] = el; }} className={styles.collectCard}>
          <PlayingCard code={c || '__'} w={cardW} />
        </div>
      ))}
    </div>
  );
}
