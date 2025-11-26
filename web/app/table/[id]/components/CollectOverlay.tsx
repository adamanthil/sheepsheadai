import React, { useRef, useEffect } from 'react';
import { PlayingCard } from '../../../../lib/components';
import { spotStyle } from '../utils/seatMath';
import styles from '../page.module.css';

interface CollectOverlayProps {
  containerRef: React.RefObject<HTMLDivElement | null>;
  yourSeat: number;
  winner: number;
  cards: string[];
  centerSize: { w: number; h: number };
}

export default function CollectOverlay({
  containerRef,
  yourSeat,
  winner,
  cards,
  centerSize,
}: CollectOverlayProps) {
  const cw = containerRef.current?.clientWidth ?? 1000;
  const ch = containerRef.current?.clientHeight ?? 400;

  const refs = useRef<Array<HTMLDivElement | null>>([]);
  if (refs.current.length !== cards.length) {
    refs.current = Array(cards.length).fill(null);
  }

  function pctNum(v: string | number | undefined): number {
    if (typeof v === 'number') return v;
    if (!v) return 0;
    return parseFloat(String(v).replace('%', ''));
  }

  function pToPx(pctLeft: string | number | undefined, pctTop: string | number | undefined) {
    const x = (pctNum(pctLeft) / 100) * cw;
    const y = (pctNum(pctTop) / 100) * ch;
    return { x, y };
  }

  useEffect(() => {
    const start = performance.now();
    const dur = 1100;

    function bezier(t: number, p0: number, c1: number, c2: number, p3: number) {
      const it = 1 - t;
      return it * it * it * p0 + 3 * it * it * t * c1 + 3 * it * t * t * c2 + t * t * t * p3;
    }

    function easeInOut(t: number) {
      return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    const fromPts = cards.map((_, idx) => {
      const absSeat = idx + 1;
      const r = (absSeat - yourSeat + 5) % 5;
      const st = spotStyle(r);
      return pToPx(st.left as any, st.top as any);
    });

    const toPt = (() => {
      const wr = (winner - yourSeat + 5) % 5;
      const st = spotStyle(wr);
      return pToPx(st.left as any, st.top as any);
    })();

    const ctrl = fromPts.map((p0) => {
      const cx1 = (p0.x + cw / 2) / 2;
      const cy1 = Math.min(p0.y, toPt.y) - 0.15 * ch;
      const cx2 = (toPt.x + cw / 2) / 2;
      const cy2 = Math.min(p0.y, toPt.y) - 0.15 * ch;
      return { cx1, cy1, cx2, cy2 };
    });

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
        const s = 1 - 0.3 * et;
        const o = 1 - 0.9 * et;
        el.style.left = `${Math.round(x)}px`;
        el.style.top = `${Math.round(y)}px`;
        el.style.transform = `translate(-50%, -50%) scale(${s})`;
        el.style.opacity = String(o);
      });

      if (t < 1) raf = requestAnimationFrame(step);
    };

    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [cards, yourSeat, winner, cw, ch]);

  return (
    <div className={styles.collectOverlay}>
      {cards.map((c, idx) => (
        <div
          key={idx}
          ref={(el) => {
            refs.current[idx] = el;
          }}
          className={styles.collectCard}
        >
          <PlayingCard label={c || '__'} width={centerSize.w} height={centerSize.h} />
        </div>
      ))}
    </div>
  );
}

