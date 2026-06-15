import React, { useRef, useEffect } from "react";
import { PlayingCard } from "../../../../lib/ds";
import { relSeat, type RingAnchor } from "../lib/seatLayout";
import styles from "./Stage.module.css";

interface CollectOverlayProps {
  yourSeat: number;
  winner: number;
  cards: string[];
  cardW: number;
  // The ring anchors for the active layout (desktop vs mobile). Cards fly from
  // each seat's anchor to the winner's, so these must match the rendered ring.
  anchors: Record<number, RingAnchor>;
}

/**
 * Animates the just-completed trick's cards flying to the winner's seat. The
 * overlay measures its own box (which fills the same positioned ancestor the
 * ring seats are placed in), so the percentage anchors convert against the
 * exact coordinate space the cards were rendered in.
 */
export default function CollectOverlay({
  yourSeat,
  winner,
  cards,
  cardW,
  anchors,
}: CollectOverlayProps) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const refs = useRef<Array<HTMLDivElement | null>>([]);
  if (refs.current.length !== cards.length) {
    refs.current = Array(cards.length).fill(null);
  }

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const cw = root.clientWidth || 1;
    const ch = root.clientHeight || 1;
    const pToPx = (rel: number) => {
      const a = anchors[rel] ?? { cardX: 50, cardY: 50 };
      return { x: (a.cardX / 100) * cw, y: (a.cardY / 100) * ch };
    };

    const bezier = (
      t: number,
      p0: number,
      c1: number,
      c2: number,
      p3: number,
    ) => {
      const it = 1 - t;
      return (
        it * it * it * p0 +
        3 * it * it * t * c1 +
        3 * it * t * t * c2 +
        t * t * t * p3
      );
    };
    const easeInOut = (t: number) =>
      t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

    const fromPts = cards.map((_, idx) => pToPx(relSeat(idx + 1, yourSeat)));
    const toPt = pToPx(relSeat(winner, yourSeat));
    const ctrl = fromPts.map((p0) => ({
      cx1: (p0.x + cw / 2) / 2,
      cy1: Math.min(p0.y, toPt.y) - 0.15 * ch,
      cx2: (toPt.x + cw / 2) / 2,
      cy2: Math.min(p0.y, toPt.y) - 0.15 * ch,
    }));

    // Seat the cards at their start points before the first frame to avoid a
    // top-left flash.
    cards.forEach((_, idx) => {
      const el = refs.current[idx];
      if (!el) return;
      el.style.left = `${Math.round(fromPts[idx].x)}px`;
      el.style.top = `${Math.round(fromPts[idx].y)}px`;
      el.style.transform = "translate(-50%, -50%)";
    });

    const start = performance.now();
    const dur = 1100;
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
  }, [cards, yourSeat, winner, cardW, anchors]);

  return (
    <div className={styles.collectOverlay} ref={rootRef}>
      {cards.map((c, idx) => (
        <div
          key={idx}
          ref={(el) => {
            refs.current[idx] = el;
          }}
          className={styles.collectCard}
        >
          <PlayingCard code={c || "__"} w={cardW} />
        </div>
      ))}
    </div>
  );
}
