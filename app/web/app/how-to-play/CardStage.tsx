"use client";

import React, { useEffect, useRef, useState } from "react";
import { PlayingCard, ds } from "../../lib/ds";
import {
  CARD_W,
  DECK,
  STAGE_H,
  type CardPlacement,
  type LabelKind,
  type Scene,
} from "./lessonScenes";
import styles from "./CardStage.module.css";

/** Cards cascade into place in scene order, capped so big moves don't drag. */
const STAGGER_MS = 14;
const MAX_STAGGER_MS = 420;

/** Swipes shorter than this count as a tap on the stage. */
const SWIPE_PX = 40;

const LABEL_CLASS: Record<LabelKind, string> = {
  overline: `${ds.overline} ${styles.labelOverline}`,
  tag: ds.badge,
  tagTrump: `${ds.badge} ${ds.badgeAccent}`,
  tagRed: `${ds.badge} ${ds.badgeAccent2}`,
  big: styles.labelBig,
};

const TONE_CLASS = {
  hi: styles.hi,
  trump: styles.trump,
  dim: styles.dim,
} as const;

/** Stage coordinates as fixed-precision strings: the server and the browser
 * serialize long floats differently, which breaks hydration of the page. */
const fixed = (v: number) => v.toFixed(3);

/** A card named in lesson text with a red suit, e.g. "A♥" or "10♦". */
const RED_CARD = /((?:10|[2-9AKQJ])[♥♦])/;

/** Text with its red-suit cards set in the cards' red, matching the
 * faces on the stage. Splitting on a capturing group puts the cards at the
 * odd indices. */
export const redSuits = (text: string) =>
  text.split(RED_CARD).map((part, i) =>
    i % 2 ? (
      <span key={i} className={styles.redCard}>
        {part}
      </span>
    ) : (
      part
    ),
  );

/**
 * One stage of all 32 cards that glide from scene to scene, on a white
 * plate like a printed figure. The stage is a size container: scene
 * coordinates are in percent of its width, rendered as cqw, so every card
 * and label scales with it without any measuring.
 * Children sit over the stage in the same coordinates (the lesson's call to
 * action), outside the aria-hidden card layer.
 */
export default function CardStage({
  scene,
  narrow,
  onSwipe,
  onTap,
  children,
}: {
  scene: Scene;
  narrow: boolean;
  /** A horizontal swipe: 1 for onward (leftward), -1 for back. */
  onSwipe?: (dir: 1 | -1) => void;
  onTap?: () => void;
  children?: React.ReactNode;
}) {
  // Placements land without transitions until the first narrow/wide answer
  // is in, so a phone doesn't watch the deck slide from the wide layout.
  const [settled, setSettled] = useState(false);
  useEffect(() => {
    let inner = 0;
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => setSettled(true));
    });
    return () => {
      cancelAnimationFrame(outer);
      cancelAnimationFrame(inner);
    };
  }, []);

  // Cards a scene leaves out fade where they last stood rather than flying
  // off to some parking spot.
  const lastPlaced = useRef<Record<string, CardPlacement>>({});
  const stageH = narrow ? STAGE_H.narrow : STAGE_H.wide;
  const order = Object.keys(scene.cards);
  const cards = DECK.map((code) => {
    const placed = scene.cards[code];
    return {
      code,
      p: placed ?? lastPlaced.current[code] ?? { x: 50, y: stageH / 2 },
      hidden: !placed,
      order: placed ? order.indexOf(code) : -1,
    };
  });
  useEffect(() => {
    Object.assign(lastPlaced.current, scene.cards);
  }, [scene]);

  const swipeStart = useRef<number | null>(null);
  const onPointerDown = (e: React.PointerEvent) => {
    swipeStart.current = e.clientX;
  };
  const onPointerUp = (e: React.PointerEvent) => {
    if (swipeStart.current === null) return;
    const dx = e.clientX - swipeStart.current;
    swipeStart.current = null;
    if (dx <= -SWIPE_PX) onSwipe?.(1);
    else if (dx >= SWIPE_PX) onSwipe?.(-1);
    else onTap?.();
  };
  const interactive = Boolean(onSwipe || onTap);

  const stageVars = {
    "--h": stageH,
    "--cw": narrow ? CARD_W.narrow : CARD_W.wide,
  } as React.CSSProperties;

  return (
    <div className={styles.frame}>
      <div className={`${styles.stageBox} ${narrow ? styles.narrow : ""}`}>
        <div
          className={[
            styles.stage,
            settled ? "" : styles.still,
            interactive ? styles.interactive : "",
          ]
            .filter(Boolean)
            .join(" ")}
          style={stageVars}
          onPointerDown={interactive ? onPointerDown : undefined}
          onPointerUp={interactive ? onPointerUp : undefined}
          onPointerCancel={() => (swipeStart.current = null)}
          aria-hidden="true"
        >
          {cards.map(({ code, p, hidden, order: o }) => {
            const slotVars = {
              "--x": fixed(p.x),
              "--y": fixed(p.y),
              "--r": `${fixed(p.r ?? 0)}deg`,
              "--d": `${hidden ? 0 : Math.min(o * STAGGER_MS, MAX_STAGGER_MS)}ms`,
              zIndex: hidden ? 0 : o + 1,
            } as React.CSSProperties;
            const classes = [
              styles.slot,
              hidden ? styles.hidden : "",
              !hidden && p.tone ? TONE_CLASS[p.tone] : "",
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <div key={code} className={classes} style={slotVars}>
                <div
                  className={`${styles.flip} ${p.faceDown ? styles.flipped : ""}`}
                >
                  <PlayingCard code={code} className={styles.face} ariaHidden />
                  <PlayingCard code="__" className={styles.back} ariaHidden />
                </div>
              </div>
            );
          })}
          {scene.labels.map((l) => (
            <div
              key={`${l.id}:${l.text}`}
              className={`${styles.label} ${l.align === "start" ? styles.alignStart : ""}`}
              style={
                {
                  "--x": fixed(l.x),
                  "--y": fixed(l.y),
                } as React.CSSProperties
              }
            >
              <span className={LABEL_CLASS[l.kind]}>{redSuits(l.text)}</span>
            </div>
          ))}
        </div>
        {children}
      </div>
    </div>
  );
}
