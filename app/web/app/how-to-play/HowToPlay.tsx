"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { PlayingCard, ds, useMediaQuery } from "../../lib/ds";
import {
  CARD_W,
  CHAPTERS,
  DECK,
  STAGE_H,
  type CardPlacement,
  type LabelKind,
} from "./lessonScenes";
import DealMeIn from "./DealMeIn";
import styles from "./HowToPlay.module.css";

const BEATS = CHAPTERS.flatMap((chapter, c) =>
  chapter.beats.map((beat, i) => ({ ...beat, chapter: c, index: i })),
);
const FIRST_BEAT = CHAPTERS.map((_, c) =>
  BEATS.findIndex((b) => b.chapter === c),
);

/** How long autoplay holds a beat: a beat of animation plus reading time. */
const beatMs = (caption: string) => 2600 + caption.length * 45;

/** Cards cascade into place in scene order, capped so big moves don't drag. */
const STAGGER_MS = 14;
const MAX_STAGGER_MS = 420;

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
const redSuits = (text: string) =>
  text.split(RED_CARD).map((part, i) =>
    i % 2 ? (
      <span key={i} className={styles.redCard}>
        {part}
      </span>
    ) : (
      part
    ),
  );

/** Swipes shorter than this count as a tap on the stage. */
const SWIPE_PX = 40;

/**
 * The How to Play lesson as a chapter player: one stage of all 32 cards that
 * glide between scenes, a caption under it, and a chapter rail that doubles
 * as the progress bar. Autoplay is driven by the active progress segment's
 * CSS animation, so pausing freezes the bar exactly where it is.
 */
export default function HowToPlay() {
  const narrow = useMediaQuery("(max-width: 600px)");
  const reducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  const [beat, setBeat] = useState(0);
  const [playing, setPlaying] = useState(true);
  // Placements land without transitions until the first narrow/wide answer
  // is in, so a phone doesn't watch the deck slide from the wide layout.
  const [settled, setSettled] = useState(false);

  const current = BEATS[beat];
  const chapter = CHAPTERS[current.chapter];
  const atEnd = beat === BEATS.length - 1;

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

  useEffect(() => {
    if (reducedMotion) setPlaying(false);
  }, [reducedMotion]);

  // Deep links: /how-to-play#following opens on that chapter.
  useEffect(() => {
    const id = window.location.hash.slice(1);
    const c = CHAPTERS.findIndex((ch) => ch.id === id);
    if (c > 0) setBeat(FIRST_BEAT[c]);
  }, []);

  const go = useCallback((next: number) => {
    setBeat(Math.max(0, Math.min(BEATS.length - 1, next)));
  }, []);

  const advance = useCallback(() => {
    if (atEnd) setPlaying(false);
    else setBeat((b) => b + 1);
  }, [atEnd]);

  const togglePlay = useCallback(() => {
    if (atEnd && !playing) {
      setBeat(0);
      setPlaying(true);
    } else {
      setPlaying((p) => !p);
    }
  }, [atEnd, playing]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        go(beat + 1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        go(beat - 1);
      } else if (e.key === " " && !(e.target instanceof HTMLButtonElement)) {
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [beat, go, togglePlay]);

  const scene = useMemo(() => current.scene(narrow), [current, narrow]);

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
    if (dx <= -SWIPE_PX) go(beat + 1);
    else if (dx >= SWIPE_PX) go(beat - 1);
    else togglePlay();
  };

  const stageVars = {
    "--h": stageH,
    "--cw": narrow ? CARD_W.narrow : CARD_W.wide,
  } as React.CSSProperties;

  return (
    <div className={`${styles.player} ${narrow ? styles.narrow : ""}`}>
      <div className={styles.frame}>
        <div className={styles.stageBox}>
          <div
            className={`${styles.stage} ${settled ? "" : styles.still}`}
            style={stageVars}
            onPointerDown={onPointerDown}
            onPointerUp={onPointerUp}
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
                    <PlayingCard
                      code={code}
                      className={styles.face}
                      ariaHidden
                    />
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
          {scene.cta && <DealMeIn key={beat} {...scene.cta} />}
        </div>
      </div>

      <div className={styles.captionBlock} aria-live="polite">
        <div className={ds.overline}>
          §{current.chapter + 1} · {chapter.title}
        </div>
        <p key={beat} className={styles.caption}>
          {redSuits(current.caption)}
        </p>
      </div>

      <div className={styles.controls}>
        <div className={styles.transport}>
          <button
            type="button"
            className={styles.iconBtn}
            onClick={() => go(beat - 1)}
            disabled={beat === 0}
            aria-label="Previous"
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M10 3 5 8l5 5" />
            </svg>
          </button>
          <button
            type="button"
            className={`${styles.iconBtn} ${styles.playBtn}`}
            onClick={togglePlay}
            aria-label={playing ? "Pause" : atEnd ? "Replay" : "Play"}
          >
            {playing ? (
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M5.5 3.5v9M10.5 3.5v9" />
              </svg>
            ) : atEnd ? (
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M3.5 8a4.5 4.5 0 1 0 1.4-3.3M3.5 2.5v2.6h2.6" />
              </svg>
            ) : (
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path className={styles.fill} d="M5 3.2v9.6L12.6 8z" />
              </svg>
            )}
          </button>
          <button
            type="button"
            className={styles.iconBtn}
            onClick={() => go(beat + 1)}
            disabled={atEnd}
            aria-label="Next"
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="m6 3 5 5-5 5" />
            </svg>
          </button>
        </div>

        <ol className={styles.rail}>
          {CHAPTERS.map((ch, c) => {
            const isCurrent = c === current.chapter;
            return (
              <li key={ch.id}>
                <button
                  type="button"
                  className={`${styles.chapterBtn} ${isCurrent ? styles.chapterCurrent : ""}`}
                  onClick={() => go(FIRST_BEAT[c])}
                  aria-current={isCurrent ? "step" : undefined}
                >
                  <span className={styles.segments}>
                    {ch.beats.map((b, i) => {
                      const g = FIRST_BEAT[c] + i;
                      return (
                        <span key={i} className={styles.segment}>
                          {g < beat && <span className={styles.segmentDone} />}
                          {g === beat && (
                            <span
                              key={beat}
                              className={styles.segmentActive}
                              style={
                                {
                                  "--dur": `${beatMs(b.caption)}ms`,
                                  animationPlayState: playing
                                    ? "running"
                                    : "paused",
                                } as React.CSSProperties
                              }
                              onAnimationEnd={advance}
                            />
                          )}
                        </span>
                      );
                    })}
                  </span>
                  <span className={styles.chapterTitle}>
                    <span className={styles.chapterNum}>{c + 1}</span>{" "}
                    {ch.title}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
