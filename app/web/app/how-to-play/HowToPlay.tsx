"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ds, useMediaQuery } from "../../lib/ds";
import { CHAPTERS } from "./lessonScenes";
import CardStage, { redSuits } from "./CardStage";
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

/**
 * The How to Play lesson as a chapter player: the card stage, a caption
 * under it, and a chapter rail that doubles as the progress bar. Autoplay is driven by the active progress segment's
 * CSS animation, so pausing freezes the bar exactly where it is.
 */
export default function HowToPlay() {
  const narrow = useMediaQuery("(max-width: 600px)");
  const reducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  const [beat, setBeat] = useState(0);
  const [playing, setPlaying] = useState(true);
  const current = BEATS[beat];
  const chapter = CHAPTERS[current.chapter];
  const atEnd = beat === BEATS.length - 1;

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

  return (
    <div className={`${styles.player} ${narrow ? styles.narrow : ""}`}>
      <CardStage
        scene={scene}
        narrow={narrow}
        onSwipe={(dir) => go(beat + dir)}
        onTap={togglePlay}
      >
        {scene.cta && <DealMeIn key={beat} {...scene.cta} />}
      </CardStage>

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
