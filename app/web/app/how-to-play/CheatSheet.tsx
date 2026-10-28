"use client";

import React, { useEffect, useMemo, useState } from "react";
import { useMediaQuery } from "../../lib/ds";
import { pointGroups, trumpAndFail, type Scene } from "./lessonScenes";
import CardStage from "./CardStage";
import ModalSheet from "./ModalSheet";
import styles from "./CheatSheet.module.css";

const PAGES: { title: string; note: string; scene: (n: boolean) => Scene }[] = [
  {
    title: "Power",
    note: "Trump beats every fail card. In all suits the 10 outranks the King.",
    scene: trumpAndFail,
  },
  {
    title: "Points",
    note: "120 points in the deck. A team needs 61 to win the hand.",
    scene: pointGroups,
  },
];

/**
 * A reminder of card power and points, brought up from the table. Two pages
 * of the lesson's stage; switching between them glides the cards from one
 * arrangement to the other. Tapping or swiping the stage, or the arrow
 * keys, flip to the other page too.
 */
export default function CheatSheet({ onClose }: { onClose: () => void }) {
  const narrow = useMediaQuery("(max-width: 600px)");
  const [page, setPage] = useState(0);
  const flip = () => setPage((p) => 1 - p);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") setPage(0);
      else if (e.key === "ArrowRight") setPage(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const current = PAGES[page];
  const scene = useMemo(() => current.scene(narrow), [current, narrow]);

  return (
    <ModalSheet
      overline="Cheat sheet"
      title="Power & points"
      onClose={onClose}
      narrowSheet
    >
      <div className={styles.body}>
        <div className={styles.tabs} role="tablist" aria-label="Cheat sheet">
          {PAGES.map((p, i) => (
            <button
              key={p.title}
              type="button"
              role="tab"
              aria-selected={i === page}
              className={`${styles.tab} ${i === page ? styles.tabOn : ""}`}
              onClick={() => setPage(i)}
            >
              {p.title}
            </button>
          ))}
        </div>
        <CardStage scene={scene} narrow={narrow} onSwipe={flip} onTap={flip} />
        <p key={page} className={styles.note} aria-live="polite">
          {current.note}
        </p>
      </div>
    </ModalSheet>
  );
}
