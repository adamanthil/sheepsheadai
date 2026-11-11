"use client";

import { useEffect, useMemo, useState } from "react";
import { useMediaQuery } from "../../lib/ds";
import { pointGroups, trumpAndFail, type Scene } from "./lessonScenes";
import CardStage from "./CardStage";
import ModalSheet from "./ModalSheet";
import styles from "./QuickReference.module.css";

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

/** Which reference page is showing, with the arrow keys flipping between
 * them. Owned by whoever lays out the tabs and the stage. */
export function useReferencePage() {
  const [page, setPage] = useState(0);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") setPage(0);
      else if (e.key === "ArrowRight") setPage(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return [page, setPage] as const;
}

/** The Power / Points switch. */
export function ReferenceTabs({
  page,
  onPage,
  className = "",
}: {
  page: number;
  onPage: (page: number) => void;
  className?: string;
}) {
  return (
    <div
      className={`${styles.tabs} ${className}`}
      role="tablist"
      aria-label="Quick reference"
    >
      {PAGES.map((p, i) => (
        <button
          key={p.title}
          type="button"
          role="tab"
          aria-selected={i === page}
          className={`${styles.tab} ${i === page ? styles.tabOn : ""}`}
          onClick={() => onPage(i)}
        >
          {p.title}
        </button>
      ))}
    </div>
  );
}

/**
 * A reminder of card power and points: two pages of the lesson's stage;
 * switching between them glides the cards from one arrangement to the
 * other. Tapping or swiping the stage flips to the other page too.
 */
export function ReferenceStage({
  page,
  onPage,
}: {
  page: number;
  onPage: (page: number) => void;
}) {
  const narrow = useMediaQuery("(max-width: 600px)");
  const flip = () => onPage(1 - page);
  const current = PAGES[page];
  const scene = useMemo(() => current.scene(narrow), [current, narrow]);

  return (
    <div className={styles.body}>
      <CardStage scene={scene} narrow={narrow} onSwipe={flip} onTap={flip} />
      <p key={page} className={styles.note} aria-live="polite">
        {current.note}
      </p>
    </div>
  );
}

/** Tabs over the stage, as shown over the table. */
function QuickReferenceBody() {
  const [page, setPage] = useReferencePage();
  return (
    <div className={styles.body}>
      <ReferenceTabs page={page} onPage={setPage} />
      <ReferenceStage page={page} onPage={setPage} />
    </div>
  );
}

/** The quick reference brought up from the table. */
export default function QuickReference({ onClose }: { onClose: () => void }) {
  return (
    <ModalSheet
      overline="Quick reference"
      title="Power & points"
      onClose={onClose}
      narrowSheet
    >
      <QuickReferenceBody />
    </ModalSheet>
  );
}
