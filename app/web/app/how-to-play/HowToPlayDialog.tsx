"use client";

import React, { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { ds } from "../../lib/ds";
import HowToPlay from "./HowToPlay";
import styles from "./HowToPlayDialog.module.css";

/** The lesson as a modal over whatever page linked to it. Rendered by the
 * intercepting route, so the URL reads /how-to-play while it is open and
 * closing it is a step back in history. */
export default function HowToPlayDialog() {
  const router = useRouter();
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  const close = () => ref.current?.close();

  return (
    <dialog
      ref={ref}
      className={styles.dialog}
      aria-labelledby="how-to-play-title"
      onClose={() => router.back()}
      // The dialog element itself is the dimmed area around the sheet.
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div className={styles.sheet}>
        <header className={styles.header}>
          <div>
            <div className={ds.overline}>The rules in two minutes</div>
            <h2 id="how-to-play-title" className={styles.title}>
              How to play
            </h2>
          </div>
          <button
            type="button"
            className={styles.close}
            onClick={close}
            aria-label="Close"
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="m4 4 8 8M12 4l-8 8" />
            </svg>
          </button>
        </header>
        <HowToPlay />
      </div>
    </dialog>
  );
}
