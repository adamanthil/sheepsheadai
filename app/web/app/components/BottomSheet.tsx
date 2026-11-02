import React, { useEffect, useRef } from "react";
import styles from "./BottomSheet.module.css";

/** A mobile menu sheet that rises from the bottom over a scrim. It focuses
 * its first button on opening and closes on Escape or a tap outside; the
 * rows are the caller's, styled from BottomSheet.module.css. */
export default function BottomSheet({
  label,
  onClose,
  children,
}: {
  label: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const sheetRef = useRef<HTMLDivElement>(null);
  // Read through a ref so the page's every re-render (each socket message)
  // doesn't re-run the effect and pull focus back to the first row.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    sheetRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCloseRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <>
      <div className={styles.scrim} onClick={onClose} aria-hidden="true" />
      <div
        ref={sheetRef}
        className={styles.sheet}
        role="dialog"
        aria-modal="true"
        aria-label={label}
      >
        <div className={styles.grab} aria-hidden="true" />
        {children}
      </div>
    </>
  );
}

/** A row's 20px stroke icon: one path, optionally inside a ring. */
export function SheetIcon({ d, ring }: { d: string; ring?: boolean }) {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      {ring && <circle cx="10" cy="10" r="7" />}
      <path d={d} />
    </svg>
  );
}
