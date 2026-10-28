"use client";

import React, { useEffect, useId, useRef } from "react";
import { ds } from "../../lib/ds";
import styles from "./ModalSheet.module.css";

/** A broadsheet modal: a native <dialog> opened on mount, with an overline,
 * a title and a close button over its content. Escape, the close button and
 * a click on the dimmed margin all close it, after which onClose runs; the
 * owner unmounts it from there. Phones get the whole screen. */
export default function ModalSheet({
  overline,
  title,
  onClose,
  narrowSheet,
  children,
}: {
  overline: string;
  title: string;
  onClose: () => void;
  /** A smaller sheet on wide screens, for content that needs less room. */
  narrowSheet?: boolean;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  const close = () => ref.current?.close();

  return (
    <dialog
      ref={ref}
      className={styles.dialog}
      aria-labelledby={titleId}
      onClose={onClose}
      // The dialog element itself is the dimmed area around the sheet.
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div
        className={`${styles.sheet} ${narrowSheet ? styles.narrowSheet : ""}`}
      >
        <header className={styles.header}>
          <div>
            <div className={ds.overline}>{overline}</div>
            <h2 id={titleId} className={styles.title}>
              {title}
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
        {children}
      </div>
    </dialog>
  );
}
