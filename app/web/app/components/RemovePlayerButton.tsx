"use client";

import { useState } from "react";
import { ds } from "../../lib/ds";
import styles from "./RemovePlayerButton.module.css";

/** Host control to remove a player from the table. Two steps, like
 * "Close table": the first click only asks for confirmation. */
export function RemovePlayerButton({
  name,
  onRemove,
}: {
  name: string;
  onRemove: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  if (!confirming) {
    return (
      <button
        type="button"
        className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm} ${styles.danger}`}
        onClick={() => setConfirming(true)}
      >
        Remove
      </button>
    );
  }
  return (
    <span className={styles.confirm}>
      <span className={styles.prompt}>Remove {name}?</span>
      <button
        type="button"
        className={`${ds.btn} ${ds.btnSm}`}
        onClick={() => {
          setConfirming(false);
          onRemove();
        }}
      >
        Remove
      </button>
      <button
        type="button"
        className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
        onClick={() => setConfirming(false)}
      >
        Cancel
      </button>
    </span>
  );
}
