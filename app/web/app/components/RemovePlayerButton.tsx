"use client";

import { useState } from "react";
import { ds } from "../../lib/ds";
import { useBusy } from "../../lib/hooks/useBusy";
import styles from "./RemovePlayerButton.module.css";

/** Host control to remove a player from the table. Two steps, like
 * "Close table": the first click only asks for confirmation. */
export function RemovePlayerButton({
  name,
  onRemove,
}: {
  name: string;
  /** Resolves false when the server refused; on success the player's row
   * goes away with the server's update. */
  onRemove: () => Promise<boolean>;
}) {
  const [confirming, setConfirming] = useState(false);
  const removing = useBusy();
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
        className={`${ds.btn} ${ds.btnSm} ${removing.busy ? ds.btnBusy : ""}`}
        onClick={() =>
          removing.run(
            async () => {
              const ok = await onRemove();
              if (!ok) setConfirming(false);
              return ok;
            },
            { hold: true },
          )
        }
        disabled={removing.busy}
        aria-busy={removing.busy}
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
