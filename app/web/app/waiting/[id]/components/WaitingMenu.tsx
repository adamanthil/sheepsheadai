import React, { useState } from "react";
import { ds } from "../../../../lib/ds";
import BottomSheet, { SheetIcon } from "../../../components/BottomSheet";
import styles from "../../../components/BottomSheet.module.css";

interface WaitingMenuProps {
  roomName: string;
  shortId: string;
  hostName: string | null;
  onLeave: () => void;
  /** Host only. */
  onCloseTable: (() => void) | null;
  /** The close is waiting on the server. */
  closing: boolean;
  onClose: () => void;
}

const ICONS = {
  leave: "M8 4H4v12h4M12 6l4 4-4 4M16 10H8",
  close: "M7.5 7.5l5 5M12.5 7.5l-5 5",
};

/** Mobile waiting-room menu: the room's details, Leave, and the host's
 * Close table. */
export default function WaitingMenu({
  roomName,
  shortId,
  hostName,
  onLeave,
  onCloseTable,
  closing,
  onClose,
}: WaitingMenuProps) {
  const [confirmClose, setConfirmClose] = useState(false);
  return (
    <BottomSheet label="Waiting room menu" onClose={onClose}>
      <div className={styles.head}>
        <span className={styles.title}>{roomName}</span>
        <span className={styles.note}>
          {shortId}
          {hostName && ` · Host: ${hostName}`}
        </span>
      </div>
      <button type="button" className={styles.row} onClick={onLeave}>
        <SheetIcon d={ICONS.leave} />
        <span className={styles.grow}>Leave table</span>
      </button>
      {onCloseTable &&
        (confirmClose ? (
          <button
            type="button"
            className={`${styles.row} ${styles.danger} ${closing ? ds.btnBusy : ""}`}
            onClick={onCloseTable}
            disabled={closing}
            aria-busy={closing}
          >
            <SheetIcon d={ICONS.close} ring />
            <span className={styles.grow}>Confirm close</span>
            <span className={styles.note}>for everyone</span>
          </button>
        ) : (
          <button
            type="button"
            className={`${styles.row} ${styles.danger}`}
            onClick={() => setConfirmClose(true)}
          >
            <SheetIcon d={ICONS.close} ring />
            <span className={styles.grow}>Close table</span>
          </button>
        ))}
      <button
        type="button"
        className={`${styles.row} ${styles.cancel}`}
        onClick={onClose}
      >
        <span className={styles.grow}>Back to the waiting room</span>
      </button>
    </BottomSheet>
  );
}
