import React from "react";
import { SeatAvatar, ds } from "../../../../lib/ds";
import styles from "./SeatCard.module.css";

export interface SeatInfo {
  seat: number;
  name: string | null;
  isAI: boolean;
}

interface SeatCardProps {
  seat: SeatInfo;
  variant: "card" | "row";
  onTake: (seat: number) => void;
}

/**
 * A waiting-room seat. `card` is the desktop 5-up grid tile; `row` is the
 * mobile single-line variant. Empty seats (and AI seats, which a human may
 * take over) expose a "take this seat" affordance.
 */
export default function SeatCard({ seat, variant, onTake }: SeatCardProps) {
  const empty = !seat.name;
  const takeable = empty || seat.isAI;

  if (variant === "row") {
    if (empty) {
      return (
        <button
          className={`${styles.row} ${styles.rowEmpty}`}
          onClick={() => onTake(seat.seat)}
        >
          <div className={styles.rowPlus}>+</div>
          <div className={styles.rowMain}>
            <div className={ds.overline} style={{ fontSize: 9 }}>
              Seat {seat.seat}
            </div>
            <div className={styles.rowTake}>Take this seat</div>
          </div>
          <div className={styles.rowArrow}>→</div>
        </button>
      );
    }
    return (
      <div className={styles.row}>
        <SeatAvatar name={seat.name ?? undefined} isAI={seat.isAI} size={36} />
        <div className={styles.rowMain}>
          <div className={styles.rowNameLine}>
            <div className={styles.rowName}>{seat.name}</div>
            <div className={ds.overline} style={{ fontSize: 9 }}>
              seat {seat.seat}
            </div>
          </div>
          <div className={styles.rowBadges}>
            {seat.isAI && (
              <span
                className={`${ds.badge} ${ds.badgeQuiet}`}
                style={{ fontSize: 9 }}
              >
                AI bot
              </span>
            )}
          </div>
        </div>
        {seat.isAI && (
          <button
            className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
            onClick={() => onTake(seat.seat)}
          >
            Take →
          </button>
        )}
      </div>
    );
  }

  // Desktop card
  if (empty) {
    return (
      <div className={`${styles.card} ${styles.cardEmpty}`}>
        <div className={ds.overline} style={{ alignSelf: "flex-start" }}>
          Seat {seat.seat}
        </div>
        <div className={styles.emptyLabel}>Empty</div>
        <button
          className={`${ds.btn} ${ds.btnSm} ${styles.takeBtn}`}
          onClick={() => onTake(seat.seat)}
        >
          Take this seat →
        </button>
      </div>
    );
  }
  return (
    <div className={`${ds.panel} ${styles.card}`}>
      <div className={styles.cardHead}>
        <div className={ds.overline}>Seat {seat.seat}</div>
      </div>
      <div className={styles.cardBody}>
        <SeatAvatar name={seat.name ?? undefined} isAI={seat.isAI} size={44} />
        <div className={styles.name}>{seat.name}</div>
        {seat.isAI && (
          <span className={`${ds.badge} ${ds.badgeQuiet}`}>AI Bot</span>
        )}
      </div>
      {takeable ? (
        <button
          className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm} ${styles.takeBtn}`}
          onClick={() => onTake(seat.seat)}
        >
          Take over →
        </button>
      ) : (
        <span />
      )}
    </div>
  );
}
