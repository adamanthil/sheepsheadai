import React, { useMemo } from "react";
import { CardText, SeatAvatar, ds } from "../../../../lib/ds";
import type { YourRole } from "../lib/phase";
import { roleBadge } from "./stage/chrome";
import styles from "./ActionBar.module.css";

/** Seconds left on the acting player's turn clock, red for the last 5. */
export function TurnClock({ secondsLeft }: { secondsLeft: number | null }) {
  if (secondsLeft === null) return null;
  return (
    <span
      className={`${styles.clock} ${secondsLeft <= 5 ? styles.clockUrgent : ""}`}
      aria-label={`${secondsLeft} seconds left to move`}
    >
      {secondsLeft}s
    </span>
  );
}

interface ActionBarProps {
  yourName: string;
  /** Your verified account username, for the avatar's seal. */
  yourAccount: string | null;
  yourSeat: number;
  yourRole: YourRole;
  isYourTurn: boolean;
  actorName: string;
  helper: string;
  validActions: number[];
  actionLookup: Record<string, string>;
  onTakeAction: (id: number) => void;
  hasLastTrick: boolean;
  showPrev: boolean;
  onTogglePrev: () => void;
  onShowScores: () => void;
  isHost: boolean;
  confirmClose: boolean;
  onConfirmClose: (v: boolean) => void;
  onCloseTable: () => void;
  /** A hand is being played, so the table can close once it ends. */
  handInPlay: boolean;
  closingAfterHand: boolean;
  onCloseAfterHand: (on: boolean) => void;
  isMobile: boolean;
  secondsLeft: number | null;
}

export default function ActionBar(props: ActionBarProps) {
  const { validActions, actionLookup } = props;

  // Primary action buttons: everything that isn't a card action (PLAY/BURY/
  // UNDER are taken by tapping cards and confirming in the center), plus the
  // explicit PLAY UNDER.
  const actionButtons = useMemo(
    () =>
      validActions
        .map((aid) => ({ id: aid, label: actionLookup[String(aid)] }))
        .filter(
          (a) =>
            a.label &&
            ((!a.label.startsWith("PLAY") &&
              !a.label.startsWith("BURY") &&
              !a.label.startsWith("UNDER ")) ||
              a.label === "PLAY UNDER"),
        ),
    [validActions, actionLookup],
  );

  const utils = (
    <div className={styles.utilRow}>
      {props.hasLastTrick && (
        <button
          className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
          onClick={props.onTogglePrev}
        >
          {props.showPrev ? "Hide prev" : "Show prev"}
        </button>
      )}
      <button
        className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
        onClick={props.onShowScores}
      >
        Scores
      </button>
      {/* Its own group, set off from Scores: once expanded, its buttons
          read as one choice rather than more of the row. */}
      {props.isHost && (
        <span className={styles.closeGroup}>
          {props.closingAfterHand ? (
            <>
              <span className={styles.armed}>Table ends after this hand</span>
              <button
                className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
                onClick={() => props.onCloseAfterHand(false)}
              >
                Undo
              </button>
            </>
          ) : props.confirmClose ? (
            <>
              {props.handInPlay && (
                <button
                  className={`${ds.btn} ${ds.btnSm}`}
                  onClick={() => {
                    props.onConfirmClose(false);
                    props.onCloseAfterHand(true);
                  }}
                >
                  End after this hand
                </button>
              )}
              <button
                className={`${ds.btn} ${ds.btnSm} ${styles.dangerBtn}`}
                onClick={props.onCloseTable}
              >
                {props.handInPlay ? "Close now" : "Confirm close"}
              </button>
              <button
                className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
                onClick={() => props.onConfirmClose(false)}
              >
                Cancel
              </button>
            </>
          ) : (
            <button
              className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm} ${styles.dangerLink}`}
              onClick={() => props.onConfirmClose(true)}
            >
              Close table
            </button>
          )}
        </span>
      )}
    </div>
  );

  const primaries = actionButtons.map((a) => {
    const accent =
      a.label === "PICK" ||
      a.label.startsWith("CALL") ||
      a.label === "ALONE" ||
      a.label === "PLAY UNDER" ||
      a.label.startsWith("CONFIRM");
    return (
      <button
        key={a.id}
        className={`${ds.btn} ${accent ? ds.btnAccent : ds.btnGhost}`}
        onClick={() => props.onTakeAction(a.id)}
      >
        <CardText>{a.label}</CardText>
      </button>
    );
  });

  if (props.isMobile) {
    return (
      <div className={styles.mob}>
        <div className={styles.mobHelper}>
          {props.isYourTurn ? props.helper : `Waiting for ${props.actorName}…`}
          <TurnClock secondsLeft={props.secondsLeft} />
        </div>
        <div className={styles.mobRow}>{primaries}</div>
        <div className={styles.mobRow}>{utils}</div>
      </div>
    );
  }

  return (
    <div className={styles.desk}>
      <div className={styles.deskWho}>
        <SeatAvatar
          name={props.yourName}
          account={props.yourAccount}
          size={32}
          tone="you"
        />
        <div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
            <div className={styles.whoName}>{props.yourName}</div>
            <span className={ds.badge} style={{ fontSize: 9 }}>
              You
            </span>
            {props.yourRole && roleBadge(props.yourRole, false, 9)}
          </div>
          <div className={styles.whoSub}>
            seat {props.yourSeat}
            {props.isYourTurn ? " · your move" : ""}
          </div>
        </div>
      </div>
      <div className={styles.deskRight}>
        {props.isYourTurn ? (
          <span className={styles.helper}>{props.helper}</span>
        ) : (
          <span className={styles.waiting}>Waiting for {props.actorName}…</span>
        )}
        <TurnClock secondsLeft={props.secondsLeft} />
        {primaries}
        {utils}
      </div>
    </div>
  );
}
