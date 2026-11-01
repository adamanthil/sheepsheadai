import React, { useState } from "react";
import { ds, isRedSuit, parseCard, suitSymbol } from "../../../../lib/ds";
import type { YourRole } from "../lib/phase";
import { formatClock, type HandContext } from "../lib/tableSelectors";
import { roleBadge } from "./stage/chrome";
import styles from "./TableStatus.module.css";

/** The running turn clock, on the local clock. */
export interface TurnClock {
  /** Local-clock ms when the turn runs out. */
  deadline: number;
  /** The clock's full length in seconds. */
  totalSeconds: number;
}

interface TableStatusProps {
  context: HandContext;
  clock: TurnClock | null;
  secondsLeft: number | null;
  isMobile: boolean;
  /** Null for a spectator, whose SpectatorBar says whose turn it is. */
  turn: {
    isYourTurn: boolean;
    /** What your turn asks of you, e.g. "Your play". */
    yourLabel: string;
    /** Whoever else is to act, if anyone. */
    actorName: string | null;
    role: YourRole;
    isDealer: boolean;
  } | null;
}

/**
 * The strip between the table and the hand: the hand's context line, the
 * turn clock draining as a bar, and one line saying whose turn it is. It
 * replaces the action bar; every action is taken on the table or the hand.
 */
export default function TableStatus({
  context,
  clock,
  secondsLeft,
  isMobile,
  turn,
}: TableStatusProps) {
  const yours = !!turn?.isYourTurn;
  return (
    <div className={`${styles.wrap} ${isMobile ? styles.mob : styles.desk}`}>
      <div className={`${styles.context} ${context.faded ? styles.faded : ""}`}>
        {context.parts.map((part, i) => {
          if (typeof part === "string")
            return <React.Fragment key={i}>{part}</React.Fragment>;
          const { rank, suit } = parseCard(part.card);
          return (
            <span
              key={i}
              className={`${styles.card} ${isRedSuit(suit) ? styles.red : ""}`}
            >
              {rank}
              {suitSymbol(suit)}
            </span>
          );
        })}
      </div>
      <div className={styles.track}>
        {clock && (
          <ClockFill
            key={clock.deadline}
            clock={clock}
            className={yours ? styles.fillYours : styles.fill}
          />
        )}
      </div>
      {turn && (
        <div className={styles.turn} role="status">
          {turn.role && roleBadge(turn.role, true)}
          {turn.isDealer && (
            <span className={`${ds.badge} ${ds.badgeQuiet} ${styles.dealer}`}>
              Dealer
            </span>
          )}
          {turn.isYourTurn ? (
            <span className={styles.yours}>{turn.yourLabel}</span>
          ) : (
            turn.actorName && (
              <span className={styles.waiting}>
                Waiting on {turn.actorName}
              </span>
            )
          )}
          {secondsLeft !== null && (
            <span
              className={`${styles.secs} ${secondsLeft <= 5 ? styles.urgent : ""}`}
              aria-label={`${secondsLeft} seconds left to move`}
            >
              {formatClock(secondsLeft)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/** The clock's fill, drained by a CSS animation from what is left of the
 * turn when it mounts. Keyed by the deadline, so a new turn starts afresh. */
function ClockFill({
  clock,
  className,
}: {
  clock: TurnClock;
  className: string;
}) {
  const [start] = useState(() => {
    const msLeft = Math.max(0, clock.deadline - Date.now());
    const total = Math.max(1, clock.totalSeconds * 1000);
    return { msLeft, from: Math.min(1, msLeft / total) };
  });
  return (
    <i
      className={className}
      style={
        {
          "--from": start.from,
          animationDuration: `${start.msLeft}ms`,
        } as React.CSSProperties
      }
    />
  );
}
