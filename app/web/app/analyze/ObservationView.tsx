import React from "react";
import { AnalyzeObservation } from "../../lib/analyzeTypes";
import { CardText, PlayingCard } from "../../lib/ds";
import styles from "./ObservationView.module.css";

interface ObservationViewProps {
  observation: AnalyzeObservation;
}

/** "1=you, 2=2 to your left, 0=not decided yet." */
function relLabel(rel: number): string {
  if (rel === 0) return "—";
  if (rel === 1) return "you";
  return `${rel} left`;
}

export default function ObservationView({
  observation,
}: ObservationViewProps) {
  const {
    partnerMode,
    isLeaster,
    playStarted,
    aloneCalled,
    calledUnder,
    currentTrick,
    calledCard,
    pickerRel,
    partnerRel,
    leaderRel,
    pickerPosition,
    hand,
    blind,
    bury,
    trick,
  } = observation;

  return (
    <div className={styles.observationView}>
      <div className={styles.sectionTitle}>Model Observation</div>
      <p className={styles.subhead}>
        What the actor actually saw at this decision &mdash; unlike the
        omniscient Game State panel above.
      </p>

      <div className={styles.chips}>
        <span className={styles.chip}>
          {partnerMode === 0 ? "Jack of Diamonds" : "Called Ace"}
        </span>

        {isLeaster && <span className={styles.chip}>Leaster</span>}
        {playStarted && <span className={styles.chip}>Play Started</span>}
        {aloneCalled && <span className={styles.chip}>Alone Called</span>}
        {calledUnder && <span className={styles.chip}>Called Under</span>}

        <span className={styles.chip}>Trick {currentTrick + 1}</span>

        {calledCard && (
          <span className={styles.chip}>
            Called <CardText>{calledCard}</CardText>
          </span>
        )}

        <span className={styles.chip}>
          Picker: {relLabel(pickerRel)}
          {pickerRel !== 0 && (
            <span className={styles.chipSub}> (seat {pickerPosition})</span>
          )}
        </span>

        <span className={styles.chip}>Partner: {relLabel(partnerRel)}</span>

        <span className={styles.chip}>Leader: {relLabel(leaderRel)}</span>
      </div>

      {hand.length > 0 && (
        <div className={styles.cardGroup}>
          <div className={styles.cardGroupLabel}>Hand</div>
          <div className={styles.cardList}>
            {hand.map((card, i) => (
              <PlayingCard key={i} code={card} w={48} />
            ))}
          </div>
        </div>
      )}

      {blind.length > 0 && (
        <div className={styles.cardGroup}>
          <div className={styles.cardGroupLabel}>Blind</div>
          <div className={styles.cardList}>
            {blind.map((card, i) => (
              <PlayingCard key={i} code={card} w={48} />
            ))}
          </div>
        </div>
      )}

      {bury.length > 0 && (
        <div className={styles.cardGroup}>
          <div className={styles.cardGroupLabel}>Bury</div>
          <div className={styles.cardList}>
            {bury.map((card, i) => (
              <PlayingCard key={i} code={card} w={48} />
            ))}
          </div>
        </div>
      )}

      <div className={styles.cardGroup}>
        <div className={styles.cardGroupLabel}>Trick (relative seating)</div>
        <div className={styles.trickList}>
          {trick.map((slot) => (
            <div key={slot.relativePosition} className={styles.trickSlot}>
              <div className={styles.trickSeatName}>{slot.seatName}</div>
              {slot.card ? (
                <PlayingCard code={slot.card} w={48} />
              ) : (
                <div className={styles.emptySlot} />
              )}
              <div className={styles.trickBadges}>
                {slot.isPicker && (
                  <span className={styles.badgePicker}>picker</span>
                )}
                {slot.isPartnerKnown && (
                  <span className={styles.badgePartner}>partner</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
