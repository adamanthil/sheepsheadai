"use client";

import React, { useState } from "react";
import { CardText } from "../../../lib/ds";
import { FAIL_BY_SUIT, TRUMP, sortByDeckOrder } from "./deck";
import styles from "./HandBlindPicker.module.css";

export type CardTarget = "hand" | "blind";

/** Locked cards are the user's fixed choices (sent to the server); dealt
 * cards are the full hand/blind the engine actually used on the last
 * analysis (locked ⊆ dealt). Null dealt = no analysis run yet. */
export interface PickerState {
  lockedHand: string[];
  lockedBlind: string[];
  dealtHand: string[] | null;
  dealtBlind: string[] | null;
}

interface HandBlindPickerProps extends PickerState {
  onChange: (next: PickerState) => void;
}

const LIMITS: Record<CardTarget, number> = { hand: 6, blind: 2 };

/** Scenario deal builder: lock any subset of cards into the hand or blind;
 * the engine deals the rest randomly on each analysis and the dealt cards
 * show up here, clickable to lock them for the next run. */
export default function HandBlindPicker({
  lockedHand,
  lockedBlind,
  dealtHand,
  dealtBlind,
  onChange,
}: HandBlindPickerProps) {
  const [target, setTarget] = useState<CardTarget>("hand");

  const locked: Record<CardTarget, string[]> = {
    hand: lockedHand,
    blind: lockedBlind,
  };
  const dealt: Record<CardTarget, string[] | null> = {
    hand: dealtHand,
    blind: dealtBlind,
  };
  const display: Record<CardTarget, string[]> = {
    hand: dealtHand ?? lockedHand,
    blind: dealtBlind ?? lockedBlind,
  };

  const zoneOf = (card: string): CardTarget | null =>
    display.hand.includes(card)
      ? "hand"
      : display.blind.includes(card)
        ? "blind"
        : null;

  const set = (
    which: CardTarget,
    nextLocked: string[],
    nextDealt: string[] | null,
  ) => {
    onChange({
      lockedHand: which === "hand" ? nextLocked : lockedHand,
      lockedBlind: which === "blind" ? nextLocked : lockedBlind,
      dealtHand: which === "hand" ? nextDealt : dealtHand,
      dealtBlind: which === "blind" ? nextDealt : dealtBlind,
    });
  };

  const toggleCard = (card: string) => {
    const zone = zoneOf(card);
    if (zone) {
      // Toggle the lock. An unlocked card stays visible only while a deal
      // backs it; before the first analysis it simply leaves the zone.
      const next = locked[zone].includes(card)
        ? locked[zone].filter((c) => c !== card)
        : sortByDeckOrder([...locked[zone], card]);
      set(zone, next, dealt[zone]);
      return;
    }

    // Free card -> lock it into the active target.
    const zoneLocked = locked[target];
    if (zoneLocked.length >= LIMITS[target]) return;
    const zoneDealt = dealt[target];
    const nextLocked = sortByDeckOrder([...zoneLocked, card]);
    if (zoneDealt === null) {
      set(target, nextLocked, null);
      return;
    }
    // The zone is already full from the last deal: swap out the weakest
    // unlocked dealt card so it stays full.
    const evict = [...zoneDealt]
      .reverse()
      .find((c) => !zoneLocked.includes(c));
    if (!evict) return;
    set(
      target,
      nextLocked,
      sortByDeckOrder([...zoneDealt.filter((c) => c !== evict), card]),
    );
  };

  const clear = (which: CardTarget) => set(which, [], null);

  const targetButton = (which: CardTarget, label: string) => {
    const cards = display[which];
    const zoneLocked = locked[which];
    return (
      <div
        className={`${styles.targetGroup} ${
          target === which ? styles.targetActive : ""
        }`}
      >
        <button
          type="button"
          className={styles.targetButton}
          onClick={() => setTarget(which)}
          aria-pressed={target === which}
        >
          <span className={styles.targetLabel}>{label}</span>
          <span className={styles.targetCount}>
            {zoneLocked.length}/{LIMITS[which]} locked
          </span>
        </button>
        <div className={styles.targetCards}>
          {cards.length === 0 ? (
            <span className={styles.randomNote}>dealt randomly</span>
          ) : (
            cards.map((card) => {
              const isLocked = zoneLocked.includes(card);
              return (
                <button
                  key={card}
                  type="button"
                  className={`${styles.selectedCard} ${
                    isLocked ? styles.selectedLocked : styles.selectedDealt
                  }`}
                  onClick={() => toggleCard(card)}
                  title={
                    isLocked
                      ? `${card} is locked — click to unlock`
                      : `${card} was dealt randomly — click to lock it`
                  }
                >
                  <CardText>{card}</CardText>
                </button>
              );
            })
          )}
        </div>
        <div className={styles.targetActions}>
          {(zoneLocked.length > 0 || dealt[which] !== null) && (
            <button
              type="button"
              className={styles.miniButton}
              onClick={() => clear(which)}
            >
              Clear
            </button>
          )}
        </div>
      </div>
    );
  };

  const cardButton = (card: string) => {
    const zone = zoneOf(card);
    const isLocked = zone !== null && locked[zone].includes(card);
    const targetFull =
      zone === null && locked[target].length >= LIMITS[target];
    const className = [
      styles.card,
      zone === "hand" ? styles.cardInHand : "",
      zone === "blind" ? styles.cardInBlind : "",
      isLocked ? styles.cardLocked : "",
      targetFull ? styles.cardDim : "",
    ]
      .filter(Boolean)
      .join(" ");
    return (
      <button
        key={card}
        type="button"
        className={className}
        onClick={() => toggleCard(card)}
        title={
          zone
            ? isLocked
              ? `Locked in ${zone} — click to unlock`
              : `Dealt into ${zone} — click to lock it there`
            : `Lock ${card} into the ${target}`
        }
      >
        <CardText>{card}</CardText>
      </button>
    );
  };

  return (
    <div className={styles.picker}>
      <div className={styles.targets}>
        {targetButton("hand", "Hand")}
        {targetButton("blind", "Blind")}
      </div>

      <div className={styles.grid}>
        <div className={styles.gridRow}>
          <span className={styles.rowLabel}>Trump</span>
          <div className={styles.rowCards}>{TRUMP.map(cardButton)}</div>
        </div>
        {Object.entries(FAIL_BY_SUIT).map(([suit, cards]) => (
          <div className={styles.gridRow} key={suit}>
            <span className={styles.rowLabel}>{suit}</span>
            <div className={styles.rowCards}>{cards.map(cardButton)}</div>
          </div>
        ))}
      </div>

      <p className={styles.pickerHint}>
        Locked cards (solid outline) are kept on every analysis; the rest are
        re-dealt each run. Click a dealt card to lock it.
      </p>
    </div>
  );
}
