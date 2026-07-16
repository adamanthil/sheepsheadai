"use client";

import React, { useState } from "react";
import { CardText } from "../../../lib/ds";
import { DECK, FAIL_BY_SUIT, TRUMP, sortByDeckOrder } from "./deck";
import styles from "./HandBlindPicker.module.css";

export type CardTarget = "hand" | "blind";

interface HandBlindPickerProps {
  hand: string[];
  blind: string[];
  onChange: (hand: string[], blind: string[]) => void;
}

const LIMITS: Record<CardTarget, number> = { hand: 6, blind: 2 };

/** Scenario deal builder: click cards in the deck grid to assign them to
 * the hand or the blind. Leaving a set empty means "deal it randomly". */
export default function HandBlindPicker({
  hand,
  blind,
  onChange,
}: HandBlindPickerProps) {
  const [target, setTarget] = useState<CardTarget>("hand");

  const assignment = (card: string): CardTarget | null =>
    hand.includes(card) ? "hand" : blind.includes(card) ? "blind" : null;

  const toggleCard = (card: string) => {
    const current = assignment(card);
    if (current === "hand") {
      onChange(
        hand.filter((c) => c !== card),
        blind,
      );
      return;
    }
    if (current === "blind") {
      onChange(
        hand,
        blind.filter((c) => c !== card),
      );
      return;
    }
    const sets = { hand, blind };
    if (sets[target].length >= LIMITS[target]) return;
    const next = sortByDeckOrder([...sets[target], card]);
    onChange(
      target === "hand" ? next : hand,
      target === "blind" ? next : blind,
    );
  };

  const randomize = (which: CardTarget) => {
    const taken = new Set(which === "hand" ? blind : hand);
    const pool = DECK.filter((c) => !taken.has(c));
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const drawn = sortByDeckOrder(pool.slice(0, LIMITS[which]));
    onChange(which === "hand" ? drawn : hand, which === "blind" ? drawn : blind);
  };

  const clear = (which: CardTarget) => {
    onChange(which === "hand" ? [] : hand, which === "blind" ? [] : blind);
  };

  const targetButton = (which: CardTarget, label: string) => {
    const cards = which === "hand" ? hand : blind;
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
            {cards.length}/{LIMITS[which]}
          </span>
        </button>
        <div className={styles.targetCards}>
          {cards.length === 0 ? (
            <span className={styles.randomNote}>dealt randomly</span>
          ) : (
            cards.map((card) => (
              <button
                key={card}
                type="button"
                className={styles.selectedCard}
                onClick={() => toggleCard(card)}
                title={`Remove ${card}`}
              >
                <CardText>{card}</CardText>
              </button>
            ))
          )}
        </div>
        <div className={styles.targetActions}>
          <button
            type="button"
            className={styles.miniButton}
            onClick={() => randomize(which)}
          >
            Randomize
          </button>
          {cards.length > 0 && (
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
    const current = assignment(card);
    const targetFull =
      current === null &&
      (target === "hand" ? hand.length >= 6 : blind.length >= 2);
    const className = [
      styles.card,
      current === "hand" ? styles.cardInHand : "",
      current === "blind" ? styles.cardInBlind : "",
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
          current
            ? `In ${current} — click to remove`
            : `Add ${card} to ${target}`
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
    </div>
  );
}
