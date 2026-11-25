import React from "react";
import Link from "next/link";
import { CardText, PlayingCard, ds } from "../../../../lib/ds";
import {
  actionCodes,
  cardLabel,
  decisionHeadline,
  decisionKicker,
  decisionMethod,
  decisionVerdict,
  signed,
  tone,
} from "../../../../lib/review";
import type { DecisionReview, HandSeat } from "../../../../lib/types";
import SuitText from "../../components/SuitText";
import TrickRow from "./TrickRow";
import styles from "./handReview.module.css";

// The most choices listed before the rest are summed up in a line.
const MAX_OPTIONS = 6;

function isRed(card: string): boolean {
  return card.endsWith("H") || card.endsWith("D");
}

/** One decision in full: the hand it was made from, how the AI would have
 * played it, and what each choice was worth. */
export default function DecisionDetail({
  decision,
  seat,
  seats,
  blind,
  analyzeHref,
  onPrevious,
  onNext,
}: {
  decision: DecisionReview;
  seat: number;
  seats: HandSeat[];
  /** The two cards of the hand's blind. */
  blind: string[];
  /** The pick analyzer opened on this hand's deal. */
  analyzeHref: string;
  onPrevious: (() => void) | null;
  onNext: (() => void) | null;
}) {
  const chosen = new Set(decision.action.split(" "));
  // The call is the picker's first look at the blind, so the two cards
  // it added are pointed out there.
  const found = new Set(
    decision.kind === "call" || decision.kind === "under" ? blind : [],
  );
  const trump = decision.high_trump;
  const differs = decision.difference != null;
  const kind = tone(decision.difference);
  // A bury has dozens of legal pairs: list the ones the AI gives a real
  // chance, and always the player's own.
  const likely = decision.policy.filter(([action, probability], i) =>
    i < MAX_OPTIONS
      ? probability >= 0.005 || i === 0 || action === decision.action
      : action === decision.action,
  );
  const unlisted = decision.policy.length - likely.length;

  return (
    <>
      <div className={ds.overline}>
        <SuitText>{decisionKicker(decision, seat)}</SuitText>
      </div>
      <h2 className={styles.detailHeadline}>
        <SuitText>{decisionHeadline(decision)}</SuitText>
      </h2>

      {decision.kind === "play" && (
        <>
          <div className={styles.sectionLabel}>The trick</div>
          <TrickRow decision={decision} seat={seat} seats={seats} />
        </>
      )}

      <div className={styles.sectionLabel}>Your hand</div>
      <div className={styles.cards}>
        {decision.hand.map((card) => {
          const face = (
            <PlayingCard
              key={card}
              code={card}
              largeIndex
              className={chosen.has(card) ? styles.cardChosen : undefined}
            />
          );
          return found.has(card) ? (
            <span key={card} className={styles.fromBlind}>
              {face}
              Blind
            </span>
          ) : (
            face
          );
        })}
      </div>
      <p className={styles.highTrump}>
        <span className={styles.sectionLabel}>High trump</span>
        {trump.card ? (
          <>
            <span>
              <span
                className={`${styles.highCard} ${isRed(trump.card) ? styles.red : ""}`}
              >
                {cardLabel(trump.card)}
              </span>{" "}
              <strong className={trump.held ? styles.gain : undefined}>
                {trump.held ? "in your hand" : "still out"}
              </strong>
            </span>
            <span>{trump.unseen} trump you have not seen</span>
          </>
        ) : (
          <strong>no trump left</strong>
        )}
      </p>

      <div className={styles.sectionLabel}>How the AI plays it</div>
      <ul className={styles.policy}>
        {likely.map(([action, probability]) => (
          <li key={action}>
            <CardText className={styles.policyAction}>
              {actionCodes(action)}
            </CardText>
            <span className={styles.policyTrack}>
              <span
                className={
                  action === decision.ai_action ? styles.policyTop : undefined
                }
                style={{ width: `${Math.max(probability * 100, 1)}%` }}
              />
            </span>
            <span className={styles.policyPct}>
              {Math.round(probability * 100)}%
            </span>
            <span className={styles.policyYou}>
              {action === decision.action && (
                <span className={`${ds.badge} ${ds.badgeInk}`}>You</span>
              )}
            </span>
          </li>
        ))}
        {unlisted > 0 && (
          <li className={styles.policyRest}>
            {unlisted} other {unlisted === 1 ? "choice" : "choices"} under 1%
          </li>
        )}
      </ul>

      <dl className={styles.figures}>
        <div>
          <dt>Your choice</dt>
          <dd>
            {signed(
              differs ? (decision.chosen_score ?? 0) : decision.expected_score,
            )}
          </dd>
          <dd className={styles.figureNote}>expected score</dd>
        </div>
        <div>
          <dt>AI’s choice</dt>
          <dd>
            {signed(
              differs ? (decision.ai_score ?? 0) : decision.expected_score,
            )}
          </dd>
          <dd className={styles.figureNote}>expected score</dd>
        </div>
        <div>
          <dt>Difference</dt>
          <dd
            className={
              kind === "cost"
                ? styles.cost
                : kind === "gain"
                  ? styles.gain
                  : undefined
            }
          >
            {differs ? signed(decision.difference ?? 0) : "—"}
          </dd>
          <dd className={styles.figureNote}>{decisionVerdict(decision)}</dd>
        </div>
      </dl>
      <p className={styles.method}>{decisionMethod(decision)}</p>

      <div className={styles.detailActions}>
        <div className={styles.stepButtons}>
          <button
            type="button"
            className={`${ds.btn} ${ds.btnGhost}`}
            disabled={!onPrevious}
            onClick={onPrevious ?? undefined}
          >
            ← Previous
          </button>
          <button
            type="button"
            className={ds.btn}
            disabled={!onNext}
            onClick={onNext ?? undefined}
          >
            Next decision →
          </button>
        </div>
        {/* The analyzer covers the choices before the first card: the
            pick, the call and the bury. */}
        {decision.kind !== "play" && (
          <Link href={analyzeHref} className={ds.link}>
            Open in Analyze AI ↗
          </Link>
        )}
      </div>
    </>
  );
}
