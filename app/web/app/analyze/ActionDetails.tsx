import React from "react";
import { AnalyzeActionDetail } from "../../lib/analyzeTypes";
import ProbabilityBar from "./ProbabilityBar";
import { PlayingCard } from "../../lib/ds";
import styles from "./ActionDetails.module.css";

interface ActionDetailsProps {
  action: AnalyzeActionDetail;
}

export default function ActionDetails({ action }: ActionDetailsProps) {
  const { view, probabilities } = action;
  const maxProb = Math.max(...probabilities.map((p) => p.prob));

  const hand = view.hand || [];
  const blind = view.blind || [];
  const bury = view.bury || [];
  const currentTrick = view.current_trick || [];

  return (
    <div className={styles.actionDetails}>
      <div className={styles.detailsGrid}>
        {/* Game State */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Game State</div>

          {hand.length > 0 && (
            <div>
              <div className={styles.cardGroupLabel}>Hand</div>
              <div className={styles.cardList}>
                {hand.map((card: string, i: number) => (
                  <PlayingCard key={i} code={card} w={64} />
                ))}
              </div>
            </div>
          )}

          {blind.length > 0 && (
            <div>
              <div className={styles.cardGroupLabel}>Blind</div>
              <div className={styles.cardList}>
                {blind.map((card: string, i: number) => (
                  <PlayingCard key={i} code={card} w={64} />
                ))}
              </div>
            </div>
          )}

          {bury.length > 0 && (
            <div>
              <div className={styles.cardGroupLabel}>Bury</div>
              <div className={styles.cardList}>
                {bury.map((card: string, i: number) => (
                  <PlayingCard key={i} code={card} w={64} />
                ))}
              </div>
            </div>
          )}

          {currentTrick.some((card: string) => card !== "") && (
            <div>
              <div className={styles.cardGroupLabel}>Current Trick</div>
              <div className={styles.cardList}>
                {currentTrick
                  .filter((card: string) => card !== "")
                  .map((card: string, i: number) => (
                    <PlayingCard key={i} code={card} w={64} />
                  ))}
              </div>
            </div>
          )}

          {/* Additional game context */}
          <div className={styles.gameContext}>
            {view.picker && <div>Picker: Seat {view.picker}</div>}
            {view.partner && <div>Partner: Seat {view.partner}</div>}
            {view.called_card && <div>Called Card: {view.called_card}</div>}
            {view.is_leaster && <div>Mode: Leaster</div>}
            {view.current_trick_index !== undefined && (
              <div>Trick: {view.current_trick_index + 1}/6</div>
            )}
            {typeof action.memoryCosineDistance === "number" && (
              <div
                title="How much this seat's memory vector changed at this decision (cosine distance: 0 = no change; see the Memory Update Magnitude chart)"
              >
                Memory Δ: {action.memoryCosineDistance.toFixed(3)}
              </div>
            )}
          </div>
        </div>

        {/* Action Probabilities */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Action Probabilities</div>
          <div className={styles.probabilitiesList}>
            {probabilities.map((prob, i) => (
              <ProbabilityBar key={i} probability={prob} maxProb={maxProb} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
