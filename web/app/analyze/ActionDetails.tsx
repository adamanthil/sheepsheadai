import React, { useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ProbabilityBar from './ProbabilityBar';
import styles from './page.module.css';

interface ActionDetailsProps {
  action: AnalyzeActionDetail;
}

export default function ActionDetails({ action }: ActionDetailsProps) {
  const [showStateVector, setShowStateVector] = useState(false);

  const { view, probabilities, state } = action;
  const maxProb = Math.max(...probabilities.map(p => p.prob));

  // Extract relevant cards from view
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
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Hand
              </div>
              <div className={styles.cardList}>
                {hand.map((card: string, i: number) => (
                  <div key={i} className={styles.card}>{card}</div>
                ))}
              </div>
            </div>
          )}

          {blind.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Blind
              </div>
              <div className={styles.cardList}>
                {blind.map((card: string, i: number) => (
                  <div key={i} className={styles.card}>{card}</div>
                ))}
              </div>
            </div>
          )}

          {bury.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Bury
              </div>
              <div className={styles.cardList}>
                {bury.map((card: string, i: number) => (
                  <div key={i} className={styles.card}>{card}</div>
                ))}
              </div>
            </div>
          )}

          {currentTrick.some((card: string) => card !== '') && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Current Trick
              </div>
              <div className={styles.cardList}>
                {currentTrick.filter((card: string) => card !== '').map((card: string, i: number) => (
                  <div key={i} className={styles.card}>{card}</div>
                ))}
              </div>
            </div>
          )}

          {/* Additional game context */}
          <div style={{ marginTop: '1rem' }}>
            <div style={{ fontSize: '0.75rem', color: '#6b7280', lineHeight: '1.4' }}>
              {view.picker && <div>Picker: Seat {view.picker}</div>}
              {view.partner && <div>Partner: Seat {view.partner}</div>}
              {view.called_card && <div>Called Card: {view.called_card}</div>}
              {view.is_leaster && <div>Mode: Leaster</div>}
              {view.current_trick_index !== undefined && (
                <div>Trick: {view.current_trick_index + 1}/6</div>
              )}
            </div>
          </div>
        </div>

        {/* Action Probabilities */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Action Probabilities</div>
          <div className={styles.probabilitiesList}>
            {probabilities.map((prob, i) => (
              <ProbabilityBar
                key={i}
                probability={prob}
                maxProb={maxProb}
              />
            ))}
          </div>
        </div>
      </div>

      {/* State Vector Section */}
      {state && (
        <div className={styles.stateVectorSection}>
          <button
            className={styles.toggleButton}
            onClick={() => setShowStateVector(!showStateVector)}
          >
            {showStateVector ? 'Hide' : 'Show'} State Vector
          </button>

          {showStateVector && (
            <>
              <div className={styles.stateVector}>
                [{state.join(', ')}]
              </div>
              <div className={styles.stateVectorLegend}>
                <strong>State Vector Legend:</strong><br />
                [0-15]: Header (position, picker, partner, etc.)<br />
                [16-47]: Hand cards<br />
                [48-79]: Blind cards<br />
                [80-111]: Bury cards<br />
                [112+]: Current trick and role information
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
