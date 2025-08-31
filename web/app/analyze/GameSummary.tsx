import React from 'react';
import { AnalyzeGameSummary } from '../../lib/analyzeTypes';
import { CardText } from '../../lib/components';
import styles from './page.module.css';

interface GameSummaryProps {
  summary: AnalyzeGameSummary;
}

export default function GameSummary({ summary }: GameSummaryProps) {
  return (
    <div className={styles.gameSummary}>
      <div className={styles.summarySection}>
        <h3 className={styles.summaryTitle}>🎯 Game Summary</h3>

        {/* Hands */}
        <div className={styles.summarySubsection}>
          <h4 className={styles.summarySubtitle}>Hands</h4>
          <div className={styles.handsGrid}>
            {Object.entries(summary.hands).map(([player, cards]) => (
              <div key={player} className={styles.handRow}>
                <div className={styles.playerName}>{player}</div>
                <div className={styles.cardList}>
                  {cards.map((card, i) => (
                    <span key={i} className={styles.card}>
                      <CardText>{card}</CardText>
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Blind */}
        {summary.blind.length > 0 && (
          <div className={styles.summarySubsection}>
            <h4 className={styles.summarySubtitle}>Blind</h4>
            <div className={styles.cardList}>
              {summary.blind.map((card, i) => (
                <span key={i} className={styles.card}>
                  <CardText>{card}</CardText>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Game Info */}
        <div className={styles.summarySubsection}>
          <h4 className={styles.summarySubtitle}>Game Result</h4>
          <div className={styles.gameInfoGrid}>
            {summary.picker && (
              <div className={styles.gameInfoItem}>
                <span className={styles.infoLabel}>Picker:</span>
                <span className={styles.infoValue}>{summary.picker}</span>
              </div>
            )}
            {summary.partner && (
              <div className={styles.gameInfoItem}>
                <span className={styles.infoLabel}>Partner:</span>
                <span className={styles.infoValue}>{summary.partner}</span>
              </div>
            )}
            {summary.bury.length > 0 && (
              <div className={styles.gameInfoItem}>
                <span className={styles.infoLabel}>Bury:</span>
                <div className={styles.cardList}>
                  {summary.bury.map((card, i) => (
                    <span key={i} className={styles.card}>
                      <CardText>{card}</CardText>
                    </span>
                  ))}
                </div>
              </div>
            )}
            <div className={styles.gameInfoItem}>
              <span className={styles.infoLabel}>Picker points:</span>
              <span className={styles.infoValue}>{summary.pickerPoints}</span>
            </div>
            <div className={styles.gameInfoItem}>
              <span className={styles.infoLabel}>Defender points:</span>
              <span className={styles.infoValue}>{summary.defenderPoints}</span>
            </div>
          </div>
        </div>

        {/* Scores */}
        <div className={styles.summarySubsection}>
          <h4 className={styles.summarySubtitle}>Final Scores</h4>
          <div className={styles.scoresGrid}>
            {summary.scores.map((score, index) => (
              <div key={index} className={styles.scoreItem}>
                <span className={styles.scorePlayer}>
                  {Object.keys(summary.hands)[index]}
                </span>
                <span className={`${styles.scoreValue} ${score > 0 ? styles.positiveScore : score < 0 ? styles.negativeScore : ''}`}>
                  {score > 0 ? '+' : ''}{score}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
