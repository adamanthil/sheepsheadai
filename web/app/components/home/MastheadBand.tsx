import React from 'react';
import { PlayingCard, ds } from '../../../lib/ds';
import styles from './MastheadBand.module.css';

/**
 * Newspaper nameplate band: thick-thin rule on top, a "what this is" label
 * left, three small cards set into the band (Q♣ highest trump · J♦ partner
 * card · A♥ a fail ace), place + year right, hairline below.
 */
export default function MastheadBand({ compact = false }: { compact?: boolean }) {
  const cardW = compact ? 24 : 32;
  return (
    <div>
      <div className={ds.headRule} />
      <div className={`${styles.band} ${compact ? styles.compact : ''}`}>
        <div className={`${ds.overline} ${styles.left}`}>
          {/* Long label on wide screens, short on narrow — CSS-toggled so there
              is no viewport JS / hydration flash. */}
          <span className={styles.longLabel}>The Card Room</span>
          <span className={styles.shortLabel}>Card Room</span>
        </div>
        <div className={styles.cards} aria-hidden="true">
          <div style={{ transform: 'rotate(-7deg) translateY(1px)' }}><PlayingCard code="QC" w={cardW} /></div>
          <div style={{ transform: 'rotate(0deg) translateY(-1px)', zIndex: 1 }}><PlayingCard code="JD" w={cardW} /></div>
          <div style={{ transform: 'rotate(7deg) translateY(1px)' }}><PlayingCard code="AH" w={cardW} /></div>
        </div>
        <div className={`${ds.overline} ${styles.right}`}>
          <span className={styles.longLabel}>Wisconsin · MMXXVI</span>
          <span className={styles.shortLabel}>MMXXVI</span>
        </div>
      </div>
      <div className={styles.hairline} />
    </div>
  );
}
