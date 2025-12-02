import React, { useMemo, useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import styles from './page.module.css';

const TRUMP = [
  'QC', 'QS', 'QH', 'QD', 'JC', 'JS', 'JH', 'JD', 'AD', '10D', 'KD', '9D', '8D', '7D'
] as const;
const FAIL = [
  'AC', '10C', 'KC', '9C', '8C', '7C',
  'AS', '10S', 'KS', '9S', '8S', '7S',
  'AH', '10H', 'KH', '9H', '8H', '7H',
] as const;
const DECK = [...TRUMP, ...FAIL];
const UNDER_TOKEN = 'UNDER';
const STATE_SIZE = 292;

type Row = {
  idx: number;
  value: number;
  micro: string;
  sub: string;
  cat: 'Header' | 'Private Card Indicators' | 'Current Trick Block';
};

interface ActionStateVectorProps {
  action: AnalyzeActionDetail;
}

export default function ActionStateVector({ action }: ActionStateVectorProps) {
  const [expanded, setExpanded] = useState(false);
  const state = action.state;

  const rows: Row[] = useMemo(() => {
    if (!state || state.length === 0) return [];

    const r: Row[] = [];

    const headerMicro: string[] = [
      'Partner selection mode (0 = JD, 1 = Called Ace)',
      'Player position (1-5)',
      'Last position to pass (5 ⇒ all passed → Leaster)',
      'Picker position (0 if not yet picked)',
      'Partner position (0 if unknown)',
      'Alone called flag (0/1)',
      'Called card one-hot — AC',
      'Called card one-hot — AS',
      'Called card one-hot — AH',
      'Called card one-hot — 10C',
      'Called card one-hot — 10S',
      'Called card one-hot — 10H',
      'Is called-under flag (0/1)',
      'Is leaster flag (0/1)',
      'Play started flag (0/1)',
      'Current trick index (0-5)'
    ];

    const headerSubs: string[] = [
      'Core header', 'Core header', 'Core header', 'Core header', 'Core header', 'Core header',
      'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot',
      'Flags', 'Flags', 'Flags', 'Current trick index'
    ];

    for (let i = 0; i <= 15; i++) {
      r.push({ idx: i, value: state[i] ?? 0, micro: headerMicro[i], sub: headerSubs[i], cat: 'Header' });
    }

    const glyph = (s: 'C' | 'S' | 'H' | 'D'): string => (s === 'C' ? '♣' : s === 'S' ? '♠' : s === 'H' ? '♥' : '♦');

    const pushPrivate = (start: number, count: number, sub: string, labelPrefix: string) => {
      for (let i = 0; i < count; i++) {
        const gi = start + i;
        const cardLabel = DECK[i] ?? `Card ${i + 1}`;
        const suit = cardLabel.endsWith('C') ? 'C' : cardLabel.endsWith('S') ? 'S' : cardLabel.endsWith('H') ? 'H' : 'D';
        const withGlyph = `${cardLabel} ${glyph(suit as any)}`;
        r.push({ idx: gi, value: state[gi] ?? 0, micro: `${labelPrefix} — ${withGlyph}`, sub, cat: 'Private Card Indicators' });
      }
    };

    pushPrivate(16, 32, 'Hand one-hot', 'Hand');
    pushPrivate(48, 32, 'Blind one-hot', 'Blind');
    pushPrivate(80, 32, 'Bury one-hot', 'Bury');

    const seatLabels = [
      'Current Trick — Self',
      'Current Trick — Player to the left',
      'Current Trick — Player 2 to the left',
      'Current Trick — Player 2 to the right',
      'Current Trick — Player to the right'
    ];
    const startBase = 112;

    for (let rel = 0; rel < 5; rel++) {
      const seatStart = startBase + rel * 36;
      for (let j = 0; j < 34; j++) {
        const gi = seatStart + j;
        let cardDesc: string;
        if (j === 0) cardDesc = 'Card one-hot — empty';
        else if (j === 33) cardDesc = `Card one-hot — ${UNDER_TOKEN}`;
        else {
          const label = DECK[j - 1];
          const s = label.endsWith('C') ? 'C' : label.endsWith('S') ? 'S' : label.endsWith('H') ? 'H' : 'D';
          cardDesc = `Card one-hot — ${label} ${glyph(s as any)}`;
        }
        r.push({ idx: gi, value: state[gi] ?? 0, micro: cardDesc, sub: seatLabels[rel], cat: 'Current Trick Block' });
      }
      r.push({ idx: seatStart + 34, value: state[seatStart + 34] ?? 0, micro: 'Role flag — is_picker (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
      r.push({ idx: seatStart + 35, value: state[seatStart + 35] ?? 0, micro: 'Role flag — is_known_partner (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
    }

    return r.slice(0, STATE_SIZE);
  }, [state]);

  const { subRowSpan, catRowSpan, rowClasses, subStartIdx, catStartIdx, subZeroOnly } = useMemo(() => {
    const subRowSpanMap: Record<number, number> = {};
    const catRowSpanMap: Record<number, number> = {};
    const classes: Record<number, string> = {};
    const subStart: Record<number, boolean> = {};
    const catStart: Record<number, boolean> = {};
    const subZero: Record<number, boolean> = {};

    const catClass = (cat: Row['cat']): string => {
      if (cat === 'Header') return styles.rowHeader;
      if (cat === 'Private Card Indicators') return styles.rowPrivate;
      return styles.rowTrick;
    };

    let i = 0;
    while (i < rows.length) {
      const cat = rows[i]?.cat;
      if (!cat) break;
      let j = i;
      while (j < rows.length && rows[j].cat === cat) j++;
      catRowSpanMap[i] = j - i;
      catStart[i] = true;
      for (let k = i; k < j; k++) classes[k] = catClass(cat);

      let s = i;
      while (s < j) {
        const sub = rows[s].sub;
        let t = s;
        while (t < j && rows[t].sub === sub) t++;
        subRowSpanMap[s] = t - s;
        subStart[s] = true;
        let allZero = true;
        for (let k = s; k < t; k++) {
          if (rows[k].value) { allZero = false; break; }
        }
        subZero[s] = allZero;
        s = t;
      }

      i = j;
    }

    return { subRowSpan: subRowSpanMap, catRowSpan: catRowSpanMap, rowClasses: classes, subStartIdx: subStart, catStartIdx: catStart, subZeroOnly: subZero };
  }, [rows]);

  if (!state || state.length === 0) {
    return null;
  }

  return (
    <div className={styles.stateVectorSection}>
      <button className={styles.toggleButton} onClick={() => setExpanded(!expanded)}>
        {expanded ? 'Hide' : 'Show'} State Vector
      </button>

      {expanded && (
        <>
          <div className={styles.stateVectorTableWrapper}>
            <div className={styles.stateVectorTableContainer}>
              <table className={styles.stateVectorTable}>
                <thead className={styles.stateVectorHead}>
                  <tr>
                    <th>Index</th>
                    <th>Value</th>
                    <th>Detail</th>
                    <th colSpan={2}>Category</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => {
                    const showSub = subRowSpan[i] !== undefined;
                    const showCat = catRowSpan[i] !== undefined;
                    const groupClass = `${rowClasses[i]} ${subStartIdx[i] ? styles.subGroupStart : ''} ${catStartIdx[i] ? styles.catGroupStart : ''}`.trim();
                    let subAnchor = i;
                    while (subAnchor > 0 && rows[subAnchor - 1].sub === row.sub && rows[subAnchor - 1].cat === row.cat) subAnchor--;
                    const dimClass = subZeroOnly[subAnchor] ? styles.subGroupDim : '';
                    return (
                      <tr key={row.idx} className={`${groupClass} ${dimClass}`.trim()}>
                        <td className={`${styles.stateVectorIndex} ${row.value ? styles.cellHighlighted : ''}`}>{row.idx}</td>
                        <td className={`${styles.stateVectorValue} ${row.value ? styles.valueHighlighted : ''}`}>{row.value}</td>
                        <td className={`${styles.stateVectorMicroCat} ${row.value ? styles.cellHighlighted : ''}`}>{row.micro}</td>
                        {showSub && (
                          <td className={styles.stateVectorSubCat} rowSpan={subRowSpan[i]}>
                            {row.sub}
                          </td>
                        )}
                        {showCat && (
                          <td className={styles.stateVectorCategory} rowSpan={catRowSpan[i]}>
                            {row.cat}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          <button className={styles.toggleButton} onClick={() => setExpanded(!expanded)}>
            ⌃ Collapse
          </button>
        </>
      )}
    </div>
  );
}

