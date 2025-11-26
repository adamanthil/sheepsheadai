import React from 'react';
import styles from '../page.module.css';

interface ScoresOverlayProps {
  onClose: () => void;
  table: any;
}

export default function ScoresOverlay({ onClose, table }: ScoresOverlayProps) {
  const history: Array<any> = table?.resultsHistory || [];
  const seats = table?.seats || {};

  const rows = history.map((h, idx) => ({
    hand: h.hand || idx + 1,
    bySeat: h.bySeat || {},
    sum: h.sum || 0,
  }));

  const initialOrder: string[] = (table?.initialSeatOrder || []).map((x: any) => String(x));
  const ids = table?.seatOccupants || {};

  const labelsById: Record<string, string> = {};
  for (let i = 1; i <= 5; i++) {
    const occ = String(ids[String(i)] || `seat-${i}`);
    labelsById[occ] = seats[String(i)] || `Seat ${i}`;
  }

  const columns: Array<{ id: string; label: string }> = (
    initialOrder.length === 5 ? initialOrder : Object.keys(labelsById)
  ).map((id: string) => ({ id, label: labelsById[id] || id }));

  const scoreFor = (row: any, id: string) => {
    const entries = row.bySeat || {};
    for (const key of Object.keys(entries)) {
      const v = entries[key];
      if (v && String(v.id) === String(id)) return v.score || 0;
    }
    return 0;
  };

  const totalById: Record<string, number> = {};
  columns.forEach((c) => {
    totalById[c.id] = 0;
  });
  rows.forEach((r) => {
    columns.forEach((c) => {
      totalById[c.id] += scoreFor(r, c.id);
    });
  });

  const overallSum = Object.values(totalById).reduce((a, b) => a + (b || 0), 0);

  return (
    <div className={styles.scoresOverlay}>
      <div className={styles.scoresBox}>
        <div className={styles.scoresHeader}>
          <div>
            <strong>Running totals</strong>
          </div>
          <div className={styles.mlAuto}>
            <button onClick={onClose}>Close</button>
          </div>
        </div>
        <div className={styles.scoresBody}>
          <div>
            <table className={styles.scoresTable}>
              <thead>
                <tr>
                  <th className={styles.thLeft}>Hand</th>
                  {columns.map((c, idx) => (
                    <th key={idx} className={styles.thRight}>
                      {c.label}
                    </th>
                  ))}
                  <th className={styles.thRight}>Sum</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, idx) => (
                  <tr key={idx}>
                    <td className={styles.tdHand}>{r.hand}</td>
                    {columns.map((c, i) => (
                      <td key={i} className={styles.tdRight}>
                        {scoreFor(r, c.id)}
                      </td>
                    ))}
                    <td className={styles.tdRightBold}>{r.sum}</td>
                  </tr>
                ))}
                <tr>
                  <td className={styles.tdTotalLabel}>Total</td>
                  {columns.map((c, i) => (
                    <td key={i} className={styles.tdRightTotal}>
                      {totalById[c.id]}
                    </td>
                  ))}
                  <td className={styles.tdRightTotal}>{overallSum}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

