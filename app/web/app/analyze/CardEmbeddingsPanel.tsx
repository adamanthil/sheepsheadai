import React, { useMemo, useState } from "react";
import { AnalyzeModelResponse } from "../../lib/analyzeTypes";
import { apiFetch } from "../../lib/api";
import { parseCard, isRedSuit } from "../../lib/ds";
import styles from "./CardEmbeddingsPanel.module.css";

const TRUMP_COUNT = 14;
const SCATTER_VB = 420;
const SCATTER_PAD = 28;
const CELL = 13;
const LABEL_SIZE = 34;

// Diverging pair (blue ↔ red) matching the app's existing --an-accent /
// --an-neg tokens, with a neutral gray midpoint — one hue per pole, lightness
// monotonic outward from 0 (see the dataviz skill's diverging-ramp rule).
const POS_POLE = [37, 99, 235]; // --an-accent #2563eb
const NEG_POLE = [220, 38, 38]; // --an-neg #dc2626
const NEUTRAL = [228, 232, 239]; // --an-rule #e4e8ef

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function divergingColor(v: number): string {
  const clamped = Math.max(-1, Math.min(1, v));
  const pole = clamped >= 0 ? POS_POLE : NEG_POLE;
  const t = Math.abs(clamped);
  const rgb = [0, 1, 2].map((i) => Math.round(lerp(NEUTRAL[i], pole[i], t)));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function cardKind(index: number, total: number): "trump" | "fail" | "under" {
  if (index === total - 1) return "under";
  if (index < TRUMP_COUNT) return "trump";
  return "fail";
}

export default function CardEmbeddingsPanel() {
  const [expanded, setExpanded] = useState(false);
  const [data, setData] = useState<AnalyzeModelResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch("/api/analyze/model");
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status} ${res.statusText}`);
      }
      const json = (await res.json()) as AnalyzeModelResponse;
      setData(json);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load model info.",
      );
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = () => {
    const next = !expanded;
    setExpanded(next);
    if (next && data === null && !loading && error === null) {
      void load();
    }
  };

  const emb = data?.cardEmbeddings ?? null;

  const scatter = useMemo(() => {
    if (!emb) return null;
    const xs = emb.pcaCoords.map((c) => c[0]);
    const ys = emb.pcaCoords.map((c) => c[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const inner = SCATTER_VB - 2 * SCATTER_PAD;
    const x = (v: number) => SCATTER_PAD + ((v - minX) / spanX) * inner;
    // SVG y grows downward; flip so higher PC2 renders higher on screen.
    const y = (v: number) => SCATTER_PAD + (1 - (v - minY) / spanY) * inner;
    return { x, y };
  }, [emb]);

  return (
    <div className={styles.panel}>
      <div className={styles.summary} onClick={handleToggle}>
        <div className={styles.title}>Card Embeddings</div>
        <div className={`${styles.expandIcon} ${expanded ? styles.expanded : ""}`}>
          ▼
        </div>
      </div>

      {expanded && (
        <div className={styles.body}>
          {loading && <div className={styles.status}>Loading model info…</div>}

          {error && !loading && (
            <div className={styles.errorBox}>
              <span>{error}</span>
              <button className={styles.retryButton} onClick={() => void load()}>
                Retry
              </button>
            </div>
          )}

          {!loading && !error && data && !emb && (
            <div className={styles.status}>
              This architecture has no card-embedding table.
            </div>
          )}

          {!loading && !error && data && emb && scatter && (
            <>
              <div className={styles.metaLine}>
                <span>
                  Arch: <strong>{data.arch}</strong>
                </span>
                <span>
                  Critic: <strong>{data.criticMode}</strong>
                </span>
                <span>
                  Dims: <strong>{emb.dims}</strong>
                </span>
                <span>
                  PCA explained variance:{" "}
                  <strong>
                    {emb.pcaExplainedVariance
                      .map((v, i) => `PC${i + 1} ${(v * 100).toFixed(1)}%`)
                      .join(", ")}
                  </strong>
                </span>
              </div>

              <div className={styles.sectionTitle}>PCA Projection</div>
              <div className={styles.scatterWrap}>
                <svg
                  className={styles.scatterSvg}
                  viewBox={`0 0 ${SCATTER_VB} ${SCATTER_VB}`}
                >
                  {emb.cards.map((entry, i) => {
                    const [px, py] = emb.pcaCoords[i];
                    const kind = cardKind(i, emb.cards.length);
                    const cx = scatter.x(px);
                    const cy = scatter.y(py);

                    if (kind === "under") {
                      return (
                        <g key={entry.id}>
                          <circle
                            cx={cx}
                            cy={cy}
                            r={3.5}
                            className={styles.pointUnder}
                          >
                            <title>{`UNDER — PC1 ${px.toFixed(2)}, PC2 ${py.toFixed(2)}`}</title>
                          </circle>
                          <text
                            x={cx + 5}
                            y={cy + 3}
                            className={styles.pointLabelUnder}
                          >
                            under
                          </text>
                        </g>
                      );
                    }

                    const color = isRedSuit(parseCard(entry.card).suit)
                      ? "var(--card-red)"
                      : "var(--card-black)";

                    return (
                      <g key={entry.id}>
                        <circle
                          cx={cx}
                          cy={cy}
                          r={4}
                          style={{
                            fill: kind === "trump" ? color : "var(--an-panel)",
                            stroke: color,
                            strokeWidth: kind === "trump" ? 0 : 1.5,
                          }}
                        >
                          <title>{`${entry.card} — PC1 ${px.toFixed(2)}, PC2 ${py.toFixed(2)}`}</title>
                        </circle>
                        <text x={cx + 5} y={cy + 3} className={styles.pointLabel}>
                          {entry.card}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
              <div className={styles.scatterLegend}>
                <span className={styles.legendItem}>
                  <span className={styles.legendDotFilled} /> Trump
                </span>
                <span className={styles.legendItem}>
                  <span className={styles.legendDotHollow} /> Fail
                </span>
                <span className={styles.legendItem}>
                  <span className={styles.legendSwatchRed} /> Hearts / Diamonds
                </span>
                <span className={styles.legendItem}>
                  <span className={styles.legendSwatchBlack} /> Clubs / Spades
                </span>
                <span className={styles.legendItem}>
                  <span className={styles.legendSwatchUnder} /> Under
                </span>
              </div>

              <div className={styles.sectionTitle}>Cosine Similarity</div>
              <div className={styles.heatmapWrap}>
                <div
                  className={styles.heatmapGrid}
                  style={{
                    gridTemplateColumns: `${LABEL_SIZE}px repeat(${emb.cards.length}, ${CELL}px)`,
                    gridTemplateRows: `${LABEL_SIZE}px repeat(${emb.cards.length}, ${CELL}px)`,
                  }}
                >
                  <div className={styles.heatCorner} />
                  {emb.cards.map((entry, j) => (
                    <div
                      key={`col-${entry.id}`}
                      className={`${styles.heatColLabel} ${
                        j === TRUMP_COUNT - 1 ? styles.boundaryRight : ""
                      }`}
                    >
                      <span>{entry.card}</span>
                    </div>
                  ))}

                  {emb.cards.map((rowEntry, i) => (
                    <React.Fragment key={`row-${rowEntry.id}`}>
                      <div
                        className={`${styles.heatRowLabel} ${
                          i === TRUMP_COUNT - 1 ? styles.boundaryBottom : ""
                        }`}
                      >
                        {rowEntry.card}
                      </div>
                      {emb.cards.map((colEntry, j) => {
                        const value = emb.cosineSim[i]?.[j] ?? 0;
                        return (
                          <div
                            key={`cell-${rowEntry.id}-${colEntry.id}`}
                            className={`${styles.heatCell} ${
                              j === TRUMP_COUNT - 1 ? styles.boundaryRight : ""
                            } ${i === TRUMP_COUNT - 1 ? styles.boundaryBottom : ""}`}
                            style={{ background: divergingColor(value) }}
                            title={`${rowEntry.card} ↔ ${colEntry.card}: ${value.toFixed(2)}`}
                          />
                        );
                      })}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
