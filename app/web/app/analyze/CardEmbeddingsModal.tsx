import React, { useCallback, useEffect, useMemo, useState } from "react";
import { AnalyzeModelResponse } from "../../lib/analyzeTypes";
import { apiFetch } from "../../lib/api";
import { apiErrorMessage, fetchFailureMessage } from "../../lib/apiError";
import { parseCard, isRedSuit } from "../../lib/ds";
import Term from "./TermHelp";
import styles from "./CardEmbeddingsModal.module.css";

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

interface CardEmbeddingsModalProps {
  open: boolean;
  onClose: () => void;
}

export default function CardEmbeddingsModal({
  open,
  onClose,
}: CardEmbeddingsModalProps) {
  const [data, setData] = useState<AnalyzeModelResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch("/api/analyze/model");
      if (!res.ok) {
        throw new Error(await apiErrorMessage(res, "/api/analyze/model"));
      }
      const json = (await res.json()) as AnalyzeModelResponse;
      setData(json);
    } catch (err) {
      setError(fetchFailureMessage(err, "Failed to load model info."));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open && data === null && !loading && error === null) {
      void load();
    }
  }, [open, data, loading, error, load]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

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

  if (!open) return null;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-label="Card embeddings"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.dialogHeader}>
          <div className={styles.title}>Card Embeddings</div>
          <button
            type="button"
            className={styles.closeButton}
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>

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
              <div className={styles.intro}>
                The model never sees suits or ranks directly — it learns an{" "}
                <Term
                  label="embedding"
                  wiki="https://en.wikipedia.org/wiki/Embedding_(machine_learning)"
                >
                  A learned vector ({emb.dims} dimensions here) that stands
                  in for each card inside the network. Training adjusts these
                  vectors, so cards that play similar roles in the game end
                  up with similar embeddings.
                </Term>{" "}
                for each card. These are static per checkpoint — independent
                of any simulated game. The two views below show which cards
                the model has learned to treat as similar.
              </div>

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
                  <Term
                    label="PCA explained variance"
                    wiki="https://en.wikipedia.org/wiki/Principal_component_analysis"
                  >
                    How much of the embeddings&rsquo; total variance the two
                    principal components capture. PC1 40% means the first
                    axis alone accounts for 40% of the variance across
                    cards; whatever the two axes don&rsquo;t capture is
                    invisible in the 2-D projection below.
                  </Term>
                  :{" "}
                  <strong>
                    {emb.pcaExplainedVariance
                      .map((v, i) => `PC${i + 1} ${(v * 100).toFixed(1)}%`)
                      .join(", ")}
                  </strong>
                </span>
              </div>

              <div className={styles.sectionTitle}>
                <Term
                  label="PCA Projection"
                  wiki="https://en.wikipedia.org/wiki/Principal_component_analysis"
                >
                  Each card&rsquo;s embedding lives in {emb.dims} dimensions
                  — too many to draw. Principal component analysis projects
                  the vectors onto the two orthogonal directions of greatest
                  variance. Cards that sit close together have similar
                  embeddings; the axes themselves carry no intrinsic
                  meaning.
                </Term>
              </div>
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

              <div className={styles.sectionTitle}>
                <Term
                  label="Cosine Similarity"
                  wiki="https://en.wikipedia.org/wiki/Cosine_similarity"
                >
                  The cosine of the angle between two cards&rsquo; embedding
                  vectors, ignoring magnitude: +1 = pointing the same way
                  (treated near-identically), 0 = orthogonal (unrelated),
                  −1 = opposite. Each cell compares the row card with the
                  column card; the diagonal is every card with itself.
                </Term>
              </div>
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
      </div>
    </div>
  );
}
