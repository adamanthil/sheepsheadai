import React, { useCallback, useEffect, useMemo, useState } from "react";
import { AnalyzeModelResponse } from "../../lib/analyzeTypes";
import { apiFetch } from "../../lib/api";
import { apiErrorMessage, fetchFailureMessage } from "../../lib/apiError";
import { parseCard, isRedSuit } from "../../lib/ds";
import Term from "./TermHelp";
import styles from "./CardEmbeddingsModal.module.css";

const TRUMP_COUNT = 14;
const SCATTER_W = 760;
const SCATTER_H = 460;
const SCATTER_PAD = 30;

// Diverging pair (blue ↔ red) matching the app's existing --an-accent /
// --an-neg tokens, with a neutral gray midpoint — one hue per pole, lightness
// monotonic outward from 0 (see the dataviz skill's diverging-ramp rule).
const POS_POLE = [37, 99, 235]; // --an-accent #2563eb
const NEG_POLE = [220, 38, 38]; // --an-neg #dc2626
const NEUTRAL = [228, 232, 239]; // --an-rule #e4e8ef

const SCALE_GRADIENT = `linear-gradient(to right, rgb(${NEG_POLE.join(",")}), rgb(${NEUTRAL.join(",")}), rgb(${POS_POLE.join(",")}))`;

// Dimension roles assigned by SharedFeatureEncoder._build_informed_card_init;
// dims past the informed block start at zero, free for the model to define.
const INFORMED_DIM_LABELS = [
  "Trump suit",
  "Clubs suit",
  "Spades suit",
  "Hearts suit",
  "Trump rank",
  "Clubs rank",
  "Spades rank",
  "Hearts rank",
  "Point value",
  "Under flag",
];

function dimLabels(dims: number): string[] {
  if (dims < INFORMED_DIM_LABELS.length) {
    return Array.from({ length: dims }, (_, i) => `Dim ${i}`);
  }
  return Array.from(
    { length: dims },
    (_, i) => INFORMED_DIM_LABELS[i] ?? `Free ${i}`,
  );
}

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

type TabId = "vectors" | "pca" | "cosine";

const TABS: { id: TabId; label: string }[] = [
  { id: "vectors", label: "Embeddings" },
  { id: "pca", label: "PCA Projection" },
  { id: "cosine", label: "Cosine Similarity" },
];

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
  const [tab, setTab] = useState<TabId>("vectors");

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
    const innerW = SCATTER_W - 2 * SCATTER_PAD;
    const innerH = SCATTER_H - 2 * SCATTER_PAD;
    const x = (v: number) => SCATTER_PAD + ((v - minX) / spanX) * innerW;
    // SVG y grows downward; flip so higher PC2 renders higher on screen.
    const y = (v: number) => SCATTER_PAD + (1 - (v - minY) / spanY) * innerH;
    return { x, y };
  }, [emb]);

  // Symmetric scale for the raw-vector heatmap: values are unbounded, so
  // normalize colors by the largest magnitude in the table.
  const vecMaxAbs = useMemo(() => {
    if (!emb) return 1;
    let m = 0;
    for (const entry of emb.cards) {
      for (const v of entry.vector) m = Math.max(m, Math.abs(v));
    }
    return m || 1;
  }, [emb]);

  const labels = useMemo(() => (emb ? dimLabels(emb.dims) : []), [emb]);

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
                of any simulated game.
              </div>

              <div className={styles.tabBar} role="tablist">
                {TABS.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    role="tab"
                    aria-selected={tab === t.id}
                    className={`${styles.tab} ${tab === t.id ? styles.tabActive : ""}`}
                    onClick={() => setTab(t.id)}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              {tab === "vectors" && (
                <div className={styles.tabPane} role="tabpanel">
                  <div className={styles.paneExplain}>
                    Every value in the table — one row per card, one column
                    per embedding dimension. Column labels show each
                    dimension&rsquo;s <em>informed initialization</em>: the
                    meaning it started training with. That is only the
                    starting point — the model adjusts all dimensions freely,
                    so their learned meanings drift and mix over time.
                  </div>
                  <div
                    className={styles.vecGrid}
                    style={{
                      gridTemplateColumns: `44px repeat(${emb.dims}, 1fr)`,
                    }}
                  >
                    <div className={styles.vecCorner} />
                    {labels.map((label, d) => (
                      <div key={`dim-${d}`} className={styles.vecColLabel}>
                        <span>{label}</span>
                      </div>
                    ))}
                    {emb.cards.map((entry, i) => (
                      <React.Fragment key={`vrow-${entry.id}`}>
                        <div
                          className={`${styles.vecRowLabel} ${
                            i === TRUMP_COUNT - 1 ? styles.boundaryBottom : ""
                          }`}
                        >
                          {entry.card}
                        </div>
                        {entry.vector.map((v, d) => (
                          <div
                            key={`vcell-${entry.id}-${d}`}
                            className={`${styles.vecCell} ${
                              i === TRUMP_COUNT - 1 ? styles.boundaryBottom : ""
                            }`}
                            style={{ background: divergingColor(v / vecMaxAbs) }}
                            title={`${entry.card} · ${labels[d]}: ${v.toFixed(3)}`}
                          />
                        ))}
                      </React.Fragment>
                    ))}
                  </div>
                  <div className={styles.scaleWrap}>
                    <span>−{vecMaxAbs.toFixed(2)}</span>
                    <div
                      className={styles.scaleBar}
                      style={{ background: SCALE_GRADIENT }}
                    />
                    <span>+{vecMaxAbs.toFixed(2)}</span>
                  </div>
                </div>
              )}

              {tab === "pca" && (
                <div className={styles.tabPane} role="tabpanel">
                  <div className={styles.paneExplain}>
                    <Term
                      label="PCA projection"
                      wiki="https://en.wikipedia.org/wiki/Principal_component_analysis"
                    >
                      Each card&rsquo;s embedding lives in {emb.dims}{" "}
                      dimensions — too many to draw. Principal component
                      analysis projects the vectors onto the two orthogonal
                      directions of greatest variance. Cards that sit close
                      together have similar embeddings; the axes themselves
                      carry no intrinsic meaning.
                    </Term>{" "}
                    of all {emb.cards.length} embeddings.{" "}
                    <Term
                      label="Explained variance"
                      wiki="https://en.wikipedia.org/wiki/Principal_component_analysis"
                    >
                      How much of the embeddings&rsquo; total variance the
                      two principal components capture. PC1 40% means the
                      first axis alone accounts for 40% of the variance
                      across cards; whatever the two axes don&rsquo;t
                      capture is invisible in this 2-D projection.
                    </Term>
                    :{" "}
                    <strong>
                      {emb.pcaExplainedVariance
                        .map((v, i) => `PC${i + 1} ${(v * 100).toFixed(1)}%`)
                        .join(", ")}
                    </strong>
                    .
                  </div>
                  <div className={styles.scatterWrap}>
                    <svg
                      className={styles.scatterSvg}
                      viewBox={`0 0 ${SCATTER_W} ${SCATTER_H}`}
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
                                fill:
                                  kind === "trump" ? color : "var(--an-panel)",
                                stroke: color,
                                strokeWidth: kind === "trump" ? 0 : 1.5,
                              }}
                            >
                              <title>{`${entry.card} — PC1 ${px.toFixed(2)}, PC2 ${py.toFixed(2)}`}</title>
                            </circle>
                            <text
                              x={cx + 5}
                              y={cy + 3}
                              className={styles.pointLabel}
                            >
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
                      <span className={styles.legendSwatchRed} /> Hearts /
                      Diamonds
                    </span>
                    <span className={styles.legendItem}>
                      <span className={styles.legendSwatchBlack} /> Clubs /
                      Spades
                    </span>
                    <span className={styles.legendItem}>
                      <span className={styles.legendSwatchUnder} /> Under
                    </span>
                  </div>
                </div>
              )}

              {tab === "cosine" && (
                <div className={styles.tabPane} role="tabpanel">
                  <div className={styles.paneExplain}>
                    <Term
                      label="Cosine similarity"
                      wiki="https://en.wikipedia.org/wiki/Cosine_similarity"
                    >
                      The cosine of the angle between two cards&rsquo;
                      embedding vectors, ignoring magnitude: +1 = pointing
                      the same way (treated near-identically), 0 = orthogonal
                      (unrelated), −1 = opposite. Each cell compares the row
                      card with the column card; the diagonal is every card
                      with itself.
                    </Term>{" "}
                    between every pair of cards. The heavy rule marks the
                    trump / fail boundary.
                  </div>
                  <div
                    className={styles.heatmapGrid}
                    style={{
                      gridTemplateColumns: `34px repeat(${emb.cards.length}, 1fr)`,
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
                                j === TRUMP_COUNT - 1
                                  ? styles.boundaryRight
                                  : ""
                              } ${
                                i === TRUMP_COUNT - 1
                                  ? styles.boundaryBottom
                                  : ""
                              }`}
                              style={{ background: divergingColor(value) }}
                              title={`${rowEntry.card} ↔ ${colEntry.card}: ${value.toFixed(2)}`}
                            />
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                  <div className={styles.scaleWrap}>
                    <span>−1</span>
                    <div
                      className={styles.scaleBar}
                      style={{ background: SCALE_GRADIENT }}
                    />
                    <span>+1</span>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
