"use client";

import React, { useState } from "react";
import {
  AnalyzeSimulateRequest,
  AnalyzeSimulateResponse,
} from "../../lib/analyzeTypes";
import ActionTimeline from "./ActionTimeline";
import GameSummary from "./GameSummary";
import CalibrationSummary from "./CalibrationSummary";
import MemoryDriftChart from "./MemoryDriftChart";
import CardEmbeddingsPanel from "./CardEmbeddingsPanel";
import { apiFetch } from "../../lib/api";
import styles from "./page.module.css";

/** Shape of the untyped `final` payload (server: runtime/views.py). */
type AnalyzeFinal = {
  mode?: string;
  winner?: number;
  picker?: number;
  partner?: number;
  picker_score?: number;
  defender_score?: number;
  scores?: number[];
};

export default function AnalyzePage() {
  // Form state
  const [partnerMode, setPartnerMode] = useState<number>(1);
  const [deterministic, setDeterministic] = useState<boolean>(true);
  const [seed, setSeed] = useState<string>("");
  const [shapingWeightPercent, setShapingWeightPercent] = useState<number>(100);
  const [terminalRewards, setTerminalRewards] = useState<boolean>(true);

  // Results state
  const [response, setResponse] = useState<AnalyzeSimulateResponse | null>(
    null,
  );
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSimulate = async () => {
    setError(null);
    setLoading(true);

    try {
      const requestBody: AnalyzeSimulateRequest = {
        partnerMode,
        deterministic,
        seed: seed ? parseInt(seed, 10) : undefined,
        maxSteps: 200,
      };

      const res = await apiFetch("/api/analyze/simulate", {
        method: "POST",
        body: JSON.stringify(requestBody),
      });

      if (!res.ok) {
        let errorMessage = `Request failed: ${res.status} ${res.statusText}`;
        try {
          const errorData = await res.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          // Ignore JSON parsing errors for error response
        }
        throw new Error(errorMessage);
      }

      const data = await res.json();
      setResponse(data);
    } catch (err) {
      console.error("Simulate error:", err);
      const errorMessage =
        err instanceof Error
          ? err.message
          : "Failed to simulate game. Please try again.";
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleSeedChange = (value: string) => {
    // Allow empty string or valid integers
    if (value === "" || /^\d+$/.test(value)) {
      setSeed(value);
    }
  };

  const final = (response?.final ?? null) as AnalyzeFinal | null;

  const seatName = (seat: number | undefined): string => {
    if (!seat || !response) return `Seat ${seat ?? "?"}`;
    return (
      response.trace.find((step) => step.seat === seat)?.seatName ??
      `Seat ${seat}`
    );
  };

  const pickerWon =
    final && final.mode === "standard" && final.picker && final.scores
      ? final.scores[final.picker - 1] > 0
      : null;

  return (
    <div className={styles.container}>
      {/* Error Display */}
      {error && (
        <div className={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Compact controls bar */}
      <div className={styles.controlsBar}>
        <div className={styles.controlGroup}>
          <label htmlFor="partnerMode" className={styles.label}>
            Partner mode
          </label>
          <select
            id="partnerMode"
            className={styles.select}
            value={partnerMode}
            onChange={(e) => setPartnerMode(parseInt(e.target.value, 10))}
          >
            <option value={0}>Jack of Diamonds</option>
            <option value={1}>Called Ace</option>
          </select>
        </div>

        <div className={styles.controlGroup}>
          <label htmlFor="seed" className={styles.label}>
            Seed
          </label>
          <input
            id="seed"
            type="text"
            className={styles.input}
            placeholder="random"
            value={seed}
            onChange={(e) => handleSeedChange(e.target.value)}
          />
        </div>

        <label className={styles.checkboxWrapper} htmlFor="deterministic">
          <input
            type="checkbox"
            id="deterministic"
            className={styles.checkbox}
            checked={deterministic}
            onChange={(e) => setDeterministic(e.target.checked)}
          />
          <span className={styles.checkboxLabel}>Deterministic</span>
        </label>

        <button
          className={styles.simulateButton}
          onClick={handleSimulate}
          disabled={loading}
        >
          {loading && <span className={styles.spinner} />}
          {loading ? "Simulating…" : "Simulate game"}
        </button>
      </div>

      {!response && !loading && (
        <div className={styles.emptyState}>
          <p>
            Simulate a full game to inspect every decision the model makes:
            action probabilities, value estimates, auxiliary-head predictions,
            and the exact observation behind each choice.
          </p>
        </div>
      )}

      {loading && (
        <div className={styles.loading}>
          <span className={styles.spinner} />
          Analyzing game decisions…
        </div>
      )}

      {response && !loading && (
        <>
          {/* Outcome banner — the verdict first, everything else below */}
          {final && (
            <section
              className={`${styles.outcomeBanner} ${
                final.mode === "leaster"
                  ? styles.outcomeLeaster
                  : pickerWon
                    ? styles.outcomeWin
                    : styles.outcomeLoss
              }`}
            >
              <div className={styles.outcomeVerdict}>
                <h2 className={styles.verdictText}>
                  {final.mode === "leaster"
                    ? "Leaster"
                    : pickerWon
                      ? "Picker side wins"
                      : "Defenders win"}
                </h2>
                <div className={styles.verdictDetail}>
                  {final.mode === "leaster" ? (
                    <>Winner: {seatName(final.winner)} (fewest points)</>
                  ) : (
                    <>
                      Picker {seatName(final.picker)}
                      {response.summary?.partner &&
                        ` · Partner ${response.summary.partner}`}
                    </>
                  )}
                </div>
              </div>

              {final.mode === "standard" && (
                <div className={styles.outcomeScore}>
                  <div className={styles.scoreFigure}>
                    <span>{final.picker_score}</span>
                    <span className={styles.scoreDivider}>:</span>
                    <span>{final.defender_score}</span>
                  </div>
                  <div className={styles.scoreLabels}>
                    <span>picker</span>
                    <span />
                    <span>defenders</span>
                  </div>
                </div>
              )}

              <div className={styles.outcomeChips}>
                <span className={styles.chip}>
                  {response.trace.length} decisions
                </span>
                <span className={styles.chip}>
                  {response.meta.partnerMode === 1
                    ? "Called Ace"
                    : "Jack of Diamonds"}
                </span>
                <span className={styles.chip}>
                  {response.meta.deterministic ? "Deterministic" : "Stochastic"}
                </span>
                {response.meta.seed != null && (
                  <span className={styles.chip}>Seed {response.meta.seed}</span>
                )}
                {response.meta.model && (
                  <span className={styles.chip}>{response.meta.model}</span>
                )}
              </div>
            </section>
          )}

          {response.summary && <GameSummary summary={response.summary} />}

          {response.calibration && (
            <CalibrationSummary calibration={response.calibration} />
          )}

          <MemoryDriftChart trace={response.trace} />

          <CardEmbeddingsPanel />

          {/* Decision timeline */}
          <section className={styles.resultsPanel}>
            <div className={styles.resultsHeader}>
              <h2 className={styles.resultsTitle}>Decision Timeline</h2>

              <div className={styles.rewardControls}>
                <label
                  className={styles.checkboxWrapper}
                  htmlFor="terminalRewards"
                >
                  <input
                    type="checkbox"
                    id="terminalRewards"
                    className={styles.checkbox}
                    checked={terminalRewards}
                    onChange={(e) => setTerminalRewards(e.target.checked)}
                  />
                  <span className={styles.checkboxLabel}>
                    Terminal-only rewards
                  </span>
                </label>

                <div
                  className={styles.shapingControl}
                  style={{ opacity: terminalRewards ? 0.45 : 1 }}
                >
                  <label htmlFor="shapingWeight" className={styles.shapingLabel}>
                    Shaping weight
                  </label>
                  <input
                    id="shapingWeight"
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={shapingWeightPercent}
                    onChange={(e) =>
                      setShapingWeightPercent(parseInt(e.target.value, 10))
                    }
                    className={styles.range}
                    disabled={terminalRewards}
                  />
                  <span className={styles.rangeValue}>
                    {terminalRewards ? "—" : `${shapingWeightPercent}%`}
                  </span>
                </div>
              </div>
            </div>

            <ActionTimeline
              trace={response.trace}
              picker={response.summary?.picker ?? undefined}
              partner={response.summary?.partner ?? undefined}
              shapingWeightPercent={shapingWeightPercent}
              terminalRewards={terminalRewards}
              gamma={response.meta.gamma}
            />
          </section>
        </>
      )}
    </div>
  );
}
