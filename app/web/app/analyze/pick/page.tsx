"use client";

import React, { useState } from "react";
import type { components } from "../../../lib/api.gen";
import type { AnalyzeActionDetail } from "../../../lib/analyzeTypes";
import { apiFetch } from "../../../lib/api";
import { apiErrorMessage, fetchFailureMessage } from "../../../lib/apiError";
import DecisionFlow from "./DecisionFlow";
import HandBlindPicker, { PickerState } from "./HandBlindPicker";
import styles from "./page.module.css";

type AnalyzePickRequest = components["schemas"]["AnalyzePickRequest"];
type AnalyzePickResponse = Omit<
  components["schemas"]["AnalyzePickResponse"],
  "decisions"
> & { decisions: AnalyzeActionDetail[] };

export default function PickAnalysisPage() {
  const [partnerMode, setPartnerMode] = useState<number>(1);
  const [seat, setSeat] = useState<number>(1);
  const [seed, setSeed] = useState<string>("");
  const [deterministic, setDeterministic] = useState<boolean>(true);
  const [picker, setPicker] = useState<PickerState>({
    lockedHand: [],
    lockedBlind: [],
    dealtHand: null,
    dealtBlind: null,
  });

  const [response, setResponse] = useState<AnalyzePickResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setError(null);
    setLoading(true);

    try {
      const requestBody: AnalyzePickRequest = {
        partnerMode,
        seat,
        deterministic,
        seed: seed ? parseInt(seed, 10) : undefined,
        hand: picker.lockedHand.length > 0 ? picker.lockedHand : undefined,
        blind: picker.lockedBlind.length > 0 ? picker.lockedBlind : undefined,
      };

      const res = await apiFetch("/api/analyze/pick", {
        method: "POST",
        body: JSON.stringify(requestBody),
      });

      if (!res.ok) {
        throw new Error(await apiErrorMessage(res, "/api/analyze/pick"));
      }

      const data: AnalyzePickResponse = await res.json();
      setResponse(data);
      // Reflect the engine's actual deal in the builder: locked cards stay
      // locked, the randomly filled remainder shows up as lockable.
      setPicker((prev) => ({
        ...prev,
        dealtHand: data.scenario.hand,
        dealtBlind: data.scenario.blind,
      }));
    } catch (err) {
      console.error("Pick analysis error:", err);
      setError(
        fetchFailureMessage(err, "Failed to analyze scenario. Please try again."),
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSeedChange = (value: string) => {
    if (value === "" || /^\d+$/.test(value)) {
      setSeed(value);
    }
  };

  return (
    <div className={styles.container}>
      {error && (
        <div className={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Scenario builder */}
      <section className={styles.builderPanel}>
        <div className={styles.builderHeader}>
          <h2 className={styles.builderTitle}>Scenario</h2>
          <p className={styles.builderHint}>
            The chosen seat faces the pick decision — earlier seats have
            passed. Lock any subset of cards; the rest are dealt randomly.
          </p>
        </div>

        <div className={styles.dials}>
          <div className={styles.controlGroup}>
            <label htmlFor="pickPartnerMode" className={styles.label}>
              Partner mode
            </label>
            <select
              id="pickPartnerMode"
              className={styles.select}
              value={partnerMode}
              onChange={(e) => setPartnerMode(parseInt(e.target.value, 10))}
            >
              <option value={0}>Jack of Diamonds</option>
              <option value={1}>Called Ace</option>
            </select>
          </div>

          <div className={styles.controlGroup}>
            <label htmlFor="pickSeat" className={styles.label}>
              Seat
            </label>
            <select
              id="pickSeat"
              className={styles.select}
              value={seat}
              onChange={(e) => setSeat(parseInt(e.target.value, 10))}
            >
              {[1, 2, 3, 4, 5].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.controlGroup}>
            <label htmlFor="pickSeed" className={styles.label}>
              Seed
            </label>
            <input
              id="pickSeed"
              type="text"
              className={styles.input}
              placeholder="random"
              value={seed}
              onChange={(e) => handleSeedChange(e.target.value)}
            />
          </div>

          <label className={styles.checkboxWrapper} htmlFor="pickDeterministic">
            <input
              type="checkbox"
              id="pickDeterministic"
              className={styles.checkbox}
              checked={deterministic}
              onChange={(e) => setDeterministic(e.target.checked)}
            />
            <span className={styles.checkboxLabel}>Deterministic</span>
          </label>
        </div>

        <HandBlindPicker {...picker} onChange={setPicker} />

        <div className={styles.analyzeRow}>
          <button
            className={styles.analyzeButton}
            onClick={handleAnalyze}
            disabled={loading}
          >
            {loading && <span className={styles.spinner} />}
            {loading ? "Analyzing…" : "Analyze decisions"}
          </button>
        </div>
      </section>

      {loading && (
        <div className={styles.loading}>
          <span className={styles.spinner} />
          Running pre-play decisions…
        </div>
      )}

      {response && !loading && (
        <DecisionFlow
          decisions={response.decisions}
          targetSeat={response.scenario.seat}
        />
      )}
    </div>
  );
}
