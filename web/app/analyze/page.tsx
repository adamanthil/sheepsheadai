"use client";

import React, { useState } from 'react';
import { AnalyzeSimulateRequest, AnalyzeSimulateResponse, AnalyzeGameSummary } from '../../lib/analyzeTypes';
import ActionTimeline from './ActionTimeline';
import GameSummary from './GameSummary';
import styles from './page.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || (() => {
  if (typeof window === 'undefined') return 'http://localhost:9000';

  // Use the same hostname as the frontend, but with backend port
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  return `${protocol}//${hostname}:9000`;
})();

export default function AnalyzePage() {
  // Form state
  const [partnerMode, setPartnerMode] = useState<number>(1);
  const [deterministic, setDeterministic] = useState<boolean>(true);
  const [seed, setSeed] = useState<string>('');

  // Results state
  const [response, setResponse] = useState<AnalyzeSimulateResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSimulate = async () => {
    setError(null);
    setLoading(true);
    setResponse(null);

    try {
      const requestBody: AnalyzeSimulateRequest = {
        partnerMode,
        deterministic,
        seed: seed ? parseInt(seed, 10) : undefined,
        maxSteps: 200
      };

      const res = await fetch(`${API_BASE}/api/analyze/simulate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
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
      console.error('Simulate error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to simulate game. Please try again.';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleSeedChange = (value: string) => {
    // Allow empty string or valid integers
    if (value === '' || /^\d+$/.test(value)) {
      setSeed(value);
    }
  };

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.title}>🧠 AI Model Analysis</h1>
        <p className={styles.subtitle}>
          Simulate a Sheepshead game and explore the AI's decisions.
        </p>
      </div>

      {/* Error Display */}
      {error && (
        <div className={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Controls Panel */}
      <div className={styles.controlsPanel}>
        <h2 className={styles.controlsTitle}>
          ⚙️ Simulation Settings
        </h2>

        <div className={styles.controlsGrid}>
          <div className={styles.controlGroup}>
            <label htmlFor="partnerMode" className={styles.label}>
              👥 Partner Mode
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
              🎲 Random Seed (optional)
            </label>
            <input
              id="seed"
              type="text"
              className={styles.input}
              placeholder="Leave empty for random"
              value={seed}
              onChange={(e) => handleSeedChange(e.target.value)}
            />
          </div>

          <div className={styles.controlGroup}>
            <label className={styles.label}>
              🎯 Decision Mode
            </label>
            <div className={styles.checkboxWrapper}>
              <input
                type="checkbox"
                id="deterministic"
                className={styles.checkbox}
                checked={deterministic}
                onChange={(e) => setDeterministic(e.target.checked)}
              />
              <label htmlFor="deterministic" className={styles.checkboxLabel}>
                Deterministic (greedy)
              </label>
            </div>
          </div>


        </div>

        <div className={styles.buttonRow}>
          <button
            className={styles.simulateButton}
            onClick={handleSimulate}
            disabled={loading}
          >
            {loading && <div className={styles.spinner} />}
            {loading ? 'Simulating...' : '🎮 Simulate Game'}
          </button>
        </div>
      </div>

      {/* Results Panel */}
      {(response || loading) && (
        <div className={styles.resultsPanel}>
          <div className={styles.resultsHeader}>
            <h2 className={styles.resultsTitle}>
              📈 Game Analysis
            </h2>
            {response && (
              <div className={styles.meta}>
                <div className={styles.metaItem}>
                  <span>Steps: {response.trace.length}</span>
                </div>
                <div className={styles.metaItem}>
                  <span>Partner: {response.meta.partnerMode === 1 ? 'Called Ace' : 'Jack of Diamonds'}</span>
                </div>
                <div className={styles.metaItem}>
                  <span>Mode: {response.meta.deterministic ? 'Deterministic' : 'Stochastic'}</span>
                </div>
                {response.meta.seed && (
                  <div className={styles.metaItem}>
                    <span>Seed: {response.meta.seed}</span>
                  </div>
                )}
                {response.final && (
                  <div className={styles.metaItem}>
                    <span>
                      {response.final.mode === 'leaster'
                        ? `Leaster - Winner: Seat ${response.final.winner}`
                        : `Picker: Seat ${response.final.picker}, Partner: Seat ${response.final.partner}`
                      }
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {loading ? (
            <div className={styles.loading}>
              <div className={styles.spinner} />
              Analyzing game decisions...
            </div>
          ) : response ? (
            <>
              {response.summary && (
                <GameSummary summary={response.summary} />
              )}
              <ActionTimeline
                trace={response.trace}
                picker={response.summary?.picker}
                partner={response.summary?.partner}
              />
            </>
          ) : null}
        </div>
      )}
    </div>
  );
}
