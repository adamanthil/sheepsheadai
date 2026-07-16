"use client";

import React, { useState } from "react";
import type { components } from "../../../lib/api.gen";
import type { AnalyzeActionDetail } from "../../../lib/analyzeTypes";
import { apiFetch } from "../../../lib/api";
import { CardText, PlayingCard } from "../../../lib/ds";
import ProbabilityBar from "../ProbabilityBar";
import ActionInsights from "../ActionInsights";
import ObservationView from "../ObservationView";
import HandBlindPicker from "./HandBlindPicker";
import styles from "./page.module.css";

type AnalyzePickRequest = components["schemas"]["AnalyzePickRequest"];
type AnalyzePickResponse = Omit<
  components["schemas"]["AnalyzePickResponse"],
  "decisions"
> & { decisions: AnalyzeActionDetail[] };

const SEAT_NAMES = ["Dan", "Kyle", "Trevor", "John", "Andrew"];

const PHASE_LABELS: Record<string, string> = {
  pick: "Pick / Pass",
  partner: "Partner call",
  bury: "Bury / Under",
};

function DecisionCard({
  decision,
  targetSeat,
}: {
  decision: AnalyzeActionDetail;
  targetSeat: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const maxProb = Math.max(...decision.probabilities.map((p) => p.prob));
  const isTarget = decision.seat === targetSeat;

  return (
    <div
      className={`${styles.decisionCard} ${isTarget ? styles.decisionTarget : ""}`}
    >
      <div className={styles.decisionHeader}>
        <div className={styles.decisionWho}>
          <span className={styles.decisionSeat}>
            {decision.seatName}
            {isTarget && <span className={styles.targetTag}>target</span>}
          </span>
          <span className={styles.decisionPhase}>
            {PHASE_LABELS[decision.phase] ?? decision.phase}
          </span>
        </div>
        <div className={styles.decisionAction}>
          <CardText>{decision.action}</CardText>
        </div>
        <div className={styles.decisionMetrics}>
          <span
            className={styles.metricChip}
            title="Value estimate (critic)"
          >
            V{" "}
            {decision.valueEstimate >= 0
              ? `+${decision.valueEstimate.toFixed(3)}`
              : decision.valueEstimate.toFixed(3)}
          </span>
          {typeof decision.oracleValue === "number" && (
            <span
              className={styles.metricChip}
              title="Oracle (full-information) value"
            >
              V*{" "}
              {decision.oracleValue >= 0
                ? `+${decision.oracleValue.toFixed(3)}`
                : decision.oracleValue.toFixed(3)}
            </span>
          )}
          {typeof decision.winProb === "number" && (
            <span className={styles.metricChip} title="Win probability">
              Win {Math.round(decision.winProb * 100)}%
            </span>
          )}
        </div>
      </div>

      <div className={styles.probabilities}>
        {decision.probabilities.map((prob, i) => (
          <ProbabilityBar key={i} probability={prob} maxProb={maxProb} />
        ))}
      </div>

      <button
        type="button"
        className={styles.expandToggle}
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? "Hide details" : "Head predictions & observation"}
      </button>

      {expanded && (
        <div className={styles.decisionDetails}>
          <ActionInsights action={decision} />
          <ObservationView observation={decision.observation} />
        </div>
      )}
    </div>
  );
}

export default function PickAnalysisPage() {
  const [partnerMode, setPartnerMode] = useState<number>(1);
  const [seat, setSeat] = useState<number>(1);
  const [seed, setSeed] = useState<string>("");
  const [deterministic, setDeterministic] = useState<boolean>(true);
  const [hand, setHand] = useState<string[]>([]);
  const [blind, setBlind] = useState<string[]>([]);

  const [response, setResponse] = useState<AnalyzePickResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handIncomplete = hand.length > 0 && hand.length !== 6;
  const blindIncomplete = blind.length > 0 && blind.length !== 2;
  const canAnalyze = !loading && !handIncomplete && !blindIncomplete;

  const handleAnalyze = async () => {
    setError(null);
    setLoading(true);

    try {
      const requestBody: AnalyzePickRequest = {
        partnerMode,
        seat,
        deterministic,
        seed: seed ? parseInt(seed, 10) : undefined,
        hand: hand.length === 6 ? hand : undefined,
        blind: blind.length === 2 ? blind : undefined,
      };

      const res = await apiFetch("/api/analyze/pick", {
        method: "POST",
        body: JSON.stringify(requestBody),
      });

      if (!res.ok) {
        let errorMessage = `Request failed: ${res.status} ${res.statusText}`;
        try {
          const errorData = await res.json();
          if (errorData.detail) {
            errorMessage =
              typeof errorData.detail === "string"
                ? errorData.detail
                : JSON.stringify(errorData.detail);
          }
        } catch {
          // Ignore JSON parsing errors for error response
        }
        throw new Error(errorMessage);
      }

      setResponse(await res.json());
    } catch (err) {
      console.error("Pick analysis error:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to analyze scenario. Please try again.",
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

  const outcome = response?.outcome;

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
            passed. Leave hand or blind empty to deal them randomly.
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
              {SEAT_NAMES.map((name, i) => (
                <option key={name} value={i + 1}>
                  {i + 1} — {name}
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

        <HandBlindPicker
          hand={hand}
          blind={blind}
          onChange={(nextHand, nextBlind) => {
            setHand(nextHand);
            setBlind(nextBlind);
          }}
        />

        <div className={styles.analyzeRow}>
          {handIncomplete && (
            <span className={styles.validation}>
              Hand needs exactly 6 cards (or none for random).
            </span>
          )}
          {blindIncomplete && (
            <span className={styles.validation}>
              Blind needs exactly 2 cards (or none for random).
            </span>
          )}
          <button
            className={styles.analyzeButton}
            onClick={handleAnalyze}
            disabled={!canAnalyze}
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
        <>
          {/* Outcome strip */}
          {outcome && (
            <section
              className={`${styles.outcomeStrip} ${
                outcome.isLeaster
                  ? styles.outcomeLeaster
                  : outcome.pickerSeat === response.scenario.seat
                    ? styles.outcomePicked
                    : styles.outcomePassed
              }`}
            >
              <div className={styles.outcomeVerdict}>
                {outcome.isLeaster
                  ? "Everyone passed — leaster"
                  : outcome.pickerSeat === response.scenario.seat
                    ? `${response.scenario.seatName} picks`
                    : `${response.scenario.seatName} passes — ${outcome.pickerName} picks`}
              </div>
              <div className={styles.outcomeFacts}>
                {outcome.aloneCalled && (
                  <span className={styles.chip}>Alone</span>
                )}
                {outcome.calledCard && (
                  <span className={styles.chip}>
                    Called <CardText>{outcome.calledCard}</CardText>
                    {outcome.calledUnder && " (under)"}
                  </span>
                )}
                {outcome.underCard && (
                  <span className={styles.chip}>
                    Under <CardText>{outcome.underCard}</CardText>
                  </span>
                )}
                {outcome.bury.length > 0 && (
                  <span className={styles.chip}>
                    Buried{" "}
                    {outcome.bury.map((card, i) => (
                      <React.Fragment key={card}>
                        {i > 0 && " "}
                        <CardText>{card}</CardText>
                      </React.Fragment>
                    ))}
                  </span>
                )}
              </div>
            </section>
          )}

          {/* Scenario recap */}
          <section className={styles.recapPanel}>
            <div className={styles.recapGroup}>
              <span className={styles.recapLabel}>
                {response.scenario.seatName}&apos;s hand
              </span>
              <div className={styles.recapCards}>
                {response.scenario.hand.map((card) => (
                  <PlayingCard key={card} code={card} w={56} />
                ))}
              </div>
            </div>
            <div className={styles.recapGroup}>
              <span className={styles.recapLabel}>Blind</span>
              <div className={styles.recapCards}>
                {response.scenario.blind.map((card) => (
                  <PlayingCard key={card} code={card} w={56} />
                ))}
              </div>
            </div>
          </section>

          {/* Decisions */}
          <section className={styles.decisions}>
            {response.decisions.map((decision) => (
              <DecisionCard
                key={decision.stepIndex}
                decision={decision}
                targetSeat={response.scenario.seat}
              />
            ))}
          </section>
        </>
      )}
    </div>
  );
}
