"use client";

import React, { useCallback, useEffect, useState } from "react";
import { ds } from "../../../lib/ds";
import { accountErrorMessage, fetchHands } from "../../../lib/account";
import { shortDateTime, signed } from "../../../lib/format";
import { latestReviews } from "../../../lib/handHistory";
import { HISTORY_VIEWS, historyHref } from "../../../lib/reviewLinks";
import type { HandHistoryRow } from "../../../lib/types";
import AccountShell from "../AccountShell";
import BetaTag from "../components/BetaTag";
import Segmented from "../components/Segmented";
import VerifiedOnly from "../components/VerifiedOnly";
import { useLinkedView } from "../useLinkedView";
import { useRequiredAccount } from "../useRequiredAccount";
import accountStyles from "../account.module.css";
import AiRow from "./AiRow";
import HandRole from "./HandRole";
import styles from "./history.module.css";

/** Every finished hand the account played, newest first, a page at a time. */
export default function HandHistoryPage() {
  const { account } = useRequiredAccount();

  return (
    <AccountShell tab="history" account={account} wide>
      <VerifiedOnly account={account} to="see your hand history">
        <History />
      </VerifiedOnly>
    </AccountShell>
  );
}

// Seconds between looks for reviews that were still waiting.
const POLL_SECONDS = 20;

function History() {
  const [hands, setHands] = useState<HandHistoryRow[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, show] = useLinkedView(HISTORY_VIEWS, historyHref);

  const load = useCallback(async (before: string | null) => {
    setBusy(true);
    setError(null);
    try {
      const page = await fetchHands(before);
      setHands((prev) => (before ? [...prev, ...page.hands] : page.hands));
      setCursor(page.next_cursor ?? null);
      setLoaded(true);
    } catch (err) {
      setError(accountErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void load(null);
  }, [load]);

  // Reviews arrive on their own: while a hand on show is still waiting,
  // look again now and then and take in whatever has been reviewed since.
  // The key is the hands waiting, so the looking stops once none are and
  // starts over when older hands are shown.
  const waiting = hands
    .filter((hand) => hand.review?.status === "queued")
    .map((hand) => hand.game_id)
    .join(",");
  useEffect(() => {
    if (!waiting) return;
    const timer = window.setInterval(() => {
      latestReviews(waiting.split(",")).then(
        (fresh) =>
          setHands((prev) =>
            prev.map((h) =>
              fresh.has(h.game_id)
                ? { ...h, review: fresh.get(h.game_id) ?? null }
                : h,
            ),
          ),
        // A missed look is harmless; the next one will do.
        () => undefined,
      );
    }, POLL_SECONDS * 1000);
    return () => window.clearInterval(timer);
  }, [waiting]);

  return (
    <>
      {error && <p className={accountStyles.error}>{error}</p>}
      <div className={styles.viewBar}>
        <Segmented
          size="large"
          label="Table view"
          options={[
            { value: "results", label: "Results" },
            {
              value: "ai",
              label: (
                <>
                  AI comparison
                  <BetaTag />
                </>
              ),
            },
          ]}
          value={view}
          onChange={show}
        />
        {view === "ai" && (
          <p className={styles.viewNote}>
            Expected score your choices lost or gained against the AI model.
          </p>
        )}
      </div>
      <div className={styles.scroller}>
        {view === "results" ? (
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col">Date</th>
                <th scope="col">Table</th>
                <th scope="col">Role</th>
                <th scope="col" className={styles.num}>
                  Score
                </th>
                <th scope="col" className={styles.mult}>
                  <span className={styles.srOnly}>Stakes</span>
                </th>
                <th scope="col" className={styles.num}>
                  Points
                </th>
                <th scope="col">Opponents</th>
                <th scope="col">
                  <span className={styles.srOnly}>Notes</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {hands.map((hand) => (
                <Row key={hand.game_id} hand={hand} />
              ))}
            </tbody>
          </table>
        ) : (
          <table className={`${styles.table} ${styles.aiTable}`}>
            <thead>
              <tr>
                <th scope="col">Date</th>
                <th scope="col" className={styles.roleCell}>
                  Role
                </th>
                <th scope="col" className={styles.num}>
                  Score
                </th>
                <th scope="col" className={styles.barCell}>
                  <span className={styles.barHead}>
                    <span data-tone="cost">Cost</span>
                    <span data-tone="gain">Gained</span>
                  </span>
                </th>
                <th scope="col" className={styles.num}>
                  Net vs AI
                </th>
                <th scope="col">
                  <span className={styles.srOnly}>Review</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {hands.map((hand) => (
                <AiRow key={hand.game_id} hand={hand} />
              ))}
            </tbody>
          </table>
        )}
        {loaded && hands.length === 0 && (
          <p className={styles.empty}>
            No finished hands yet. They show up here once a hand is played out.
          </p>
        )}
      </div>
      {cursor && (
        <div className={accountStyles.actions}>
          <button
            type="button"
            className={`${ds.btn} ${ds.btnSm}`}
            disabled={busy}
            onClick={() => void load(cursor)}
          >
            {busy ? "Loading…" : "Show older"}
          </button>
        </div>
      )}
    </>
  );
}

function Row({ hand }: { hand: HandHistoryRow }) {
  const zeroed = hand.counted_score !== hand.score;
  const { humans, ai } = hand.opponents;
  return (
    <tr className={hand.counted_score < 0 ? styles.loss : undefined}>
      <td className={styles.when}>{shortDateTime(hand.time_closed)}</td>
      <td className={styles.tone}>
        <span className={styles.tableName} title={hand.table_name}>
          {hand.table_name}
        </span>
      </td>
      <td className={styles.tone}>
        <HandRole hand={hand} />
      </td>
      <td className={`${styles.num} ${styles.tone}`}>
        {zeroed ? (
          <span title="Abandoned: a win counts as 0">
            <s className={styles.muted}>{signed(hand.score)}</s> → 0
          </span>
        ) : (
          signed(hand.score)
        )}
      </td>
      <td className={`${styles.mult} ${styles.tone}`}>
        {hand.multiplier > 1 && (
          <span
            className={styles.muted}
            title={`Played for ${hand.multiplier}× stakes (included in the score)`}
          >
            {hand.multiplier}×
          </span>
        )}
      </td>
      <td className={`${styles.num} ${styles.tone}`}>
        {hand.points_taken}
        <span className={styles.muted}> {hand.points_scope}</span>
      </td>
      <td className={styles.muted}>
        {humans} human · {ai} AI
      </td>
      <td>
        {hand.abandoned ? (
          <span
            className={`${ds.badge} ${ds.badgeAccent2}`}
            title="The AI made too many of your decisions in this hand"
          >
            Abandoned
          </span>
        ) : hand.ai_assisted ? (
          <span
            className={`${ds.badge} ${ds.badgeQuiet}`}
            title="The AI made some of your decisions in this hand"
          >
            AI-assisted
          </span>
        ) : null}
      </td>
    </tr>
  );
}
