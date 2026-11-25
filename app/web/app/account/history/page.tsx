"use client";

import React, { useCallback, useEffect, useState } from "react";
import { ds } from "../../../lib/ds";
import {
  accountErrorMessage,
  fetchHandReview,
  fetchHands,
} from "../../../lib/account";
import {
  historyHref,
  netTone,
  type HistoryView as View,
} from "../../../lib/review";
import { latestReviews } from "../../../lib/handHistory";
import type { HandReview } from "../../../lib/types";
import BetaTag from "../components/BetaTag";
import CostGainBar from "../components/CostGainBar";
import Segmented from "../components/Segmented";
import HandStrip from "./HandStrip";
import type { HandHistoryRow } from "../../../lib/types";
import AccountShell from "../AccountShell";
import VerifiedOnly from "../components/VerifiedOnly";
import { useRequiredAccount } from "../useRequiredAccount";
import { signed } from "../../../lib/format";
import accountStyles from "../account.module.css";
import styles from "./history.module.css";

const ROLE_LABEL: Record<HandHistoryRow["role"], string> = {
  picker: "Picker",
  partner: "Partner",
  defender: "Defender",
  leaster: "Leaster",
};

const WHEN = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

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

/** The view a link asked for (?view=ai); results otherwise. */
function linkedView(): View {
  const asked = new URLSearchParams(window.location.search).get("view");
  return asked === "ai" ? "ai" : "results";
}

// Seconds between looks for reviews that were still waiting.
const POLL_SECONDS = 20;

function History() {
  const [hands, setHands] = useState<HandHistoryRow[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>(linkedView);
  // The address follows the view, so a link (or the way back from a
  // hand's review) can open the AI comparison.
  const show = (next: View) => {
    setView(next);
    window.history.replaceState(null, "", historyHref(next));
  };

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

const WHY_NOT: Record<string, string> = {
  skipped: "The AI made some of your decisions in this hand",
  failed: "This hand could not be reviewed",
};

/** A hand in the AI comparison: what its choices cost and gained against
 * the AI's, and the net. Opens to a strip of its decisions. */
function AiRow({ hand }: { hand: HandHistoryRow }) {
  const [open, setOpen] = useState(false);
  const [review, setReview] = useState<HandReview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const status = hand.review?.status ?? null;
  const reviewed = status === "reviewed";
  const cost = hand.review?.cost ?? 0;
  const gained = hand.review?.gained ?? 0;
  const handNet = hand.review?.net ?? 0;
  // Most hands differ from the AI's line a little each way and come out
  // near even. Those are drawn quietly, so the few where a decision moved
  // the hand a full point are the ones the eye finds.
  const notable = hand.review?.notable ?? false;
  const lineTone = hand.counted_score < 0 ? styles.loss : "";

  const toggle = () => {
    setOpen(!open);
    if (!open && !review) {
      setError(null);
      fetchHandReview(hand.game_id).then(setReview, (err) =>
        setError(accountErrorMessage(err)),
      );
    }
  };
  const role = (
    <>
      {ROLE_LABEL[hand.role]}
      {hand.alone && hand.role === "picker" && (
        <span className={styles.muted}> · alone</span>
      )}
    </>
  );

  return (
    <>
      {/* The whole row opens it; the button is the same control for the
          keyboard and for screen readers. */}
      <tr
        className={`${lineTone} ${open ? styles.openRow : ""} ${reviewed ? styles.opens : ""}`}
        onClick={reviewed ? toggle : undefined}
      >
        <td className={styles.when}>
          {WHEN.format(new Date(hand.time_closed))}
          <span className={`${styles.roleUnder} ${styles.tone}`}>{role}</span>
        </td>
        <td className={`${styles.tone} ${styles.roleCell}`}>{role}</td>
        <td className={`${styles.num} ${styles.tone}`}>{signed(hand.score)}</td>
        {reviewed ? (
          <>
            <td className={styles.barCell}>
              <CostGainBar cost={cost} gained={gained} quiet={!notable} />
            </td>
            <td
              className={`${styles.num} ${styles.net} ${notable ? styles.netNotable : ""}`}
              data-tone={netTone(handNet, notable)}
            >
              {cost === 0 && gained === 0 ? "—" : signed(handNet, 1)}
            </td>
            <td className={styles.expandCell}>
              <button
                type="button"
                className={styles.expand}
                aria-expanded={open}
                aria-label={`Cost ${cost.toFixed(1)}, gained ${gained.toFixed(1)}, net ${signed(handNet, 1)} points of expected score. ${open ? "Hide" : "Show"} decisions`}
                onClick={(event) => {
                  // The row would toggle it straight back.
                  event.stopPropagation();
                  toggle();
                }}
              >
                <span className={open ? styles.caretOpen : undefined}>▾</span>
              </button>
            </td>
          </>
        ) : (
          <td colSpan={3} className={styles.pending}>
            {status === "queued" ? (
              <span role="status">Waiting for review</span>
            ) : (
              <span title={status ? WHY_NOT[status] : "Not reviewed"}>—</span>
            )}
          </td>
        )}
      </tr>
      {open && (
        <tr className={styles.stripRow}>
          <td colSpan={6}>
            {review ? (
              <HandStrip review={review} />
            ) : (
              <p className={error ? accountStyles.error : accountStyles.note}>
                {error ?? "Loading…"}
              </p>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

function Row({ hand }: { hand: HandHistoryRow }) {
  const zeroed = hand.counted_score !== hand.score;
  const { humans, ai } = hand.opponents;
  return (
    <tr className={hand.counted_score < 0 ? styles.loss : undefined}>
      <td className={styles.when}>{WHEN.format(new Date(hand.time_closed))}</td>
      <td className={styles.tone}>
        <span className={styles.tableName} title={hand.table_name}>
          {hand.table_name}
        </span>
      </td>
      <td className={styles.tone}>
        {ROLE_LABEL[hand.role]}
        {hand.alone && hand.role === "picker" && (
          <span className={styles.muted}> · alone</span>
        )}
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
