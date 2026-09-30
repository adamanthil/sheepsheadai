"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ds } from "../../../lib/ds";
import {
  accountErrorMessage,
  fetchHands,
  useAccount,
} from "../../../lib/account";
import type { HandHistoryRow } from "../../../lib/types";
import AccountShell from "../AccountShell";
import { signed } from "../components/StatsPanel";
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
  const router = useRouter();
  const { me } = useAccount();
  const account = me?.account ?? null;

  useEffect(() => {
    if (me !== undefined && !account) router.replace("/account");
  }, [me, account, router]);

  let body: React.ReactNode;
  if (!account) {
    body = <p className={accountStyles.note}>Loading…</p>;
  } else if (!account.email_verified) {
    body = (
      <p className={accountStyles.note}>
        Confirm your email to see your hand history.{" "}
        <Link href="/account" className={ds.link}>
          Back to your account
        </Link>
      </p>
    );
  } else {
    body = <History />;
  }

  return (
    <AccountShell
      title={account ? `@${account.username}` : "Your account"}
      tab="history"
      signedIn={!!account}
      wide
    >
      {body}
    </AccountShell>
  );
}

function History() {
  const [hands, setHands] = useState<HandHistoryRow[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <>
      <p className={styles.caption}>
        Points are card points taken: your team&rsquo;s, or your own in a
        leaster.
      </p>
      {error && <p className={accountStyles.error}>{error}</p>}
      <div className={styles.scroller}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Date</th>
              <th scope="col">Table</th>
              <th scope="col">Role</th>
              <th scope="col" className={styles.num}>
                Score
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
    <tr>
      <td className={styles.when}>{WHEN.format(new Date(hand.time_closed))}</td>
      <td>
        <span className={styles.tableName} title={hand.table_name}>
          {hand.table_name}
        </span>
      </td>
      <td>
        {ROLE_LABEL[hand.role]}
        {hand.alone && hand.role === "picker" && (
          <span className={styles.muted}> · alone</span>
        )}
      </td>
      <td className={styles.num}>
        {zeroed ? (
          <span title="Abandoned: a win counts as 0">
            <s className={styles.muted}>{signed(hand.score)}</s> → 0
          </span>
        ) : (
          signed(hand.score)
        )}
        {hand.multiplier > 1 && (
          <span
            className={styles.muted}
            title={`Played for ${hand.multiplier}× stakes (included in the score)`}
          >
            {" "}
            · {hand.multiplier}×
          </span>
        )}
      </td>
      <td className={styles.num}>
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
