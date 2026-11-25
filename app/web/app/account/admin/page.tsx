"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import AccountShell from "../AccountShell";
import accountStyles from "../account.module.css";
import {
  accountErrorMessage,
  fetchReviewQueue,
  useAccount,
} from "../../../lib/account";
import type { ReviewQueue } from "../../../lib/types";
import styles from "./admin.module.css";

const WHEN = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

// Seconds between refreshes while the page is open.
const REFRESH_SECONDS = 15;

function seconds(ms: number | null | undefined): string {
  return ms == null ? "—" : `${(ms / 1000).toFixed(1)} s`;
}

/** How long ago, to the nearest sensible unit. */
function age(since: string, now: number): string {
  const minutes = Math.max(0, (now - new Date(since).getTime()) / 60000);
  if (minutes < 1) return "under a minute";
  if (minutes < 90) return `${Math.round(minutes)} min`;
  if (minutes < 36 * 60) return `${Math.round(minutes / 60)} h`;
  return `${Math.round(minutes / 1440)} days`;
}

/** The admin area. For now, the state of the AI review queue. */
export default function AdminPage() {
  const router = useRouter();
  const { me } = useAccount();
  const account = me?.account ?? null;
  const allowed = !!account?.is_admin;

  useEffect(() => {
    if (me !== undefined && !allowed) router.replace("/account");
  }, [me, allowed, router]);

  return (
    <AccountShell
      title={account ? `@${account.username}` : "Your account"}
      tab="admin"
      account={account}
      wide
    >
      {allowed ? <Queue /> : <p className={accountStyles.note}>Loading…</p>}
    </AccountShell>
  );
}

function Queue() {
  const [queue, setQueue] = useState<ReviewQueue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const load = () =>
      fetchReviewQueue().then(
        (data) => {
          setQueue(data);
          setNow(Date.now());
          setError(null);
        },
        (err) => setError(accountErrorMessage(err)),
      );
    void load();
    const timer = window.setInterval(load, REFRESH_SECONDS * 1000);
    return () => window.clearInterval(timer);
  }, []);

  if (!queue) {
    return (
      <p className={error ? accountStyles.error : accountStyles.note}>
        {error ?? "Loading…"}
      </p>
    );
  }

  const tables = `${queue.tables_in_play} ${queue.tables_in_play === 1 ? "table" : "tables"} in play`;
  const state = queue.paused
    ? `Paused · ${tables}`
    : queue.waiting > 0
      ? `Running · ${tables}`
      : `Idle · ${tables}`;

  return (
    <section aria-label="AI review queue">
      {error && <p className={accountStyles.error}>{error}</p>}
      <div className={styles.sectionHead}>
        <span>AI review queue</span>
        <span
          className={queue.paused ? styles.paused : undefined}
          role="status"
        >
          {state}
        </span>
      </div>
      <dl className={styles.figures}>
        <div>
          <dt>Waiting</dt>
          <dd>{queue.waiting}</dd>
          <dd className={styles.figureNote}>hands in the queue</dd>
        </div>
        <div>
          <dt>Oldest waiting</dt>
          <dd>{queue.oldest_queued ? age(queue.oldest_queued, now) : "—"}</dd>
          <dd className={styles.figureNote}>
            {queue.oldest_queued
              ? `queued ${WHEN.format(new Date(queue.oldest_queued))}`
              : "nothing waiting"}
          </dd>
        </div>
        <div>
          <dt>Reviewed, last 24 h</dt>
          <dd>{queue.reviewed_24h}</dd>
          <dd className={styles.figureNote}>hands</dd>
        </div>
        <div>
          <dt>Average time</dt>
          <dd>{seconds(queue.average_ms_24h)}</dd>
          <dd className={styles.figureNote}>per hand, last 24 h</dd>
        </div>
      </dl>
      <dl className={styles.rows}>
        <div>
          <dt>Reviewed in total</dt>
          <dd>
            {queue.reviewed} of {queue.total} queued
          </dd>
        </div>
        <div>
          <dt>Not reviewable (the AI made a call or bury)</dt>
          <dd>{queue.skipped}</dd>
        </div>
        <div>
          <dt>Failed to replay</dt>
          <dd>{queue.failed}</dd>
        </div>
        <div>
          <dt>Slowest hand, last 24 h</dt>
          <dd>{seconds(queue.slowest_ms_24h)}</dd>
        </div>
        <div>
          <dt>Last hand reviewed</dt>
          <dd>
            {queue.last_reviewed
              ? WHEN.format(new Date(queue.last_reviewed))
              : "—"}
          </dd>
        </div>
        <div>
          <dt>Reviewing with</dt>
          <dd>{queue.ai_model}</dd>
        </div>
        <div>
          <dt>Reviews from another AI version</dt>
          <dd>{queue.other_model_reviews}</dd>
        </div>
      </dl>
      {queue.failures.length > 0 && (
        <>
          <h2 className={styles.subHead}>Failed hands</h2>
          <ul className={styles.failures}>
            {queue.failures.map((f) => (
              <li key={`${f.game_id}-${f.seat}`}>
                <span className={styles.mono}>
                  {f.game_id} · seat {f.seat}
                </span>
                <span>{f.reason ?? "no reason recorded"}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <p className={styles.footnote}>
        The queue runs while at most one table is in play. This page refreshes
        every {REFRESH_SECONDS} seconds.
      </p>
    </section>
  );
}
