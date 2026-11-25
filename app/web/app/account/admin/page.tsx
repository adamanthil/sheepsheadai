"use client";

import React, { useEffect, useState } from "react";
import AccountShell from "../AccountShell";
import { useRequiredAccount } from "../useRequiredAccount";
import accountStyles from "../account.module.css";
import { accountErrorMessage, fetchReviewQueue } from "../../../lib/account";
import type { ReviewQueue } from "../../../lib/types";
import { BigFigure, BigFigures } from "../components/BigFigures";
import SectionHead from "../components/SectionHead";
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
  const { account } = useRequiredAccount((a) => a.is_admin);

  return (
    <AccountShell tab="admin" account={account} wide>
      {account ? <Queue /> : <p className={accountStyles.note}>Loading…</p>}
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
      <SectionHead
        title="AI review queue"
        aside={
          <span
            className={queue.paused ? styles.paused : undefined}
            role="status"
          >
            {state}
          </span>
        }
      />
      <BigFigures minWidth={170}>
        <BigFigure label="Waiting" note="hands in the queue">
          {queue.waiting}
        </BigFigure>
        <BigFigure
          label="Oldest waiting"
          note={
            queue.oldest_queued
              ? `queued ${WHEN.format(new Date(queue.oldest_queued))}`
              : "nothing waiting"
          }
        >
          {queue.oldest_queued ? age(queue.oldest_queued, now) : "—"}
        </BigFigure>
        <BigFigure label="Reviewed, last 24 h" note="hands">
          {queue.reviewed_24h}
        </BigFigure>
        <BigFigure label="Average time" note="per hand, last 24 h">
          {seconds(queue.average_ms_24h)}
        </BigFigure>
      </BigFigures>
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
          <h2 className={accountStyles.subHead}>Failed hands</h2>
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
      <p className={accountStyles.footnote}>
        The queue runs while at most one table is in play. This page refreshes
        every {REFRESH_SECONDS} seconds.
      </p>
    </section>
  );
}
