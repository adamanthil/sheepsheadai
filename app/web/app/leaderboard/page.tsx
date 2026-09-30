"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ds } from "../../lib/ds";
import { apiFetch } from "../../lib/api";
import { useAccount } from "../../lib/account";
import type {
  AccountStats,
  Leaderboard,
  LeaderboardRow,
  LeaderboardSort,
} from "../../lib/types";
import PageShell from "../components/PageShell";
import styles from "./leaderboard.module.css";

const COLUMNS: { key: LeaderboardSort; label: string; short: string }[] = [
  { key: "hands", label: "Hands", short: "Hands" },
  { key: "total", label: "Total Score", short: "Total" },
  { key: "sph", label: "Score / hand", short: "Score/h" },
  { key: "win_pct", label: "Win rate", short: "Win" },
  { key: "pick_pct", label: "Pick rate", short: "Pick" },
];
const SORTS = COLUMNS.map((c) => c.key);
const DEFAULT_SORT: LeaderboardSort = "total";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const signed = (x: number) => (x > 0 ? `+${x}` : String(x));

function cell(row: LeaderboardRow, key: LeaderboardSort): string {
  switch (key) {
    case "hands":
      return String(row.hands);
    case "total":
      return signed(row.total);
    case "sph":
      return row.sph.toFixed(2);
    case "win_pct":
      return pct(row.win_pct);
    case "pick_pct":
      return pct(row.pick_pct);
  }
}

function sortFromUrl(): LeaderboardSort {
  const s = new URLSearchParams(window.location.search).get("sort");
  return SORTS.includes(s as LeaderboardSort)
    ? (s as LeaderboardSort)
    : DEFAULT_SORT;
}

/** The public board: one page, best first by the chosen column. Clicking a
 * header re-sorts by that column; there is deliberately no reverse order,
 * and nothing below the top of the board is ever served. */
export default function LeaderboardPage() {
  const [sort, setSort] = useState<LeaderboardSort | null>(null);
  const [board, setBoard] = useState<Leaderboard | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setSort(sortFromUrl());
  }, []);

  useEffect(() => {
    if (!sort) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await apiFetch(`/api/leaderboard?sort=${sort}`);
        if (!res.ok) throw new Error(String(res.status));
        const data = (await res.json()) as Leaderboard;
        if (!cancelled) {
          setBoard(data);
          setFailed(false);
        }
      } catch {
        if (!cancelled) setFailed(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sort]);

  const choose = (key: LeaderboardSort) => {
    setSort(key);
    const url = key === DEFAULT_SORT ? "/leaderboard" : `?sort=${key}`;
    window.history.replaceState(null, "", url);
  };

  return (
    <PageShell title="Leaderboard" wide>
      <p className={styles.caption}>
        Confirmed accounts with {board?.min_hands ?? 50}+ finished hands · top
        20.
      </p>
      {failed && (
        <p className={styles.error}>Couldn&rsquo;t load the leaderboard.</p>
      )}
      <div className={styles.scroller}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col" className={styles.rank}>
                #
              </th>
              <th scope="col" className={styles.player}>
                Player
              </th>
              {COLUMNS.map((c) => (
                <th
                  key={c.key}
                  scope="col"
                  className={styles.num}
                  aria-sort={sort === c.key ? "descending" : undefined}
                >
                  <button
                    type="button"
                    className={`${styles.sortButton} ${sort === c.key ? styles.sortActive : ""}`}
                    onClick={() => choose(c.key)}
                    title={`Rank by ${c.label.toLowerCase()}`}
                  >
                    <span className={styles.longLabel}>{c.label}</span>
                    <span className={styles.shortLabel}>{c.short}</span>
                    <span className={styles.arrow} aria-hidden="true">
                      {sort === c.key ? "▼" : ""}
                    </span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {board?.rows.map((row) => (
              <Row key={row.username} row={row} sort={sort} />
            ))}
            {board?.you && (
              <>
                <tr aria-hidden="true" className={styles.gap}>
                  <td colSpan={2 + COLUMNS.length}>⋯</td>
                </tr>
                <Row row={board.you} sort={sort} />
              </>
            )}
          </tbody>
        </table>
        {board && board.rows.length === 0 && (
          <p className={styles.empty}>
            Nobody has qualified yet. {board.min_hands} finished hands with a
            confirmed account gets you on the board.
          </p>
        )}
      </div>
      {board && <YourStanding board={board} />}
    </PageShell>
  );
}

function Row({
  row,
  sort,
}: {
  row: LeaderboardRow;
  sort: LeaderboardSort | null;
}) {
  return (
    <tr className={row.is_you ? styles.you : undefined}>
      <td className={styles.rank}>{row.rank}</td>
      <th scope="row" className={styles.player}>
        @{row.username}
        {row.is_you && (
          <span className={`${ds.badge} ${styles.youBadge}`}>You</span>
        )}
      </th>
      {COLUMNS.map((c) => (
        <td
          key={c.key}
          className={`${styles.num} ${sort === c.key ? styles.numActive : ""}`}
        >
          {cell(row, c.key)}
        </td>
      ))}
    </tr>
  );
}

/** A line for the viewer when they aren't on the board: how to get there. */
function YourStanding({ board }: { board: Leaderboard }) {
  const { me } = useAccount();
  const [stats, setStats] = useState<AccountStats | null>(null);
  const account = me?.account;
  const listed = board.you != null || board.rows.some((r) => r.is_you);
  const verified = !!account?.email_verified;

  useEffect(() => {
    if (!verified || listed) return;
    let cancelled = false;
    apiFetch("/api/account/stats")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data) setStats(data as AccountStats);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [verified, listed]);

  if (me === undefined || listed) return null;
  let text: React.ReactNode;
  if (!account) {
    text = (
      <>
        <Link href="/account" className={ds.link}>
          Create an account
        </Link>{" "}
        to put your name here.
      </>
    );
  } else if (!verified) {
    text = (
      <>
        Confirm your email to appear here.{" "}
        <Link href="/account" className={ds.link}>
          Your account
        </Link>
      </>
    );
  } else if (stats?.qualifies_in) {
    text = (
      <>
        @{account.username}: {stats.qualifies_in} more finished hand
        {stats.qualifies_in === 1 ? "" : "s"} to qualify.
      </>
    );
  } else {
    return null;
  }
  return <p className={styles.standing}>{text}</p>;
}
