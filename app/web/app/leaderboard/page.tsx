"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ds } from "../../lib/ds";
import { apiFetch } from "../../lib/api";
import { useAccount } from "../../lib/account";
import type {
  AccountStats,
  Leaderboard,
  LeaderboardBoard,
  LeaderboardRow,
  LeaderboardSort,
} from "../../lib/types";
import PageShell from "../components/PageShell";
import styles from "./leaderboard.module.css";

/** Hands is shown but can't be ranked by: the server doesn't offer it. */
type ColumnKey = LeaderboardSort | "hands";
const COLUMNS: { key: ColumnKey; label: string; short: string }[] = [
  { key: "hands", label: "Hands", short: "Hands" },
  { key: "total", label: "Total Score", short: "Total" },
  { key: "sph", label: "Score / hand", short: "Score/h" },
  { key: "win_pct", label: "Win rate", short: "Win" },
  { key: "pick_pct", label: "Pick rate", short: "Pick" },
];
const isSort = (key: ColumnKey): key is LeaderboardSort => key !== "hands";
const SORTS = COLUMNS.map((c) => c.key).filter(isSort);

/** Each board counts only its own kind of hand; `hands` names that kind
 * in the captions. */
const BOARDS: {
  key: LeaderboardBoard;
  label: string;
  hands: string;
  defaultSort: LeaderboardSort;
}[] = [
  {
    key: "overall",
    label: "Overall",
    hands: "finished hands",
    defaultSort: "total",
  },
  {
    key: "called_ace",
    label: "Called Ace",
    hands: "finished called-ace hands",
    defaultSort: "total",
  },
  {
    key: "jd",
    label: "Jack of Diamonds",
    hands: "finished jack-of-diamonds hands",
    defaultSort: "total",
  },
  {
    key: "vs_ai",
    label: "Vs AI",
    hands: "finished hands against only the AI",
    defaultSort: "sph",
  },
];
const DEFAULT_BOARD: LeaderboardBoard = "overall";

const boardInfo = (key: LeaderboardBoard) => BOARDS.find((b) => b.key === key)!;

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const signed = (x: number) => (x > 0 ? `+${x}` : String(x));

function cell(row: LeaderboardRow, key: ColumnKey): string {
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

type View = { board: LeaderboardBoard; sort: LeaderboardSort };

function viewFromUrl(): View {
  const params = new URLSearchParams(window.location.search);
  const b = params.get("board");
  const board = BOARDS.some((x) => x.key === b)
    ? (b as LeaderboardBoard)
    : DEFAULT_BOARD;
  const s = params.get("sort");
  const sort = SORTS.includes(s as LeaderboardSort)
    ? (s as LeaderboardSort)
    : boardInfo(board).defaultSort;
  return { board, sort };
}

/** The view's URL, leaving out whatever is already the default. */
function viewUrl({ board, sort }: View): string {
  const params = new URLSearchParams();
  if (board !== DEFAULT_BOARD) params.set("board", board);
  if (sort !== boardInfo(board).defaultSort) params.set("sort", sort);
  const query = params.toString();
  return query ? `?${query}` : "/leaderboard";
}

/** The public boards: one page each, best first by the chosen column. Tabs
 * pick which hands count; clicking a header re-sorts by that column. There
 * is deliberately no reverse order, and nothing below the top of a board is
 * ever served. */
export default function LeaderboardPage() {
  const [view, setView] = useState<View | null>(null);
  const [board, setBoard] = useState<Leaderboard | null>(null);
  const [failed, setFailed] = useState(false);
  const sort = view?.sort ?? null;
  const info = boardInfo(view?.board ?? DEFAULT_BOARD);

  useEffect(() => {
    setView(viewFromUrl());
  }, []);

  useEffect(() => {
    if (!view) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await apiFetch(
          `/api/leaderboard?board=${view.board}&sort=${view.sort}`,
        );
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
  }, [view]);

  const show = (next: View) => {
    if (next.board !== view?.board) setBoard(null);
    setView(next);
    window.history.replaceState(null, "", viewUrl(next));
  };
  const choose = (key: LeaderboardSort) => show({ board: info.key, sort: key });
  const chooseBoard = (key: LeaderboardBoard) =>
    show({ board: key, sort: boardInfo(key).defaultSort });

  return (
    <PageShell title="Leaderboard" wide>
      <div className={styles.tabs} role="tablist">
        {BOARDS.map((b) => (
          <button
            key={b.key}
            type="button"
            role="tab"
            aria-selected={view?.board === b.key}
            className={`${styles.tab} ${view?.board === b.key ? styles.tabActive : ""}`}
            onClick={() => chooseBoard(b.key)}
          >
            {b.label}
          </button>
        ))}
      </div>
      <p className={styles.caption}>
        {board
          ? `Confirmed accounts with ${board.min_hands}+ ${info.hands} · top 20.`
          : "\u00a0"}
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
              {COLUMNS.map((c) => {
                const { key } = c;
                const label = (
                  <>
                    <span className={styles.longLabel}>{c.label}</span>
                    <span className={styles.shortLabel}>{c.short}</span>
                    <span className={styles.arrow} aria-hidden="true">
                      {sort === key ? "▼" : ""}
                    </span>
                  </>
                );
                return (
                  <th
                    key={key}
                    scope="col"
                    className={styles.num}
                    aria-sort={sort === key ? "descending" : undefined}
                  >
                    {isSort(key) ? (
                      <button
                        type="button"
                        className={`${styles.sortButton} ${sort === key ? styles.sortActive : ""}`}
                        onClick={() => choose(key)}
                        title={`Rank by ${c.label.toLowerCase()}`}
                      >
                        {label}
                      </button>
                    ) : (
                      <span className={styles.columnLabel}>{label}</span>
                    )}
                  </th>
                );
              })}
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
            Nobody has qualified yet. {board.min_hands} {info.hands} with a
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
  } else if (board.board === "overall" && stats?.qualifies_in) {
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
