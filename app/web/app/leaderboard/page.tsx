"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ds } from "../../lib/ds";
import { apiFetch } from "../../lib/api";
import { HttpStatusError } from "../../lib/apiError";
import { reportCaught } from "../../lib/reportCaught";
import { useAccount } from "../../lib/account";
import type {
  Leaderboard,
  LeaderboardBoard,
  LeaderboardPeriod,
  LeaderboardRow,
  LeaderboardSort,
} from "../../lib/types";
import PageShell from "../components/PageShell";
import styles from "./leaderboard.module.css";

/** Hands and pick rate are shown but can't be ranked by: the server
 * doesn't offer them. */
type ColumnKey = LeaderboardSort | "hands" | "pick_pct";
const COLUMNS: { key: ColumnKey; label: string; short: string }[] = [
  { key: "sph", label: "Score / hand", short: "Score/h" },
  { key: "total", label: "Total Score", short: "Total" },
  { key: "win_pct", label: "Win rate", short: "Win" },
  { key: "pick_pct", label: "Pick rate", short: "Pick" },
  { key: "hands", label: "Hands", short: "Hands" },
];
const isSort = (key: ColumnKey): key is LeaderboardSort =>
  key !== "hands" && key !== "pick_pct";
const SORTS = COLUMNS.map((c) => c.key).filter(isSort);

/** Each board counts only its own kind of hand; `hands` names that kind
 * in the captions. */
const BOARDS: {
  key: LeaderboardBoard;
  label: string;
  hands: string;
}[] = [
  {
    key: "overall",
    label: "Overall",
    hands: "finished hands",
  },
  {
    key: "called_ace",
    label: "Called Ace",
    hands: "finished called-ace hands",
  },
  {
    key: "jd",
    label: "Jack of Diamonds",
    hands: "finished jack-of-diamonds hands",
  },
  {
    key: "vs_ai",
    label: "Vs AI",
    hands: "finished hands against only the AI",
  },
];
const DEFAULT_BOARD: LeaderboardBoard = "overall";
const DEFAULT_SORT: LeaderboardSort = "sph";

const boardInfo = (key: LeaderboardBoard) => BOARDS.find((b) => b.key === key)!;

/** Monthly boards reset at midnight on the 1st, Central time (the server
 * holds the bounds; this zone only names them). In switch order: the
 * current month first, then back in time. */
const PERIODS: LeaderboardPeriod[] = ["month", "last_month", "all_time"];
const DEFAULT_PERIOD: LeaderboardPeriod = "month";
const BOARD_TZ = "America/Chicago";
const MONTH_NAME = new Intl.DateTimeFormat("en-US", {
  month: "long",
  timeZone: "UTC",
});
const YEAR_MONTH = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "numeric",
  timeZone: BOARD_TZ,
});
const RESET_DAY = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  timeZone: BOARD_TZ,
});

type MonthNames = { month: string; lastMonth: string };

/** The current and previous months' names, as of ``now`` read in Central
 * time. */
function monthNames(now: Date): MonthNames {
  const parts = YEAR_MONTH.formatToParts(now);
  const part = (type: string) =>
    Number(parts.find((p) => p.type === type)!.value);
  const [year, month] = [part("year"), part("month")];
  // Mid-month in UTC is safely inside the same month anywhere.
  const name = (back: number) =>
    MONTH_NAME.format(new Date(Date.UTC(year, month - 1 - back, 15)));
  return { month: name(0), lastMonth: name(1) };
}

/** Month names for a board: as of its served bounds once loaded, so they
 * agree with the server across a rollover; the viewer's clock until then. */
function boardMonths(board: Leaderboard | null): MonthNames {
  const current =
    board?.period === "last_month" ? board.period_end : board?.period_start;
  return monthNames(current ? new Date(current) : new Date());
}

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

type View = {
  board: LeaderboardBoard;
  sort: LeaderboardSort;
  period: LeaderboardPeriod;
};

function viewFromUrl(): View {
  const params = new URLSearchParams(window.location.search);
  const b = params.get("board");
  const board = BOARDS.some((x) => x.key === b)
    ? (b as LeaderboardBoard)
    : DEFAULT_BOARD;
  const s = params.get("sort");
  const sort = SORTS.includes(s as LeaderboardSort)
    ? (s as LeaderboardSort)
    : DEFAULT_SORT;
  const p = params.get("period");
  const period = PERIODS.includes(p as LeaderboardPeriod)
    ? (p as LeaderboardPeriod)
    : DEFAULT_PERIOD;
  return { board, sort, period };
}

/** The view's URL, leaving out whatever is already the default. */
function viewUrl({ board, sort, period }: View): string {
  const params = new URLSearchParams();
  if (period !== DEFAULT_PERIOD) params.set("period", period);
  if (board !== DEFAULT_BOARD) params.set("board", board);
  if (sort !== DEFAULT_SORT) params.set("sort", sort);
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
          `/api/leaderboard?board=${view.board}&sort=${view.sort}&period=${view.period}`,
        );
        if (!res.ok) throw new HttpStatusError(res.status);
        const data = (await res.json()) as Leaderboard;
        if (!cancelled) {
          setBoard(data);
          setFailed(false);
        }
      } catch (err) {
        reportCaught(err, "leaderboard");
        if (!cancelled) setFailed(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [view]);

  const period = view?.period ?? DEFAULT_PERIOD;
  const names = boardMonths(board);
  const periodLabel: Record<LeaderboardPeriod, string> = {
    all_time: "All time",
    last_month: names.lastMonth,
    month: names.month,
  };

  const show = (next: View) => {
    if (next.board !== view?.board || next.period !== view?.period)
      setBoard(null);
    setView(next);
    window.history.replaceState(null, "", viewUrl(next));
  };
  const choose = (key: LeaderboardSort) =>
    show({ board: info.key, sort: key, period });
  const chooseBoard = (key: LeaderboardBoard) =>
    show({ board: key, sort: DEFAULT_SORT, period });
  const choosePeriod = (next: LeaderboardPeriod) =>
    view && next !== period && show({ ...view, period: next });

  return (
    <PageShell title="Leaderboard" wide>
      <div className={styles.boardBar}>
        <div className={styles.period} role="group" aria-label="Period">
          {PERIODS.map((p) => (
            <button
              key={p}
              type="button"
              aria-pressed={period === p}
              className={`${styles.periodBtn} ${period === p ? styles.periodActive : ""}`}
              onClick={() => choosePeriod(p)}
            >
              {periodLabel[p]}
            </button>
          ))}
        </div>
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
      </div>
      <p className={styles.caption}>
        {board ? (
          <Caption board={board} hands={info.hands} names={names} />
        ) : (
          "\u00a0"
        )}
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
            {
              {
                all_time: `Nobody has qualified yet. ${board.min_hands} ${info.hands} with a confirmed account gets you on the board.`,
                month: `Nobody has qualified for ${names.month} yet. ${board.min_hands} ${info.hands} this month with a confirmed account gets you on the board.`,
                last_month: `Nobody qualified in ${names.lastMonth}.`,
              }[board.period]
            }
          </p>
        )}
      </div>
      {board && (
        <YourStanding board={board} hands={info.hands} month={names.month} />
      )}
    </PageShell>
  );
}

/** What the board counts, and for a month when it closes or closed. */
function Caption({
  board,
  hands,
  names,
}: {
  board: Leaderboard;
  hands: string;
  names: MonthNames;
}) {
  const who = `Confirmed accounts with ${board.min_hands}+ ${hands}`;
  switch (board.period) {
    case "all_time":
      return <>{who} · top 20.</>;
    case "month":
      return (
        <>
          {who} in {names.month} · top 20.
          {board.period_end && (
            <span className={styles.reset}>
              Resets {RESET_DAY.format(new Date(board.period_end))} at midnight
              Central.
            </span>
          )}
        </>
      );
    case "last_month":
      return (
        <>
          Final standings for {names.lastMonth}. {who} that month · top 20.
        </>
      );
  }
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
function YourStanding({
  board,
  hands,
  month,
}: {
  board: Leaderboard;
  hands: string;
  month: string;
}) {
  const { me } = useAccount();
  const account = me?.account;
  const listed = board.you != null || board.rows.some((r) => r.is_you);

  // A finished month can't be joined any more.
  if (me === undefined || listed || board.period === "last_month") return null;
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
  } else if (!account.email_verified) {
    text = (
      <>
        Confirm your email to appear here.{" "}
        <Link href="/account" className={ds.link}>
          Your account
        </Link>
      </>
    );
  } else if (board.qualifies_in) {
    const n = board.qualifies_in;
    text = (
      <>
        @{account.username}: {n} more{" "}
        {n === 1 ? hands.replace(/\bhands\b/, "hand") : hands}
        {board.period === "month" && ` in ${month}`} to qualify.
      </>
    );
  } else {
    return null;
  }
  return <p className={styles.standing}>{text}</p>;
}
