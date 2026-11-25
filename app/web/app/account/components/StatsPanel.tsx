"use client";

import { signed } from "../../../lib/format";
import React, { useState } from "react";
import { ds } from "../../../lib/ds";
import type {
  AccountRunningScore,
  AccountStats,
  StatsMode,
} from "../../../lib/types";
import account from "../account.module.css";
import {
  RunningScoreChart,
  Meter,
  PickByPosition,
  RangePicker,
  RangeRows,
  RoleBars,
  runningScoreWindow,
  type Range,
} from "./StatsCharts";
import Segmented from "./Segmented";
import { count, perHand } from "./statsFormat";
import styles from "./StatsPanel.module.css";

function Head({ title, aside }: { title: string; aside?: React.ReactNode }) {
  return (
    <div className={`${ds.headRule} ${account.statsHead} ${styles.head}`}>
      <span className={ds.overline}>{title}</span>
      {aside != null && <span className={ds.overline}>{aside}</span>}
    </div>
  );
}

function Figure({
  label,
  value,
  big = false,
}: {
  label: string;
  value: string;
  big?: boolean;
}) {
  return (
    <div className={`${styles.figure} ${big ? styles.figureBig : ""}`}>
      <dt className={ds.overline}>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

const plural = (n: number, word: string) =>
  `${count(n)} ${word}${n === 1 ? "" : "s"}`;

/** The modes the stats can be narrowed to, named as the leaderboard
 * names their boards; `hand` names one of the mode's hands. */
export const STATS_MODES: { key: StatsMode; label: string; hand: string }[] = [
  { key: "all", label: "All games", hand: "hand" },
  { key: "called_ace", label: "Called Ace", hand: "called-ace hand" },
  { key: "jd", label: "Jack of Diamonds", hand: "jack-of-diamonds hand" },
];

/** The running-score chart with its range presets. */
function RunningScore({
  running,
  hand,
}: {
  running: AccountRunningScore | null;
  hand: string;
}) {
  const total = running?.scores.length ?? 0;
  const [range, setRange] = useState<Range | null>(null);
  // Recent hands by default; every hand until there are more than 100. A
  // chosen range outlives a mode switch only if the new mode offers it.
  const fits = range != null && (range === 0 || range < total);
  const shown: Range = fits ? range : total > 100 ? 100 : 0;
  const data = running && total ? runningScoreWindow(running, shown) : null;
  const net = data ? data.scores.reduce((a, b) => a + b, 0) : 0;
  return (
    <>
      <Head
        title="Running score"
        aside={
          data &&
          `${signed(net)} over ${shown === 0 ? "all" : "the last"} ${plural(data.scores.length, "hand")}`
        }
      />
      {running === null ? (
        <p className={account.note}>Loading…</p>
      ) : !data ? (
        <p className={account.note}>
          Your running score starts with your first finished {hand}.
        </p>
      ) : (
        <>
          <div className={styles.filters}>
            <RangePicker value={shown} onChange={setRange} total={total} />
          </div>
          <RunningScoreChart data={data} />
        </>
      )}
    </>
  );
}

/** The verified account's numbers. Desktop: totals, rates and picking
 * down the side of the running-score chart, then role and company side by side.
 * Narrow screens: one column, chart second (see StatsPanel.module.css).
 * `mode` is the one chosen, `statsMode` the one `stats` were loaded for:
 * they differ while a switch loads, and the words follow the numbers. */
export default function StatsPanel({
  mode,
  onMode,
  stats,
  statsMode,
  running,
  busy,
}: {
  mode: StatsMode;
  onMode: (mode: StatsMode) => void;
  stats: AccountStats;
  statsMode: StatsMode;
  running: AccountRunningScore | null;
  busy: boolean;
}) {
  const info = STATS_MODES.find((m) => m.key === statsMode)!;
  const board =
    statsMode === "all" ? "leaderboard" : `${info.label} leaderboard`;
  return (
    <div className={styles.panel} aria-busy={busy}>
      <div className={styles.mode}>
        <Segmented
          size="dense"
          label="Games counted"
          options={STATS_MODES.map((m) => ({ value: m.key, label: m.label }))}
          value={mode}
          onChange={onMode}
        />
      </div>

      <section aria-label="Your stats" className={styles.figures}>
        <Head
          title="Your stats"
          aside={stats.rank != null ? `Rank #${stats.rank}` : undefined}
        />
        <dl className={styles.figureGrid}>
          <Figure label="Score / hand" value={perHand(stats.sph)} big />
          <Figure label="Total score" value={signed(stats.total)} />
          <Figure label="Hands" value={count(stats.hands)} />
        </dl>
        <p className={account.note}>
          {stats.rank != null
            ? `Rank #${stats.rank} on the all-time ${board} by score per hand.`
            : `${plural(stats.qualifies_in ?? 0, `more finished ${info.hand}`)} to make the ${board}, which lists confirmed accounts with at least ${stats.min_hands}.`}
        </p>
      </section>

      <section aria-label="Running score" className={styles.running}>
        <RunningScore running={running} hand={info.hand} />
      </section>

      <section aria-label="Rates" className={styles.rates}>
        <Head title="Rates" />
        <div className={styles.meters}>
          <Meter label="Win rate" value={stats.win_pct} note="hands won" />
          <Meter
            label="Pick rate"
            value={stats.pick_pct}
            note="of every deal"
          />
          <Meter
            label="Pick when offered"
            value={stats.pick_offered_pct}
            note={`of ${plural(stats.offered_hands, "deal")} the pick reached you`}
          >
            {stats.offered_hands > 0 && (
              <PickByPosition positions={stats.pick_offers_by_position} />
            )}
          </Meter>
          <Meter
            label="Completed"
            value={stats.completion_rate}
            note="hands not abandoned*"
          />
        </div>
      </section>

      <section aria-label="When you pick" className={styles.picking}>
        <Head title="When you pick" />
        <dl className={styles.figureRow}>
          <Figure label="Picks" value={count(stats.picks)} />
          <Figure
            label="Trump / pick"
            value={stats.trump_per_pick?.toFixed(1) ?? "—"}
          />
          <Figure
            label="Queens / pick"
            value={stats.queens_per_pick?.toFixed(1) ?? "—"}
          />
        </dl>
        <p className={account.note}>
          Of the six cards dealt to you. Picks the AI made for you are left out.
        </p>
      </section>

      <div className={styles.pair}>
        <section aria-label="By role">
          <Head title="By role" />
          <RoleBars roles={stats.roles} hands={stats.hands} />
        </section>
        <section aria-label="Who you played with">
          <Head title="Who you played with" />
          <RangeRows
            rows={[
              { label: "Vs the AI", split: stats.vs_ai },
              { label: "With people", split: stats.with_people },
            ]}
          />
          <p className={account.note}>
            Vs the AI: all other seats played by the AI. The bar is a 95% range
            on score per hand, after {stats.split_margin_min_hands} hands; where
            ranges overlap, the gap could be luck.
          </p>
        </section>
      </div>

      <section aria-label="Hand completion" className={styles.completion}>
        <Head title="Hand completion" />
        <dl className={styles.figureRow}>
          <Figure label="Abandoned*" value={count(stats.abandoned_hands)} />
          <Figure
            label="Forfeited score*"
            value={String(stats.forfeited_score)}
          />
          <Figure label="Leasters" value={count(stats.leaster_hands)} />
        </dl>
        <p className={account.note}>
          * A hand counts as abandoned when the AI makes more than{" "}
          {stats.abandon_threshold} of your decisions in it (after a disconnect,
          leaving mid-hand, or turn timeouts). In an abandoned hand a loss
          counts in full, but a win counts as 0 and isn&rsquo;t a win. Moves the
          AI makes because the host closed the table, you were removed, or the
          server restarted aren&rsquo;t marked as abandoned.
        </p>
      </section>
    </div>
  );
}
