import React from "react";
import { ds } from "../../../lib/ds";
import type { AccountStats } from "../../../lib/types";
import styles from "../account.module.css";

export const pct = (x: number | null | undefined) =>
  x == null ? "—" : `${(x * 100).toFixed(1)}%`;
export const signed = (x: number) => (x > 0 ? `+${x}` : String(x));

function Tiles({ tiles }: { tiles: [string, string][] }) {
  return (
    <dl className={styles.stats}>
      {tiles.map(([label, value]) => (
        <div key={label} className={styles.stat}>
          <dt className={ds.overline}>{label}</dt>
          <dd className={styles.statValue}>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** The verified account's numbers: score tiles (abandoned wins already
 * counted as 0), then how many hands they saw through, with the rule. */
export default function StatsPanel({ stats }: { stats: AccountStats }) {
  const tiles: [string, string][] = [
    ["Hands", String(stats.hands)],
    ["Total score", signed(stats.total)],
    ["Score / hand", stats.sph == null ? "—" : stats.sph.toFixed(2)],
    ["Win rate", pct(stats.win_pct)],
    ["Pick rate", pct(stats.pick_pct)],
    ["Leasters", String(stats.leaster_hands)],
  ];
  const completion: [string, string][] = [
    ["Completed*", pct(stats.completion_rate)],
    ["Abandoned*", String(stats.abandoned_hands)],
    ["Forfeited score*", String(stats.forfeited_score)],
  ];
  return (
    <section aria-label="Your stats">
      <div className={`${ds.headRule} ${styles.statsHead}`}>
        <span className={ds.overline}>Your stats</span>
        <span className={ds.overline}>
          {stats.rank != null
            ? `Rank #${stats.rank} by total`
            : `${stats.qualifies_in} more hand${stats.qualifies_in === 1 ? "" : "s"} to qualify`}
        </span>
      </div>
      <Tiles tiles={tiles} />
      <p className={styles.note}>
        The leaderboard lists confirmed accounts with at least {stats.min_hands}{" "}
        finished hands.
      </p>

      <div className={`${ds.headRule} ${styles.statsHead} ${styles.subHead}`}>
        <span className={ds.overline}>Hand completion</span>
      </div>
      <Tiles tiles={completion} />
      <p className={styles.note}>
        * A hand counts as abandoned when the AI makes more than{" "}
        {stats.abandon_threshold} of your decisions in it &mdash; after a
        disconnect, leaving mid-hand, or turn timeouts. In an abandoned hand a
        loss counts in full, but a win counts as 0 and isn&rsquo;t a win. Moves
        the AI makes because the host closed the table, you were removed, or the
        server restarted aren&rsquo;t marked as abandoned.
      </p>
    </section>
  );
}
