import React from "react";
import { ds } from "../../../lib/ds";
import type { AccountStats } from "../../../lib/types";
import styles from "../account.module.css";

type Split = AccountStats["vs_ai"];

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

const perHand = (x: number) => (x > 0 ? `+${x.toFixed(2)}` : x.toFixed(2));

/** Score/hand and win rate against the AI alone and with other people. */
function Splits({ vsAi, withPeople }: { vsAi: Split; withPeople: Split }) {
  const rows: [string, Split][] = [
    ["Vs the AI", vsAi],
    ["With people", withPeople],
  ];
  return (
    <table className={styles.splits}>
      <thead>
        <tr>
          <td />
          <th scope="col" className={ds.overline}>
            Hands
          </th>
          <th scope="col" className={ds.overline}>
            Score / hand
          </th>
          <td />
          <th scope="col" className={ds.overline}>
            Win rate
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, split]) => (
          <tr key={label}>
            <th scope="row">{label}</th>
            <td>{split.hands}</td>
            <td>{split.sph == null ? "—" : perHand(split.sph)}</td>
            <td className={styles.margin}>
              {split.sph_margin != null && `± ${split.sph_margin.toFixed(2)}`}
            </td>
            <td>{pct(split.win_pct)}</td>
          </tr>
        ))}
      </tbody>
    </table>
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
  const picking: [string, string][] = [
    ["Picks", String(stats.picks)],
    ["Trump / pick", stats.trump_per_pick?.toFixed(1) ?? "—"],
    ["Queens / pick", stats.queens_per_pick?.toFixed(1) ?? "—"],
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
        <span className={ds.overline}>Who you played with</span>
      </div>
      <Splits vsAi={stats.vs_ai} withPeople={stats.with_people} />
      <p className={styles.note}>
        Vs the AI: all other seats played by the AI. ± is a 95% range, after{" "}
        {stats.split_margin_min_hands} hands.
      </p>

      <div className={`${ds.headRule} ${styles.statsHead} ${styles.subHead}`}>
        <span className={ds.overline}>When you pick</span>
      </div>
      <Tiles tiles={picking} />
      <p className={styles.note}>Picks the AI made for you are left out.</p>

      <div className={`${ds.headRule} ${styles.statsHead} ${styles.subHead}`}>
        <span className={ds.overline}>Hand completion</span>
      </div>
      <Tiles tiles={completion} />
      <p className={styles.note}>
        * A hand counts as abandoned when the AI makes more than{" "}
        {stats.abandon_threshold} of your decisions in it (after a disconnect,
        leaving mid-hand, or turn timeouts). In an abandoned hand a loss counts
        in full, but a win counts as 0 and isn&rsquo;t a win. Moves the AI makes
        because the host closed the table, you were removed, or the server
        restarted aren&rsquo;t marked as abandoned.
      </p>
    </section>
  );
}
