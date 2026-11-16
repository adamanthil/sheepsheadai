"use client";

import React, { useId, useMemo, useState } from "react";
import type { AccountForm, AccountStats } from "../../../lib/types";
import { count, pct, perHand, signed } from "./statsFormat";
import styles from "./StatsCharts.module.css";

/* Hand-drawn SVG and positioned HTML; no chart library. Win green and
   loss red are validated as a pair on the page surface (CVD ΔE 7.8,
   deutan), which is legal because every chart here also separates them
   by position: above or below zero. */

type Role = AccountForm["roles"][number];

const ROLE_LABEL: Record<Role, string> = {
  picker: "Picker",
  partner: "Partner",
  defender: "Defender",
  leaster: "Leaster",
};
const ROLE_WORD: Record<Role, string> = {
  picker: "as picker",
  partner: "as partner",
  defender: "on defense",
  leaster: "in a leaster",
};
const DAY = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
});

/** Round tick values covering [lo, hi], about `count` of them. */
function niceTicks(lo: number, hi: number, count = 4): number[] {
  if (lo === hi) hi = lo + 1;
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const ticks: number[] = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + step * 1e-9; v += step)
    ticks.push(Math.round(v * 1e6) / 1e6);
  if (ticks[ticks.length - 1] < hi) ticks.push(ticks[ticks.length - 1] + step);
  return ticks;
}

/* ---------------------------------------------------------------------- */
/* Form: the running score over a per-hand result strip                    */
/* ---------------------------------------------------------------------- */

/** Past this many hands the line keeps each bucket's low and high rather
 * than every point, about one pair per pixel of a wide chart. */
const LINE_BUCKETS = 600;
/** Past this many hands the strip shows each bucket's net, not each hand. */
const STRIP_BARS = 250;

export type FormWindow = {
  /** The first hand's number in the player's whole history (1-based). */
  first: number;
  scores: number[];
  roles: Role[];
  multipliers: number[];
  closed: number[];
};

export function formWindow(form: AccountForm, last: number): FormWindow {
  const start = last ? Math.max(0, form.scores.length - last) : 0;
  return {
    first: start + 1,
    scores: form.scores.slice(start),
    roles: form.roles.slice(start),
    multipliers: form.multipliers.slice(start),
    closed: form.closed.slice(start),
  };
}

/** The line's path over x in [0, 100], thinned to each bucket's low and
 * high (in the order they occur) when there are many hands. */
function linePath(run: number[], y: (v: number) => number): string {
  const n = run.length - 1;
  const x = (i: number) => (i / n) * 100;
  const pts: number[] = [];
  if (n <= LINE_BUCKETS * 2) {
    for (let i = 0; i <= n; i++) pts.push(i);
  } else {
    const size = n / LINE_BUCKETS;
    pts.push(0);
    for (let b = 0; b < LINE_BUCKETS; b++) {
      const from = Math.floor(b * size) + 1;
      const to = Math.floor((b + 1) * size);
      let lo = from;
      let hi = from;
      for (let i = from; i <= to; i++) {
        if (run[i] < run[lo]) lo = i;
        if (run[i] > run[hi]) hi = i;
      }
      pts.push(Math.min(lo, hi), Math.max(lo, hi));
    }
    pts.push(n);
  }
  return pts.map((i, k) => `${k ? "L" : "M"}${x(i)} ${y(run[i])}`).join("");
}

/** The strip's columns: one per hand, or each bucket's net past
 * STRIP_BARS hands. Each is [first hand index, hands in it, net]. */
function stripBars(scores: number[]): [number, number, number][] {
  const n = scores.length;
  const size = n > STRIP_BARS ? n / STRIP_BARS : 1;
  const count = Math.min(n, STRIP_BARS);
  const bars: [number, number, number][] = [];
  for (let b = 0; b < count; b++) {
    const from = Math.floor(b * size);
    const to = Math.floor((b + 1) * size);
    let net = 0;
    for (let i = from; i < to; i++) net += scores[i];
    bars.push([from, to - from, net]);
  }
  return bars;
}

/**
 * The running score over a window of hands, counted from 0 at its start
 * so the slope and end are the window's own, with each hand's result (or
 * each bucket's net, for long windows) as a column underneath. One
 * crosshair spans both; the arrow keys step it hand by hand.
 */
export function FormChart({
  data,
  height = 300,
}: {
  data: FormWindow;
  height?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const n = data.scores.length;

  const chart = useMemo(() => {
    const run: number[] = [0];
    for (const s of data.scores) run.push(run[run.length - 1] + s);
    const ticks = niceTicks(Math.min(0, ...run), Math.max(0, ...run));
    const yMin = ticks[0];
    const yMax = ticks[ticks.length - 1];
    const y = (v: number) => ((yMax - v) / (yMax - yMin)) * 100;
    const line = linePath(run, y);
    const bars = stripBars(data.scores);
    const maxAbs = Math.max(1, ...bars.map(([, , net]) => Math.abs(net)));
    // Memoized with the data so moving the crosshair redraws only it.
    const strip = (
      <svg viewBox={`0 0 ${n} 2`} preserveAspectRatio="none">
        {bars.map(([from, size, net]) => {
          if (net === 0) return null;
          const v = Math.abs(net) / maxAbs;
          return (
            <rect
              key={from}
              x={from + size * 0.15}
              width={size * 0.7}
              y={net > 0 ? 1 - v : 1}
              height={v}
              className={net > 0 ? styles.barWin : styles.barLoss}
            />
          );
        })}
      </svg>
    );
    const plot = (
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        className={styles.formSvg}
        aria-hidden="true"
      >
        <defs>
          <clipPath id="form-above">
            <rect x="0" y="0" width="100" height={y(0)} />
          </clipPath>
          <clipPath id="form-below">
            <rect x="0" y={y(0)} width="100" height={100 - y(0)} />
          </clipPath>
        </defs>
        {ticks.map((t) => (
          <line
            key={t}
            x1="0"
            x2="100"
            y1={y(t)}
            y2={y(t)}
            className={t === 0 ? styles.zero : styles.grid}
            vectorEffect="non-scaling-stroke"
          />
        ))}
        <path
          d={`${line}L100 ${y(0)}L0 ${y(0)}Z`}
          clipPath="url(#form-above)"
          className={styles.washWin}
        />
        <path
          d={`${line}L100 ${y(0)}L0 ${y(0)}Z`}
          clipPath="url(#form-below)"
          className={styles.washLoss}
        />
        <path
          d={line}
          className={styles.line}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    );
    return { run, ticks, y, strip, plot, bucketed: n > STRIP_BARS };
  }, [data, n]);

  const { run, ticks, y } = chart;
  const x = (i: number) => (i / n) * 100;
  const at = hover ?? n;
  const net = run[n];
  const day = (i: number) => DAY.format(new Date(data.closed[i] * 1000));

  const move = (e: React.PointerEvent<HTMLDivElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const f = (e.clientX - box.left) / box.width;
    setHover(Math.min(n, Math.max(1, Math.round(f * n))));
  };
  const key = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 1;
    if (e.key === "ArrowLeft") setHover(Math.max(1, (hover ?? n) - step));
    else if (e.key === "ArrowRight") setHover(Math.min(n, (hover ?? n) + step));
    else return;
    e.preventDefault();
  };

  return (
    <figure
      className={styles.form}
      style={{ "--plot-h": `${height}px` } as React.CSSProperties}
    >
      <div className={styles.formTicks} aria-hidden="true">
        {ticks.map((t) => (
          <span key={t} style={{ top: `${y(t)}%` }}>
            {signed(t)}
          </span>
        ))}
      </div>
      <div className={styles.formPlot}>
        {chart.plot}
        {hover != null && (
          <div className={styles.crosshair} style={{ left: `${x(at)}%` }} />
        )}
        <div
          className={styles.dot}
          style={{ left: `${x(at)}%`, top: `${y(run[at])}%` }}
        />
        {hover != null && (
          <div
            className={styles.tip}
            data-side={x(at) > 62 ? "left" : "right"}
            style={{ left: `${x(at)}%`, top: `${Math.min(y(run[at]), 70)}%` }}
          >
            <strong>{signed(run[at])}</strong>
            <span>
              Hand {count(data.first + at - 1)} · {day(at - 1)}
            </span>
            <span>
              {signed(data.scores[at - 1])} {ROLE_WORD[data.roles[at - 1]]}
              {data.multipliers[at - 1] > 1
                ? ` · ${data.multipliers[at - 1]}×`
                : ""}
            </span>
          </div>
        )}
        <div
          className={styles.hit}
          tabIndex={0}
          role="img"
          aria-label={`Running score over ${count(n)} hands, ${signed(net)} net. Use the arrow keys to step through hands.`}
          onPointerMove={move}
          onPointerLeave={() => setHover(null)}
          onBlur={() => setHover(null)}
          onKeyDown={key}
        />
      </div>
      <div className={styles.formEnd} aria-hidden="true">
        <span style={{ top: `${y(net)}%` }}>{signed(net)}</span>
      </div>
      <div className={styles.strip} aria-hidden="true">
        {chart.strip}
        {hover != null && (
          <div
            className={styles.stripMark}
            style={{ left: `${((at - 0.5) / n) * 100}%` }}
          />
        )}
      </div>
      <figcaption className={styles.formAxis}>
        <span>
          Hand {count(data.first)} · {day(0)}
        </span>
        <span className={styles.formAxisNote}>
          {chart.bucketed
            ? "Columns: net per group of hands"
            : "Columns: each hand"}
        </span>
        <span>
          Hand {count(data.first + n - 1)} · {day(n - 1)}
        </span>
      </figcaption>
    </figure>
  );
}

/** Last-N presets for the form chart; 0 is every hand. */
export const RANGES = [50, 100, 250, 1000, 0] as const;
export type Range = (typeof RANGES)[number];

/** A row of buttons, one pressed: a small filter. */
export function Segmented<T extends string | number>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className={styles.seg} role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={value === o.value}
          className={value === o.value ? styles.segOn : undefined}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function RangePicker({
  value,
  onChange,
  total,
}: {
  value: Range;
  onChange: (r: Range) => void;
  total: number;
}) {
  return (
    <Segmented
      label="Hands shown"
      options={RANGES.filter((r) => r === 0 || r < total).map((r) => ({
        value: r,
        label: r === 0 ? "All" : `Last ${r.toLocaleString("en-US")}`,
      }))}
      value={value}
      onChange={onChange}
    />
  );
}

/* ---------------------------------------------------------------------- */
/* Rates                                                                   */
/* ---------------------------------------------------------------------- */

/** One rate as a labelled meter: name and figure, the bar, a note. */
export function Meter({
  label,
  value,
  note,
  children,
}: {
  label: string;
  value: number | null;
  note: string;
  /** Shown under the note: a breakdown of this rate. */
  children?: React.ReactNode;
}) {
  return (
    <div className={styles.meterItem}>
      <div className={styles.meterHead}>
        <span>{label}</span>
        <strong>{pct(value)}</strong>
      </div>
      <div className={styles.meter}>
        {value != null && (
          <div
            className={styles.meterFill}
            style={{ width: `${value * 100}%` }}
          />
        )}
      </div>
      <span className={styles.meterNote}>{note}</span>
      {children}
    </div>
  );
}

const ORDINAL = ["1st", "2nd", "3rd", "4th", "5th"];

/** Pick when offered by bidding position (named as on the table: 1st
 * position bids first), behind a toggle: one small meter per position,
 * with the picks and offers it comes from. */
export function PickByPosition({
  positions,
}: {
  positions: AccountStats["pick_offers_by_position"];
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className={styles.expand}>
      <button
        type="button"
        className={styles.expandToggle}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((o) => !o)}
      >
        By position
        <svg
          className={styles.expandCaret}
          viewBox="0 0 10 6"
          aria-hidden="true"
        >
          <path d="M1 1l4 4 4-4" />
        </svg>
      </button>
      {open && (
        <ul id={id} className={styles.positions}>
          {positions.map((p) => {
            const rate = p.offered_hands ? p.picks / p.offered_hands : null;
            return (
              <li key={p.position}>
                <div className={styles.positionHead}>
                  <span>{ORDINAL[p.position - 1]} position</span>
                  <span>
                    <strong>{pct(rate)}</strong>
                    <span className={styles.positionCount}>
                      {p.offered_hands
                        ? `${count(p.picks)} of ${count(p.offered_hands)}`
                        : "never offered"}
                    </span>
                  </span>
                </div>
                <div className={`${styles.meter} ${styles.meterSmall}`}>
                  {rate != null && (
                    <div
                      className={styles.meterFill}
                      style={{ width: `${rate * 100}%` }}
                    />
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------------- */
/* Score per hand with its 95% range, on one shared axis                   */
/* ---------------------------------------------------------------------- */

type Split = AccountStats["vs_ai"];

export function RangeRows({
  rows,
}: {
  rows: { label: string; split: Split }[];
}) {
  const reach = Math.max(
    0.5,
    ...rows.map(
      ({ split }) => Math.abs(split.sph ?? 0) + (split.sph_margin ?? 0),
    ),
  );
  const ticks = niceTicks(-reach, reach, 4);
  const lo = ticks[0];
  const hi = ticks[ticks.length - 1];
  const x = (v: number) => ((v - lo) / (hi - lo)) * 100;
  return (
    <div className={styles.ranges}>
      {rows.map(({ label, split }) => (
        <div key={label} className={styles.rangeRow}>
          <div className={styles.rangeName}>
            {label}
            <span>
              {count(split.hands)} hand{split.hands === 1 ? "" : "s"}
            </span>
          </div>
          <div className={styles.rangeTrack}>
            <div className={styles.rangeZero} style={{ left: `${x(0)}%` }} />
            {split.sph != null && split.sph_margin != null && (
              <div
                className={styles.rangeSpan}
                style={{
                  left: `${x(split.sph - split.sph_margin)}%`,
                  width: `${x(split.sph + split.sph_margin) - x(split.sph - split.sph_margin)}%`,
                }}
              />
            )}
            {split.sph != null && (
              <div
                className={`${styles.rangeDot} ${split.sph < 0 ? styles.rangeDotLoss : ""}`}
                style={{ left: `${x(split.sph)}%` }}
              />
            )}
          </div>
          <div className={styles.rangeValue}>
            {perHand(split.sph)}
            {split.sph_margin != null && (
              <span>± {split.sph_margin.toFixed(2)}</span>
            )}
          </div>
        </div>
      ))}
      <div className={styles.rangeAxis} aria-hidden="true">
        <div />
        <div className={styles.rangeTicks}>
          {ticks.map((t) => (
            <span key={t} style={{ left: `${x(t)}%` }}>
              {t > 0 ? `+${t}` : t}
            </span>
          ))}
        </div>
        <div />
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------- */
/* By role: hands, win rate, and score per hand as a bar from zero         */
/* ---------------------------------------------------------------------- */

export function RoleBars({
  roles,
  hands,
}: {
  roles: AccountStats["roles"];
  hands: number;
}) {
  const reach = Math.max(0.5, ...roles.map((r) => Math.abs(r.sph ?? 0)));
  return (
    <table className={styles.roles}>
      <thead>
        <tr>
          <td />
          <th scope="col">Hands</th>
          <th scope="col">Win rate</th>
          <th scope="col" className={styles.rolesBarHead}>
            Score / hand
          </th>
        </tr>
      </thead>
      <tbody>
        {roles.map((r) => {
          const w = r.sph == null ? 0 : (Math.abs(r.sph) / reach) * 50;
          return (
            <tr key={r.role}>
              <th scope="row">{ROLE_LABEL[r.role]}</th>
              <td>
                {count(r.hands)}
                <span className={styles.rolesShare}>
                  {hands ? `${Math.round((r.hands / hands) * 100)}%` : ""}
                </span>
              </td>
              <td>{pct(r.win_pct)}</td>
              <td className={styles.rolesBarCell}>
                <div className={styles.rolesBar}>
                  <div className={styles.rolesZero} />
                  {r.sph != null && (
                    <div
                      className={
                        r.sph >= 0 ? styles.rolesWin : styles.rolesLoss
                      }
                      style={
                        r.sph >= 0
                          ? { left: "50%", width: `${w}%` }
                          : { right: "50%", width: `${w}%` }
                      }
                    />
                  )}
                </div>
                <span className={styles.rolesValue}>{perHand(r.sph)}</span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
