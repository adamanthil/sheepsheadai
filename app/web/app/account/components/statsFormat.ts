/** Number formats shared by the account stats and hand history. */

export const pct = (x: number | null | undefined) =>
  x == null ? "—" : `${(x * 100).toFixed(1)}%`;

export const signed = (x: number) => (x > 0 ? `+${x}` : String(x));

/** A count, with thousands separators. */
export const count = (n: number) => n.toLocaleString("en-US");

/** Score per hand, signed, to two places. */
export const perHand = (x: number | null | undefined) =>
  x == null ? "—" : x > 0 ? `+${x.toFixed(2)}` : x.toFixed(2);
