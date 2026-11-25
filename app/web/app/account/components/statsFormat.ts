/** Number formats shared by the account stats and hand history. */
import { signed } from "../../../lib/format";

export const pct = (x: number | null | undefined) =>
  x == null ? "—" : `${(x * 100).toFixed(1)}%`;

/** A count, with thousands separators. */
export const count = (n: number) => n.toLocaleString("en-US");

/** Score per hand, signed, to two places. */
export const perHand = (x: number | null | undefined) =>
  x == null ? "—" : signed(x, 2);
