/** Number and date formats shared across pages. */

const MINUS = "−";

/** A number with its sign: "+2", "−1.2", "0.0". A true minus sign, and
 * never a signed zero: a value that rounds to zero at ``digits`` decimal
 * places is shown bare. */
export function signed(value: number, digits = 0): string {
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return text;
  return (value > 0 ? "+" : MINUS) + text;
}

const DAY = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
});

const DAY_AND_TIME = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

/** "Nov 16", in the reader's own locale, from an ISO timestamp. */
export function shortDate(iso: string): string {
  return DAY.format(new Date(iso));
}

/** "Nov 16, 2:46 PM", in the reader's own locale and time zone. */
export function shortDateTime(iso: string): string {
  return DAY_AND_TIME.format(new Date(iso));
}
