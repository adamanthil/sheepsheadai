/** Number formats shared across pages. */

const MINUS = "−";

/** A number with its sign: "+2", "−1.2", "0.0". A true minus sign, and
 * never a signed zero: a value that rounds to zero at ``digits`` decimal
 * places is shown bare. */
export function signed(value: number, digits = 0): string {
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return text;
  return (value > 0 ? "+" : MINUS) + text;
}
