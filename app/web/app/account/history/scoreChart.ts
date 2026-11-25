/** Geometry the two charts of expected score share: the strip under a
 * history row and the full chart on a hand's page. Each gives a decision
 * a column a hundred SVG units wide and plots zero on the centre line. */

/** Points of score the axis reaches either side of zero, however flat
 * the hand was. */
const LEAST_REACH = 4;

/** The SVG x of the middle of column ``i``. */
export function columnX(i: number): number {
  return 50 + 100 * i;
}

/** How far the axis runs either side of zero to hold every value: a whole
 * number of points, and never fewer than LEAST_REACH. */
export function axisReach(values: number[]): number {
  return Math.ceil(Math.max(LEAST_REACH, ...values.map(Math.abs)));
}
