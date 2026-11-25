/** Addresses of the hand history and AI review pages. Each page with two
 * views names the one to open in ?view=; its first view is the default
 * and has the bare address. */

export const HISTORY_VIEWS = ["results", "ai"] as const;
export type HistoryView = (typeof HISTORY_VIEWS)[number];

export const REVIEW_VIEWS = ["decisions", "picking"] as const;
export type ReviewView = (typeof REVIEW_VIEWS)[number];

function viewHref<T extends string>(
  path: string,
  views: readonly T[],
  view: T,
): string {
  return view === views[0] ? path : `${path}?view=${view}`;
}

/** The hand history page opened on ``view``. */
export function historyHref(view: HistoryView): string {
  return viewHref("/account/history", HISTORY_VIEWS, view);
}

/** The AI review tab opened on ``view``. */
export function reviewHref(view: ReviewView): string {
  return viewHref("/account/review", REVIEW_VIEWS, view);
}

/** A hand's review, opened on one of its decisions if ``decision`` (the
 * decision's index) is given and on its headline otherwise. */
export function handReviewHref(gameId: string, decision?: number): string {
  const path = `/account/history/${gameId}`;
  return decision === undefined ? path : `${path}?d=${decision}`;
}
