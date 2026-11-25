import { describe, expect, it, vi } from "vitest";
import { latestReviews } from "./handHistory";
import type { HandHistory, HandHistoryRow } from "./types";

function hand(id: string, status: "queued" | "reviewed"): HandHistoryRow {
  return {
    game_id: id,
    review: { status, cost: null, gained: null, net: null, notable: false },
  } as HandHistoryRow;
}

/** A history of two-hand pages; the cursor is the next page's number. */
function history(...pages: HandHistoryRow[][]) {
  return vi.fn(async (before: string | null): Promise<HandHistory> => {
    const at = before === null ? 0 : Number(before);
    return {
      hands: pages[at],
      next_cursor: at + 1 < pages.length ? String(at + 1) : null,
    };
  });
}

describe("latestReviews", () => {
  it("reads only as far down as the hands asked for", async () => {
    const fetchPage = history(
      [hand("a", "reviewed"), hand("b", "reviewed")],
      [hand("c", "queued"), hand("d", "reviewed")],
      [hand("e", "queued"), hand("f", "queued")],
    );

    const reviews = await latestReviews(["a"], fetchPage);

    expect(reviews.get("a")?.status).toBe("reviewed");
    expect(fetchPage).toHaveBeenCalledTimes(1);
  });

  it("follows the history down to a hand on a later page", async () => {
    const fetchPage = history(
      [hand("a", "reviewed"), hand("b", "reviewed")],
      [hand("c", "queued"), hand("d", "reviewed")],
      [hand("e", "queued"), hand("f", "queued")],
    );

    const reviews = await latestReviews(["d", "b"], fetchPage);

    expect([...reviews.keys()].sort()).toEqual(["b", "d"]);
    expect(fetchPage).toHaveBeenCalledTimes(2);
  });

  it("stops at the end of the history for a hand that is not there", async () => {
    const fetchPage = history([hand("a", "reviewed")]);

    const reviews = await latestReviews(["gone"], fetchPage);

    expect(reviews.size).toBe(0);
    expect(fetchPage).toHaveBeenCalledTimes(1);
  });
});
