import { describe, expect, it } from "vitest";
import { handReviewHref, historyHref, reviewHref } from "./reviewLinks";

describe("review links", () => {
  it("gives a page's first view the bare address", () => {
    expect(historyHref("results")).toBe("/account/history");
    expect(historyHref("ai")).toBe("/account/history?view=ai");
    expect(reviewHref("decisions")).toBe("/account/review");
    expect(reviewHref("picking")).toBe("/account/review?view=picking");
  });

  it("opens a hand on a decision when one is named", () => {
    expect(handReviewHref("g1")).toBe("/account/history/g1");
    expect(handReviewHref("g1", 0)).toBe("/account/history/g1?d=0");
  });
});
