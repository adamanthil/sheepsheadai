import { describe, expect, it } from "vitest";

import { AccountError } from "./account";
import { HttpStatusError } from "./apiError";
import { isExpectedFailure } from "./reportCaught";

describe("isExpectedFailure", () => {
  it("drops network failures, aborts and API errors", () => {
    expect(isExpectedFailure(new TypeError("Failed to fetch"))).toBe(true);
    expect(isExpectedFailure(new TypeError("Load failed"))).toBe(true);
    expect(
      isExpectedFailure(
        new TypeError("NetworkError when attempting to fetch resource."),
      ),
    ).toBe(true);
    expect(isExpectedFailure(new DOMException("aborted", "AbortError"))).toBe(
      true,
    );
    expect(isExpectedFailure(new AccountError("network", 0))).toBe(true);
    expect(isExpectedFailure(new HttpStatusError(502))).toBe(true);
  });

  it("keeps bugs", () => {
    expect(
      isExpectedFailure(
        new TypeError("Cannot read properties of undefined (reading 'map')"),
      ),
    ).toBe(false);
    expect(isExpectedFailure(new SyntaxError("Unexpected token <"))).toBe(
      false,
    );
    expect(isExpectedFailure(new Error("boom"))).toBe(false);
    expect(isExpectedFailure("a string")).toBe(false);
  });
});
