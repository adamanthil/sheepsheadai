import { describe, expect, it } from "vitest";
import { signed } from "./format";

describe("signed", () => {
  it("uses a real minus sign and never shows a signed zero", () => {
    expect(signed(2)).toBe("+2");
    expect(signed(-3)).toBe("−3");
    expect(signed(0)).toBe("0");
    expect(signed(1.24, 1)).toBe("+1.2");
    expect(signed(-0.7, 1)).toBe("−0.7");
    expect(signed(0, 1)).toBe("0.0");
    expect(signed(-0.04, 1)).toBe("0.0");
    expect(signed(-0.126, 2)).toBe("−0.13");
  });
});
