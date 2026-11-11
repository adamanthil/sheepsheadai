import { describe, expect, it } from "vitest";
import { TYPING_STALE_MS, applyTyping, dropStale } from "./typing";

describe("applyTyping", () => {
  it("adds a typist and drops them when they stop", () => {
    const one = applyTyping([], "c1", "Kyle", true, 0);
    expect(one).toEqual([
      { clientId: "c1", name: "Kyle", until: TYPING_STALE_MS },
    ]);
    expect(applyTyping(one, "c1", "Kyle", false, 10)).toEqual([]);
  });

  it("keeps the order typists started in as repeats arrive", () => {
    let list = applyTyping([], "c1", "Kyle", true, 0);
    list = applyTyping(list, "c2", "Ann", true, 100);
    list = applyTyping(list, "c1", "Kyle", true, 3000);
    expect(list.map((t) => t.name)).toEqual(["Kyle", "Ann"]);
    expect(list[0].until).toBe(3000 + TYPING_STALE_MS);
  });

  it("leaves the list as is for a stop from someone not typing", () => {
    const list = applyTyping([], "c1", "Kyle", true, 0);
    expect(applyTyping(list, "c2", "Ann", false, 10)).toBe(list);
  });
});

describe("dropStale", () => {
  it("drops typists not heard from in time", () => {
    let list = applyTyping([], "c1", "Kyle", true, 0);
    list = applyTyping(list, "c2", "Ann", true, 2000);
    expect(dropStale(list, TYPING_STALE_MS).map((t) => t.name)).toEqual([
      "Ann",
    ]);
    expect(dropStale(list, 10)).toBe(list);
  });
});
