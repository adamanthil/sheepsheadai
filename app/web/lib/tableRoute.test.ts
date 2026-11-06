import { describe, expect, it } from "vitest";
import { tableRoute } from "./tableRoute";

describe("tableRoute", () => {
  it("keeps an open table, or one not yet known, in the waiting room", () => {
    expect(tableRoute("t1", "open")).toBe("/waiting/t1");
    expect(tableRoute("t1", undefined)).toBe("/waiting/t1");
  });

  it("sends a dealt table to the table page, between hands too", () => {
    expect(tableRoute("t1", "playing")).toBe("/table/t1");
    expect(tableRoute("t1", "finished")).toBe("/table/t1");
  });
});
