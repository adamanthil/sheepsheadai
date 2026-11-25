import { describe, expect, it } from "vitest";
import { parsePickScenario, pickScenarioHref } from "./pickScenario";

const scenario = {
  seat: 3,
  partnerMode: 1,
  hand: ["QS", "JC", "JS", "10D", "KC", "AH"],
  blind: ["10C", "8S"],
};

describe("pick scenario links", () => {
  it("round-trips a deal through a link", () => {
    const href = pickScenarioHref(scenario);
    expect(href.startsWith("/analyze/pick?")).toBe(true);
    expect(parsePickScenario(href.slice(href.indexOf("?")))).toEqual(scenario);
  });

  it("reads nothing from a plain visit", () => {
    expect(parsePickScenario("")).toBeNull();
    expect(parsePickScenario("?seat=2")).toBeNull();
  });

  it("refuses a deal that could not be dealt", () => {
    expect(parsePickScenario("?hand=QS,ZZ")).toBeNull();
    expect(parsePickScenario("?hand=QS,QS")).toBeNull();
    expect(parsePickScenario("?hand=QS&blind=QS")).toBeNull();
    expect(parsePickScenario("?hand=QS&seat=6")).toBeNull();
    expect(parsePickScenario("?hand=QS&mode=2")).toBeNull();
    expect(parsePickScenario("?hand=QS,JC,JS,10D,KC,AH,7C")).toBeNull();
    expect(parsePickScenario("?hand=QS&blind=7C,8C,9C")).toBeNull();
  });

  it("takes a partial deal, as the analyzer does", () => {
    expect(parsePickScenario("?hand=QS,JC")).toEqual({
      seat: 1,
      partnerMode: 1,
      hand: ["QS", "JC"],
      blind: [],
    });
  });
});
