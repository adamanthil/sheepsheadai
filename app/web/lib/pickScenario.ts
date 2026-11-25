/** A pick scenario carried in a link: the seat, its six cards, the blind
 * and the partner mode. The hand review writes these links and the pick
 * analyzer (/analyze/pick) reads them, opening on exactly that deal. */

import { isCardCode } from "./ds/cardUtils";

export interface PickScenario {
  /** 1..5; the seats before it have passed. */
  seat: number;
  /** 0 = jack of diamonds, 1 = called ace. */
  partnerMode: number;
  hand: string[];
  blind: string[];
}

export function pickScenarioHref(scenario: PickScenario): string {
  const query = new URLSearchParams({
    seat: String(scenario.seat),
    mode: String(scenario.partnerMode),
    hand: scenario.hand.join(","),
    blind: scenario.blind.join(","),
  });
  return `/analyze/pick?${query}`;
}

function cards(value: string | null, limit: number): string[] | null {
  if (!value) return [];
  const list = value.split(",");
  return list.length <= limit && list.every(isCardCode) ? list : null;
}

/** The scenario a link's query string names, or null if it names none or
 * names one that could not be dealt. */
export function parsePickScenario(search: string): PickScenario | null {
  const query = new URLSearchParams(search);
  if (!query.has("hand") && !query.has("blind")) return null;
  const seat = Number(query.get("seat") ?? "1");
  const partnerMode = Number(query.get("mode") ?? "1");
  const hand = cards(query.get("hand"), 6);
  const blind = cards(query.get("blind"), 2);
  if (!hand || !blind) return null;
  if (![1, 2, 3, 4, 5].includes(seat) || ![0, 1].includes(partnerMode))
    return null;
  if (new Set([...hand, ...blind]).size !== hand.length + blind.length)
    return null;
  return { seat, partnerMode, hand, blind };
}
