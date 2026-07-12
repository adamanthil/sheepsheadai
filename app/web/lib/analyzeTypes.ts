// Generated from the server's OpenAPI schema — regenerate with
// `npm run gen:api` after changing server/api/schemas.py.
import type { components } from "./api.gen";

export type AnalyzeSimulateRequest =
  components["schemas"]["AnalyzeSimulateRequest"];
export type AnalyzeProbability = components["schemas"]["AnalyzeProbability"];
export type AnalyzePointEstimate =
  components["schemas"]["AnalyzePointEstimate"];
export type AnalyzeTrumpSeenMaskEntry =
  components["schemas"]["AnalyzeTrumpSeenMaskEntry"];
export type AnalyzeGameSummary = components["schemas"]["AnalyzeGameSummary"];

/** The per-step `view` is an untyped dict server-side (built straight from
 * the game engine); type the fields the analyze UI reads. */
export type AnalyzeView = {
  player?: number;
  picker?: number;
  partner?: number;
  is_leaster?: boolean;
  called_card?: string | null;
  current_trick?: string[];
  current_trick_index?: number;
  hand?: string[];
  blind?: string[];
  bury?: string[];
  [key: string]: unknown;
};

export type AnalyzeMeta = {
  partnerMode?: number;
  deterministic?: boolean;
  seed?: number | null;
  model?: string;
  gamma?: number;
  [key: string]: unknown;
};

export type AnalyzeActionDetail = Omit<
  components["schemas"]["AnalyzeActionDetail"],
  "view"
> & { view: AnalyzeView };

export type AnalyzeSimulateResponse = Omit<
  components["schemas"]["AnalyzeSimulateResponse"],
  "meta" | "trace"
> & { meta: AnalyzeMeta; trace: AnalyzeActionDetail[] };
