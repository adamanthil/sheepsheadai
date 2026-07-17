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
export type AnalyzeMemoryObserve =
  components["schemas"]["AnalyzeMemoryObserve"];
export type AnalyzeCalibrationSummary =
  components["schemas"]["AnalyzeCalibrationSummary"];
export type AnalyzeSeatCalibration =
  components["schemas"]["AnalyzeSeatCalibration"];
export type AnalyzeModelResponse =
  components["schemas"]["AnalyzeModelResponse"];
export type AnalyzeCardEmbeddings =
  components["schemas"]["AnalyzeCardEmbeddings"];
export type AnalyzeCardEmbeddingEntry =
  components["schemas"]["AnalyzeCardEmbeddingEntry"];

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

export type AnalyzeActionDetail = Omit<
  components["schemas"]["AnalyzeActionDetail"],
  "view"
> & { view: AnalyzeView };

export type AnalyzeSimulateResponse = Omit<
  components["schemas"]["AnalyzeSimulateResponse"],
  "trace"
> & { trace: AnalyzeActionDetail[] };
