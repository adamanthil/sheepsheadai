import type { AnalyzeActionDetail } from "../../lib/analyzeTypes";

/** Sheepshead always has exactly 5 players. */
export const PLAYERS_PER_TRICK = 5;

export type TraceBoundary =
  | { beforeIndex: number; kind: "phase"; fromPhase: string; toPhase: string }
  | { beforeIndex: number; kind: "trick"; trickNumber: number };

/** Phase changes and trick starts within a trace, computed in one pass.
 * A boundary sits just before trace[beforeIndex]. Trick boundaries only
 * appear between consecutive play-phase steps — the phase change into
 * "play" already marks the first trick. Shared by the decision timeline
 * (dividers) and the memory update chart (dashed rules). */
export function traceBoundaries(
  trace: AnalyzeActionDetail[],
): TraceBoundary[] {
  const bounds: TraceBoundary[] = [];
  let playActionsBefore = 0;
  for (let i = 0; i < trace.length; i++) {
    const current = trace[i];
    const prev = i > 0 ? trace[i - 1] : null;
    if (prev && current.phase !== prev.phase) {
      bounds.push({
        beforeIndex: i,
        kind: "phase",
        fromPhase: prev.phase,
        toPhase: current.phase,
      });
    } else if (
      prev &&
      current.phase === "play" &&
      prev.phase === "play" &&
      playActionsBefore > 0 &&
      playActionsBefore % PLAYERS_PER_TRICK === 0
    ) {
      bounds.push({
        beforeIndex: i,
        kind: "trick",
        trickNumber: Math.floor(playActionsBefore / PLAYERS_PER_TRICK) + 1,
      });
    }
    if (current.phase === "play") playActionsBefore += 1;
  }
  return bounds;
}
