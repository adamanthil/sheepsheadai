/** House-rule display names, shared by every surface that reports them.
 *
 * The lobby, the waiting room and the in-game header all name the same
 * modes, so the strings live here rather than being spelled out per view.
 * Rules arrive as a loose object over the wire; anything unrecognised falls
 * back to the mode a table gets by default.
 */

type Rules = Record<string, unknown> | undefined;

export function partnerModeLabel(rules: Rules): string {
  return rules?.partnerMode === 0 ? "Jack of Diamonds" : "Called Ace";
}

export function allPassModeLabel(rules: Rules): string {
  return rules?.allPassMode === "doublers" ? "Doublers" : "Leasters";
}

export function scoringModeLabel(rules: Rules): string {
  return rules?.doubleOnTheBump ? "Double on Bump" : "Symmetric";
}

/** Partner and all-pass modes: what kind of game this is, in the fewest
 * words that still distinguish one table from another. */
export function gameModeLabel(rules: Rules): string {
  return `${partnerModeLabel(rules)} · ${allPassModeLabel(rules)}`;
}
