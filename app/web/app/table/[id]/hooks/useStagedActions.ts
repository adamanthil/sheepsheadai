import { useEffect, useState } from "react";
import { interludeMode } from "../lib/phase";

/** Staging state and handlers consumed by the center-stage bury/under UI. */
export interface StagedActions {
  /** Cards the server has already accepted as buried (masked to the picker).
   * Normally empty until confirm; non-empty mid-confirm or after a reconnect. */
  serverBury: string[];
  burySelection: string[];
  underSelection: string | null;
  /** True while a confirm is POSTing; disables the confirm buttons. */
  busy: boolean;
  onDeselectBury: (card: string) => void;
  onDeselectUnder: () => void;
  onConfirmBury: () => void;
  onConfirmUnder: () => void;
}

export interface UseStagedActionsArgs {
  serverBury: string[];
  validActionStrings: Set<string>;
  actionIdByString: Record<string, number>;
  takeAction: (actionId: number) => Promise<boolean>;
}

export interface UseStagedActionsReturn {
  staging: StagedActions;
  /** Cards staged in the center and therefore hidden from the hand fan. */
  stagedCards: string[];
  /** Stage a tapped card. Returns true when the tap was consumed. */
  stageCard: (card: string) => boolean;
}

/**
 * Local staging for the picker's bury and under-card choices: card taps only
 * stage a selection (shown face-up in the center, tappable to put back);
 * the BURY/UNDER actions are sent when the player confirms.
 */
export function useStagedActions({
  serverBury,
  validActionStrings,
  actionIdByString,
  takeAction,
}: UseStagedActionsArgs): UseStagedActionsReturn {
  const [burySelection, setBurySelection] = useState<string[]>([]);
  const [underSelection, setUnderSelection] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Drop staged cards that stopped being valid choices: the state advanced,
  // a reconnect resynced us, or the card was committed server-side.
  useEffect(() => {
    setBurySelection((sel) => {
      const kept = sel.filter((c) => validActionStrings.has(`BURY ${c}`));
      return kept.length === sel.length ? sel : kept;
    });
    setUnderSelection((sel) =>
      sel && !validActionStrings.has(`UNDER ${sel}`) ? null : sel,
    );
  }, [validActionStrings]);

  const stageCard = (card: string): boolean => {
    if (validActionStrings.has(`BURY ${card}`)) {
      setBurySelection((sel) =>
        sel.includes(card) || sel.length + serverBury.length >= 2
          ? sel
          : [...sel, card],
      );
      return true;
    }
    if (validActionStrings.has(`UNDER ${card}`)) {
      // Tapping another card just moves the selection.
      setUnderSelection(card);
      return true;
    }
    return false;
  };

  async function confirmBury() {
    if (busy) return;
    const ids = burySelection.map((c) => actionIdByString[`BURY ${c}`]);
    if (ids.length === 0 || ids.some((id) => id === undefined)) return;
    setBusy(true);
    try {
      for (const id of ids) {
        if (!(await takeAction(id as number))) break;
      }
    } finally {
      setBusy(false);
    }
  }

  async function confirmUnder() {
    if (busy || !underSelection) return;
    const id = actionIdByString[`UNDER ${underSelection}`];
    if (id === undefined) return;
    setBusy(true);
    try {
      await takeAction(id);
    } finally {
      setBusy(false);
    }
  }

  const mode = interludeMode(validActionStrings);
  const stagedCards =
    mode === "bury"
      ? burySelection
      : mode === "under" && underSelection
        ? [underSelection]
        : [];

  return {
    staging: {
      serverBury,
      burySelection,
      underSelection,
      busy,
      onDeselectBury: (card) =>
        setBurySelection((sel) => sel.filter((c) => c !== card)),
      onDeselectUnder: () => setUnderSelection(null),
      onConfirmBury: () => void confirmBury(),
      onConfirmUnder: () => void confirmUnder(),
    },
    stagedCards,
    stageCard,
  };
}
