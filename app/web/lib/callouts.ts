import type { TableStateMsg } from "./types";

export interface TrickCompleteEvent {
  cards: string[];
  winner: number;
}

export type PhaseCallout =
  | { kind: "picker"; pickerName: string }
  | { kind: "leaster" }
  | { kind: "alone"; pickerName: string }
  | { kind: "call"; pickerName: string; cardDisplay: string; under: boolean };

export interface CalloutDiff {
  trickComplete: TrickCompleteEvent | null;
  phase: PhaseCallout | null;
}

/** Compare the previous and incoming table state and report which callouts
 * (trick-complete banner, picker/leaster/alone/call announcement) the
 * transition implies. Pure — callers decide what to do with the result. */
export function diffCallouts(
  prev: TableStateMsg | null,
  next: TableStateMsg,
): CalloutDiff {
  const prevWas = prev?.view?.was_trick_just_completed;
  const curWas = next.view?.was_trick_just_completed;
  let trickComplete: TrickCompleteEvent | null = null;
  if (curWas && !prevWas) {
    const last = next.view?.last_trick as string[] | null;
    const win = next.view?.last_trick_winner as number;
    if (last && win) {
      trickComplete = { cards: last, winner: win };
    }
  }

  const prevPicker = prev?.view?.picker || 0;
  const curPicker = next.view?.picker || 0;
  const prevLeaster = !!prev?.view?.is_leaster;
  const curLeaster = !!next.view?.is_leaster;
  const prevCalled = prev?.view?.called_card || null;
  const curCalled = next.view?.called_card || null;
  const curCalledDisplay = next.view?.called_card_display || curCalled;
  const curCalledUnder = !!next.view?.called_under;
  const prevAlone = !!prev?.view?.alone;
  const curAlone = !!next.view?.alone;

  const playStarted =
    !!(next.state?.play_started === 1) ||
    next.view?.current_trick_index > 0 ||
    next.view?.current_trick?.some((c: string) => c !== "");

  const getPickerName = () => {
    const seat = next.view?.picker;
    return next.table?.seats?.[String(seat)] || `Seat ${seat}`;
  };

  let phase: PhaseCallout | null = null;
  if (curPicker > 0 && prevPicker === 0) {
    phase = { kind: "picker", pickerName: getPickerName() };
  } else if (!prevLeaster && curLeaster) {
    phase = { kind: "leaster" };
  } else if (curPicker > 0 && !playStarted && !prevAlone && curAlone) {
    phase = { kind: "alone", pickerName: getPickerName() };
  } else if (
    curPicker > 0 &&
    !playStarted &&
    curCalled &&
    prevCalled !== curCalled
  ) {
    phase = {
      kind: "call",
      pickerName: getPickerName(),
      cardDisplay: curCalledDisplay ?? curCalled ?? "",
      under: curCalledUnder,
    };
  }

  return { trickComplete, phase };
}
