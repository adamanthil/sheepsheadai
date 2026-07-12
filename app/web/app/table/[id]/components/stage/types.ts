import React from "react";
import type { TablePhase, SeatRole, InterludeMode } from "../../lib/phase";
import type { AnimTrick } from "../../hooks/useTrickAnimation";

export interface SeatView {
  absSeat: number;
  rel: number;
  name: string;
  isAI: boolean;
  role: SeatRole;
  you: boolean;
}

export interface CallOption {
  actionId: number;
  label: string; // raw action label, e.g. "CALL AC" / "ALONE"
  code: string | null; // card code for ace options, null for ALONE/JD
  display: string; // human label
}

export interface StageProps {
  seats: SeatView[];
  yourSeat: number;
  phase: TablePhase;
  isLeaster: boolean;
  yourMode: InterludeMode;
  isYourTurn: boolean;
  handLen: number;
  trickIndex: number;
  totalTricks: number;
  displayCards: string[]; // current_trick, or last_trick when showing prev
  winnerSeat: number | null;
  showPrev: boolean;
  prevText: string | null;
  callOptions: CallOption[];
  selectedCall: string | null;
  onAction: (actionId: number) => void;
  // Pick-phase decision: action ids for the centered Pick (the blind) / Pass
  // buttons, or null when it isn't your decision (the blind then shows as
  // context only, not as a button).
  pickActionId?: number | null;
  passActionId?: number | null;
  isMobile: boolean;
  // Desktop card/size multiplier (1 = base, >1 on large desktops). Ignored on
  // mobile, which has its own fixed compact sizing.
  uiScale?: number;
  trickBoxRef: React.RefObject<HTMLDivElement | null>;
  animTrick: AnimTrick | null;
  callout: string | null;
}
