import type { HandHistoryRow } from "./types";

/** A player's part in a hand. */
export type Role = HandHistoryRow["role"];

export const ROLE_LABEL: Record<Role, string> = {
  picker: "Picker",
  partner: "Partner",
  defender: "Defender",
  leaster: "Leaster",
};
