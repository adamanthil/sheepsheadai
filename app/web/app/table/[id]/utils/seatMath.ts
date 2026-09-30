import type { TableView } from "../../../../lib/types";

// Get player name for a seat from table data
export function nameForSeat(
  seat: number | null | undefined,
  table: TableView | null | undefined,
): string {
  if (!seat) return "";
  return table?.seats?.[String(seat)] || `Seat ${seat}`;
}

// The seated human's verified account username, if any
export function accountForSeat(
  seat: number,
  table: TableView | null | undefined,
): string | null {
  return table?.seatAccount?.[String(seat)] ?? null;
}

// Whether a spectator may take over a seat now: the AI holds it and, mid-
// hand, was also dealt it (a human's hand stays theirs to reclaim)
export function isTakeableSeat(
  seat: number,
  table: TableView | null | undefined,
): boolean {
  return Boolean(table?.seatTakeable?.[String(seat)]);
}

// Check if a seat is AI-controlled
export function isAiSeat(
  seat: number,
  table: TableView | null | undefined,
): boolean {
  return Boolean(table?.seatIsAI?.[String(seat)]);
}
