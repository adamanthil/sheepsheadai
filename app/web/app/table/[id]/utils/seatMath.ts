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

// Check if a seat is AI-controlled
export function isAiSeat(
  seat: number,
  table: TableView | null | undefined,
): boolean {
  return Boolean(table?.seatIsAI?.[String(seat)]);
}
