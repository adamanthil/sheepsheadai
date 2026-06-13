import type { TableView } from "../../../../lib/types";

// Get player name for a seat from table data
export function nameForSeat(
  seat: number | null | undefined,
  table: TableView | null | undefined,
): string {
  if (!seat) return "";
  return table?.seats?.[String(seat)] || `Seat ${seat}`;
}

// Check if a seat is AI-controlled
export function isAiSeat(
  seat: number,
  table: TableView | null | undefined,
): boolean {
  return Boolean(table?.seatIsAI?.[String(seat)]);
}
