import type React from 'react';

// Convert absolute seat (1-5) to relative seat (0-4) from viewer's perspective
export function relSeat(absSeat: number, mySeat: number): number {
  return (absSeat - mySeat + 5) % 5;
}

// Get CSS positioning for a relative seat position on the table
export function spotStyle(rel: number): React.CSSProperties {
  const base: React.CSSProperties = {
    position: 'absolute',
    transform: 'translate(-50%, -50%)',
  };
  const positions: Record<number, React.CSSProperties> = {
    0: { left: '50%', top: '84%' },   // Bottom center (you)
    1: { left: '17%', top: '66%' },   // Bottom left
    2: { left: '27%', top: '20%' },   // Top left
    3: { left: '73%', top: '20%' },   // Top right
    4: { left: '83%', top: '66%' },   // Bottom right
  };
  return { ...base, ...positions[rel] };
}

// Get player name for a seat from table data
export function nameForSeat(seat: number | null, table: any): string {
  if (!seat) return '';
  return table?.seats?.[String(seat)] || `Seat ${seat}`;
}

// Check if a seat is AI-controlled
export function isAiSeat(seat: number, table: any): boolean {
  return Boolean(table?.seatIsAI?.[String(seat)]);
}

