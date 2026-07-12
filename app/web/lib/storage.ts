export const STORAGE_KEYS = {
  clientId: (tableId: string) => `sheepshead_client_id_${tableId}`,
  playerId: "sheepshead_player_id",
  displayName: "sheepshead_display_name",
  sessionToken: "sheepshead_session_token",
} as const;
