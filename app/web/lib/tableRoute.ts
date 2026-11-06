/** Where a seated player belongs at a table: its waiting room until the
 * first deal, the table itself from then on, between hands included (a
 * "finished" table is one whose last hand is over and the next not yet
 * dealt -- only the table page can deal it). */
export function tableRoute(tableId: string, status: string | undefined) {
  return status === undefined || status === "open"
    ? `/waiting/${tableId}`
    : `/table/${tableId}`;
}
