import { expect, test, type APIRequestContext } from "@playwright/test";
import { createTable } from "./helpers";

// A client with no seat watches the public view and can take over an AI
// seat mid-hand. Four scripted humans fill the table (each holding a socket
// open so it reads as connected), a sixth client joins as a spectator, and
// one scripted human drops so the disconnect grace hands their seat to an
// AI — which the spectator then takes over from the SpectatorBar.

const apiBase = "http://127.0.0.1:9100";

type Joined = { client_id: string; session_token: string };

async function joinViaApi(
  request: APIRequestContext,
  tableId: string,
  name: string,
): Promise<Joined> {
  const res = await request.post(`${apiBase}/api/tables/${tableId}/join`, {
    data: { display_name: name },
  });
  expect(res.ok()).toBe(true);
  return (await res.json()) as Joined;
}

function openSocket(tableId: string, joined: Joined): Promise<WebSocket> {
  const ws = new WebSocket(`ws://127.0.0.1:9100/ws/table/${tableId}`, [
    `sheepshead.client.${joined.client_id}`,
    `sheepshead.token.${joined.session_token}`,
  ]);
  return new Promise((resolve, reject) => {
    ws.onopen = () => resolve(ws);
    ws.onerror = () => reject(new Error("scripted socket failed"));
  });
}

test("spectator takes over an AI seat mid-hand", async ({
  browser,
  page,
  request,
}) => {
  const tableId = await createTable(page, "Host", "e2e-spectate");

  const humans = [];
  for (let i = 1; i <= 4; i++) {
    const joined = await joinViaApi(request, tableId, `Scripted ${i}`);
    humans.push({ joined, ws: await openSocket(tableId, joined) });
  }
  // All five seats are human, so this join lands unseated.
  const watcher = await joinViaApi(request, tableId, "Watcher");

  await page.getByRole("button", { name: "Deal cards →" }).click();
  await page.waitForURL(/\/table\//);

  const context = await browser.newContext();
  const spectator = await context.newPage();
  await spectator.addInitScript(
    ([id, clientId, token]) => {
      window.localStorage.setItem(`sheepshead_client_id_${id}`, clientId);
      window.localStorage.setItem("sheepshead_session_token", token);
    },
    [tableId, watcher.client_id, watcher.session_token],
  );
  await spectator.goto(`/table/${tableId}`);
  await expect(spectator.getByText(/^Watching/)).toBeVisible();
  await expect(spectator.getByText("No AI seats to take over")).toBeVisible();

  // The scripted human drops; after the 10s grace an AI holds their seat.
  humans[0].ws.close();
  const takeSeat = spectator.getByRole("button", { name: /^Take seat \d/ });
  await expect(takeSeat).toBeVisible({ timeout: 30_000 });
  await takeSeat.click();

  await expect(spectator.getByText(/^Watching/)).toHaveCount(0);
  await expect(
    spectator.getByText("You", { exact: true }).first(),
  ).toBeVisible();

  for (const h of humans.slice(1)) h.ws.close();
  await context.close();
});
