import {
  expect,
  test,
  type APIRequestContext,
  type Browser,
  type Page,
} from "@playwright/test";
import { createTable } from "./helpers";
import type { TablePublic } from "../lib/types";

// A client with no seat watches the public view, and may take over a seat
// mid-hand only if the AI was dealt it: a seat dealt to a human stays with
// the AI covering for them until the hand ends. Four scripted humans and the
// host fill the table (each script holding a socket open so it reads as
// connected), and a sixth client joins as the spectator.

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

/** Host a table, seat four scripted humans and join a sixth client, who
 * finds every seat taken and watches. */
async function fullTableWithWatcher(
  page: Page,
  request: APIRequestContext,
  tableName: string,
) {
  const tableId = await createTable(page, "Host", tableName);
  const humans = [];
  for (let i = 1; i <= 4; i++) {
    const joined = await joinViaApi(request, tableId, `Scripted ${i}`);
    humans.push({ joined, ws: await openSocket(tableId, joined) });
  }
  const watcher = await joinViaApi(request, tableId, "Watcher");
  return { tableId, humans, watcher };
}

async function openAsWatcher(
  browser: Browser,
  tableId: string,
  watcher: Joined,
): Promise<{ spectator: Page; close: () => Promise<void> }> {
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
  return { spectator, close: () => context.close() };
}

async function deal(page: Page) {
  await page.getByRole("button", { name: "Deal cards →" }).click();
  await page.waitForURL(/\/table\//);
}

test("a seat dealt to a human is not offered when they drop", async ({
  browser,
  page,
  request,
}) => {
  const { tableId, humans, watcher } = await fullTableWithWatcher(
    page,
    request,
    "e2e-spectate-held",
  );
  await deal(page);
  const { spectator, close } = await openAsWatcher(browser, tableId, watcher);
  await expect(
    spectator.getByText("No seats to take over this hand"),
  ).toBeVisible();

  // After the (e2e) 5s grace an AI covers the seat, but keeps it for its
  // owner. (No chat notice to wait on: it is muted so soon after the join.)
  const publicTable = async () => {
    const res = await request.get(`${apiBase}/api/tables`);
    const tables = (await res.json()) as TablePublic[];
    return tables.find((t) => t.id === tableId)!;
  };
  const before = await publicTable();
  const seat = Object.keys(before.seatOccupants).find(
    (s) => before.seatOccupants[s] === humans[0].joined.client_id,
  )!;
  humans[0].ws.close();
  await expect
    .poll(async () => (await publicTable()).seatIsAI[seat], {
      timeout: 30_000,
    })
    .toBe(true);
  const after = await publicTable();
  expect(after.seatTakeable[seat]).toBe(false);
  const aiName = after.seats[seat]!;
  // The spectator has the same table_update: the AI's name is in the ring.
  await expect(spectator.getByText(aiName).first()).toBeVisible();
  await expect(
    spectator.getByText("No seats to take over this hand"),
  ).toBeVisible();
  await expect(
    spectator.getByRole("button", { name: /^Take seat/ }),
  ).toHaveCount(0);

  for (const h of humans.slice(1)) h.ws.close();
  await close();
});

test("a spectator takes over a seat the AI was dealt", async ({
  browser,
  page,
  request,
}) => {
  const { tableId, humans, watcher } = await fullTableWithWatcher(
    page,
    request,
    "e2e-spectate-take",
  );
  // Dropping before the deal hands the seat to an AI at once, so the AI
  // is dealt it.
  humans[0].ws.close();
  await expect(
    page.getByRole("group", { name: /^Seat \d$/ }).filter({
      hasText: "Scripted 1",
    }),
  ).toHaveCount(0, { timeout: 15_000 });
  await deal(page);

  const { spectator, close } = await openAsWatcher(browser, tableId, watcher);
  const takeSeat = spectator.getByRole("button", { name: /^Take seat \d/ });
  await expect(takeSeat).toHaveCount(1);
  await takeSeat.click();

  await expect(spectator.getByText(/^Watching/)).toHaveCount(0);
  await expect(
    spectator.getByText("You", { exact: true }).first(),
  ).toBeVisible();

  for (const h of humans.slice(1)) h.ws.close();
  await close();
});
