import { expect, test } from "@playwright/test";
import { createTable } from "./helpers";

// A deploy restarts the server, which forgets every table. The server
// announces the restart; the reconnect then finds the table gone and the
// player lands in the lobby told why. The socket is routed so the test can
// play the server's part.

/** Code the server closes a socket with when its table doesn't exist. */
const TABLE_GONE = 4404;

test("a restart is announced, then explained in the lobby", async ({
  page,
}) => {
  let toPage!: { send: (msg: string) => void; close: (o: object) => void };
  await page.routeWebSocket(/\/ws\//, (ws) => {
    ws.connectToServer();
    toPage = ws;
  });
  await createTable(page, "Host", "e2e-restart");
  await expect(
    page.getByRole("button", { name: "Deal cards →" }),
  ).toBeVisible();

  toPage.send(JSON.stringify({ type: "server_restart" }));
  await expect(page.getByText("Server restarting…")).toBeVisible();

  toPage.close({ code: TABLE_GONE });
  await page.waitForURL((url) => url.pathname === "/");
  await expect(
    page.getByText(/The server restarted for an update/),
  ).toBeVisible();
});

test("a table gone without warning says it closed", async ({ page }) => {
  let toPage!: { close: (o: object) => void };
  await page.routeWebSocket(/\/ws\//, (ws) => {
    ws.connectToServer();
    toPage = ws;
  });
  await createTable(page, "Host", "e2e-gone");
  await expect(
    page.getByRole("button", { name: "Deal cards →" }),
  ).toBeVisible();

  toPage.close({ code: TABLE_GONE });
  await page.waitForURL((url) => url.pathname === "/");
  await expect(page.getByText("That table has closed.")).toBeVisible();
});
