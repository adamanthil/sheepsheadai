import { expect, type Browser, type Page } from "@playwright/test";

/** Create a table from the home page and land in its waiting room. Returns
 * the table id. */
export async function createTable(
  page: Page,
  playerName: string,
  tableName: string,
): Promise<string> {
  await page.goto("/");

  // The home page hydrates identity from localStorage after mount, which can
  // clobber values typed too early — refill until the create button enables.
  const nameInput = page.locator("label:has-text('Your name') + input");
  const tableInput = page.locator("label:has-text('Table name') + input");
  const createButton = page.getByRole("button", { name: "Create table →" });
  await expect(createButton).toBeVisible();
  await expect
    .poll(
      async () => {
        await nameInput.fill(playerName);
        await tableInput.fill(tableName);
        return createButton.isEnabled();
      },
      { timeout: 30_000 },
    )
    .toBe(true);
  await createButton.click();

  await page.waitForURL(/\/waiting\//);
  return page.url().split("/waiting/")[1];
}

/**
 * Create a table, fill the empty seats with AI, and deal — landing on the
 * table view. Seat assignment is not deterministic; pass `takeSeat` to sit
 * in a specific seat before dealing (seat 1 holds the first pick decision).
 */
export async function createTableAndDeal(
  page: Page,
  playerName: string,
  tableName: string,
  opts: { takeSeat?: number } = {},
) {
  await createTable(page, playerName, tableName);

  if (opts.takeSeat) {
    const seatCard = page
      .locator('[class*="card"]')
      .filter({ hasText: new RegExp(`^Seat ${opts.takeSeat}`) })
      .first();
    // Wait for the seat grid to render before deciding anything — count()
    // does not auto-wait, so checking the button straight away races the
    // table fetch and silently skips the seat change.
    await expect(seatCard).toBeVisible();
    if (!((await seatCard.textContent()) ?? "").includes(playerName)) {
      await seatCard
        .getByRole("button", { name: /Take this seat|Take over/ })
        .click();
      await expect(seatCard).toContainText(playerName);
    }
  }

  await page.getByRole("button", { name: /Fill (empty )?with AI/ }).click();
  await page.getByRole("button", { name: "Deal cards →" }).click();
  await page.waitForURL(/\/table\//);
}

const apiBase = "http://127.0.0.1:9100";

/** Join ``tableId`` as ``name`` through the API in a fresh browser context
 * (its own identity) and open the waiting room there. */
export async function joinInNewPage(
  browser: Browser,
  tableId: string,
  name: string,
): Promise<Page> {
  const context = await browser.newContext();
  const page = await context.newPage();
  const res = await page.request.post(`${apiBase}/api/tables/${tableId}/join`, {
    data: { display_name: name },
  });
  expect(res.ok()).toBe(true);
  const joined = await res.json();
  await page.addInitScript(
    ([id, clientId, token]) => {
      window.localStorage.setItem(`sheepshead_client_id_${id}`, clientId);
      window.localStorage.setItem("sheepshead_session_token", token);
    },
    [tableId, joined.client_id, joined.session_token],
  );
  await page.goto(`/waiting/${tableId}`);
  return page;
}
