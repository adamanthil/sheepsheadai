import { expect, test } from "@playwright/test";

// The lobby names each table's house rules, so a player knows what kind of
// game they are joining before they join it.

test("lobby lists each table's game mode", async ({ page, browser }) => {
  const suffix = Date.now();
  const api = await browser.newContext({ baseURL: "http://127.0.0.1:9100" });
  const seeded = [
    [`Doublers ${suffix}`, { partnerMode: 1, allPassMode: "doublers" }],
    [`Jack ${suffix}`, { partnerMode: 0, allPassMode: "leasters" }],
  ] as const;
  for (const [name, rules] of seeded) {
    const created = await api.request.post("/api/tables", {
      data: { name, fillWithAI: true, rules },
    });
    expect(created.ok()).toBeTruthy();
  }

  await page.goto("/");

  const row = (name: string) =>
    page.locator('div[class$="__row"]').filter({ hasText: name });

  await expect(row(`Doublers ${suffix}`)).toContainText("Called Ace");
  await expect(row(`Doublers ${suffix}`)).toContainText("Doublers");
  await expect(row(`Jack ${suffix}`)).toContainText("Jack of Diamonds");
  await expect(row(`Jack ${suffix}`)).toContainText("Leasters");
});

test("a rule change in the waiting room reaches an open lobby", async ({
  page,
  context,
}) => {
  const tableName = `Live ${Date.now()}`;

  // Tab 1: host the table and land in the waiting room.
  const nameInput = page.locator("label:has-text('Your name') + input");
  const tableInput = page.locator("label:has-text('Table name') + input");
  const createButton = page.getByRole("button", { name: "Create table →" });
  await page.goto("/");
  await expect(createButton).toBeVisible();
  await expect
    .poll(
      async () => {
        await nameInput.fill("Host");
        await tableInput.fill(tableName);
        return createButton.isEnabled();
      },
      { timeout: 30_000 },
    )
    .toBe(true);
  await createButton.click();
  await page.waitForURL(/\/waiting\//);

  // Tab 2: a player sitting on the lobby, reading the default.
  const lobby = await context.newPage();
  await lobby.goto("/");
  const row = lobby
    .locator('div[class$="__row"]')
    .filter({ hasText: tableName });
  await expect(row).toContainText("Leasters");

  // Host switches to doublers; the open lobby follows without a reload.
  const doublers = page.getByRole("button", { name: "Doublers", exact: true });
  await expect
    .poll(
      async () => {
        await doublers.click();
        const cls = (await doublers.getAttribute("class")) ?? "";
        return cls.includes("segActive");
      },
      { timeout: 30_000 },
    )
    .toBe(true);

  await expect(row).toContainText("Doublers", { timeout: 15_000 });
  await expect(row).not.toContainText("Leasters");
});
