import { expect, type Page } from "@playwright/test";

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
