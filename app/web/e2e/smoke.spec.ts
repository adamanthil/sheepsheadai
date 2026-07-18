import { expect, test } from "@playwright/test";

// Happy path: create a table, fill the remaining seats with AI, deal, then
// act whenever the UI offers a decision until this player has played a card
// into a trick. Exercises the REST create/join/start flow, the table
// websocket, the AI turn loop, and hand persistence end to end.

test("create table, start with AI, play a card", async ({ page }) => {
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
        await nameInput.fill("Smoke Tester");
        await tableInput.fill("e2e-smoke");
        return createButton.isEnabled();
      },
      { timeout: 30_000 },
    )
    .toBe(true);
  await createButton.click();

  await page.waitForURL(/\/waiting\//);
  await page.getByRole("button", { name: /Fill (empty )?with AI/ }).click();
  await page.getByRole("button", { name: "Deal cards →" }).click();

  await page.waitForURL(/\/table\//);

  const actionButton = page
    .locator("button")
    .filter({ hasText: /^(PICK|PASS|ALONE|JD PARTNER|CALL |BURY |UNDER |PLAY )/ })
    .first();
  const clickableCard = page.locator("[data-clickable='true']").first();

  let playedCard = false;
  for (let turn = 0; turn < 40 && !playedCard; turn++) {
    if (await actionButton.isVisible().catch(() => false)) {
      const label = (await actionButton.textContent()) ?? "";
      await actionButton.click();
      if (label.trim().startsWith("PLAY ")) playedCard = true;
      continue;
    }
    if (await clickableCard.isVisible().catch(() => false)) {
      await clickableCard.click();
      playedCard = true;
      continue;
    }
    await page.waitForTimeout(500);
  }

  expect(playedCard).toBe(true);
});
