import { expect, test } from "@playwright/test";

// The doublers house rule as the host sets it: the waiting-room control
// writes it to the table, and the table header badge reports it back. The
// throw-in itself needs all five seats to pass, which the AI seats do not
// do on demand — that half is covered by app/server/tests/test_doublers.py.

test("host picks doublers and the table badges it", async ({ page }) => {
  await page.goto("/");

  const nameInput = page.locator("label:has-text('Your name') + input");
  const tableInput = page.locator("label:has-text('Table name') + input");
  const createButton = page.getByRole("button", { name: "Create table →" });
  await expect(createButton).toBeVisible();
  await expect
    .poll(
      async () => {
        await nameInput.fill("Doubler Tester");
        await tableInput.fill("e2e-doublers");
        return createButton.isEnabled();
      },
      { timeout: 30_000 },
    )
    .toBe(true);
  await createButton.click();
  await page.waitForURL(/\/waiting\//);

  // Leasters is the default, and describes itself.
  const panel = page.locator('div[class*="RulesPanel-module"]').first();
  const leasters = page.getByRole("button", { name: "Leasters", exact: true });
  const doublers = page.getByRole("button", { name: "Doublers", exact: true });
  await expect(page.getByText("When everyone passes")).toBeVisible();
  await expect(page.getByText(/played with no picker/)).toBeVisible();
  await expect(leasters).toHaveClass(/segActive/);

  // The rules PATCH needs the client_id the page hydrates from localStorage
  // after mount, so an early click is a no-op -- retry until it lands, the
  // same way helpers.ts polls the create button.
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
  await expect(page.getByText(/thrown in and redealt/)).toBeVisible();
  await expect(leasters).not.toHaveClass(/segActive/);
  // The selection must survive the table_update the PATCH broadcasts back.
  await page.waitForTimeout(1000);
  await expect(doublers).toHaveClass(/segActive/);
  await panel.screenshot({ path: "test-results/rules-doublers.png" });

  await page.getByRole("button", { name: /Fill (empty )?with AI/ }).click();
  await page.getByRole("button", { name: "Deal cards →" }).click();
  await page.waitForURL(/\/table\//);

  // The rule survives the deal and is badged in the table header.
  const badge = page.getByText(/Called Ace · Doublers/);
  await expect(badge).toBeVisible();
  await page
    .locator('div[class*="TableHeader-module"]')
    .first()
    .screenshot({ path: "test-results/table-header-doublers.png" });
  // At the base stake there is no doubler badge to distract from it.
  await expect(page.getByText(/× Doubler/)).toHaveCount(0);
});
