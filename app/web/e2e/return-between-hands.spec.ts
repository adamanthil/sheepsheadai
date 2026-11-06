import { expect, test } from "@playwright/test";
import { createTableAndDeal } from "./helpers";

// Between hands the table is "finished" until the host deals again, and
// only the table page can deal it. A player who comes back then -- by the
// waiting room's address or the lobby's Join -- must land on the table
// with Next hand, not in a waiting room whose Deal cards the server
// refuses. The (2s, in e2e) turn clock plays the hand out.

test("a player back between hands lands on the table", async ({ page }) => {
  await createTableAndDeal(page, "Host", "e2e-return-between");
  const tableId = page.url().split("/table/")[1];
  const nextHand = page.getByRole("button", { name: "Next hand →" });
  await expect(nextHand).toBeVisible({ timeout: 100_000 });

  await page.goto(`/waiting/${tableId}`);
  await page.waitForURL(`**/table/${tableId}`);
  await expect(nextHand).toBeVisible();

  await page.goto("/");
  await page
    .getByRole("listitem")
    .filter({ hasText: "e2e-return-between" })
    .getByRole("button", { name: "Join →" })
    .click();
  await page.waitForURL(`**/table/${tableId}`);

  await nextHand.click();
  await expect(nextHand).toHaveCount(0);
  await expect(page.getByRole("group", { name: "Your hand" })).toBeVisible();
});
