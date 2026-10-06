import { expect, test } from "@playwright/test";
import { createTableAndDeal, joinInNewPage } from "./helpers";

// Leave and Close table each offer to wait for the hand in play to end.
// Nobody moves; the (2s, in e2e) turn clock plays the hand out.

test("a guest leaves once the hand ends, onto its final scores", async ({
  browser,
  page,
}) => {
  await createTableAndDeal(page, "Host", "e2e-leave-after");
  const tableId = page.url().split("/table/")[1];
  const guest = await joinInNewPage(browser, tableId, "Guest");
  await guest.waitForURL(/\/table\//);

  await guest.getByText("Leave", { exact: true }).click();
  await expect(guest.getByText("Leave now", { exact: true })).toBeVisible();
  await guest.getByText("After this hand", { exact: true }).click();
  await expect(guest.getByText("Leaving after this hand")).toBeVisible();
  // Everyone else sees who is on their way out.
  await expect(page.getByText(/· leaving/)).toBeVisible();

  await expect(guest.getByText("You left after hand 1")).toBeVisible({
    timeout: 100_000,
  });
  await expect(guest.getByRole("button", { name: "Rejoin" })).toBeVisible();
  await expect(guest.getByText("Connection rejected")).toHaveCount(0);
  // The host plays on.
  await expect(page.getByRole("button", { name: "Redeal →" })).toBeVisible();
});

test("the host ends the table once the hand ends", async ({ page }) => {
  await createTableAndDeal(page, "Host", "e2e-close-after");

  await page.getByRole("button", { name: "Close table" }).click();
  await expect(page.getByRole("button", { name: "Close now" })).toBeVisible();
  await page.getByRole("button", { name: "End after this hand" }).click();
  await expect(page.getByText("Table ends after this hand")).toBeVisible();

  await expect(page.getByText("The host ended the table")).toBeVisible({
    timeout: 100_000,
  });
  await expect(page.getByRole("button", { name: "Lobby" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Rejoin" })).toHaveCount(0);
});
