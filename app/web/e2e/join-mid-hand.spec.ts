import { expect, test } from "@playwright/test";
import { createTableAndDeal, joinInNewPage } from "./helpers";

// A player who joins mid-hand watches it through and takes over from the AI
// at the next deal. Nobody moves; the (2s, in e2e) turn clock plays the
// hand out.

test("a mid-hand joiner watches, then sits down at the next deal", async ({
  browser,
  page,
}) => {
  await createTableAndDeal(page, "Host", "e2e-join-mid-hand");
  const tableId = page.url().split("/table/")[1];
  const guest = await joinInNewPage(browser, tableId, "Guest");
  await guest.waitForURL(/\/table\//);

  await expect(guest.getByText(/^Watching/)).toBeVisible();
  await expect(
    guest.getByText(/^You take over from .+ next hand$/),
  ).toBeVisible();
  await expect(
    guest.getByRole("button", { name: "Take seat now" }),
  ).toBeVisible();

  const nextHand = page.getByRole("button", { name: "Next hand →" });
  await expect(nextHand).toBeVisible({ timeout: 100_000 });
  // Between hands the next deal seats them anyway: no button, just the wait.
  await expect(guest.getByText("Waiting for Host to deal…")).toBeVisible();
  await expect(
    guest.getByRole("button", { name: "Take seat now" }),
  ).toHaveCount(0);
  await nextHand.click();

  await expect(guest.getByText(/^Watching/)).toHaveCount(0);
  await expect(guest.getByText("You", { exact: true }).first()).toBeVisible();
});
