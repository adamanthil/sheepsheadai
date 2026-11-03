import { expect, test } from "@playwright/test";
import { createTableAndDeal, joinInNewPage } from "./helpers";

// Only the host can deal the next hand, so only the host is offered the button when a
// hand ends; everyone else is told who they're waiting on. Both players sit
// out their turns and the (2s, in e2e) turn clock plays the hand out.

test("only the host is offered Next hand when the hand ends", async ({
  browser,
  page,
}) => {
  await createTableAndDeal(page, "Host", "e2e-next-hand");
  const tableId = page.url().split("/table/")[1];
  const guest = await joinInNewPage(browser, tableId, "Guest");
  await guest.waitForURL(/\/table\//);

  await expect(page.getByRole("button", { name: "Next hand →" })).toBeVisible({
    timeout: 100_000,
  });
  await expect(
    guest.getByText("Waiting for Host to deal the next hand…"),
  ).toBeVisible();
  await expect(guest.getByRole("button", { name: "Next hand →" })).toHaveCount(
    0,
  );
});
