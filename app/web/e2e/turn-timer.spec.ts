import { expect, test } from "@playwright/test";
import { createTableAndDeal } from "./helpers";

// A player who never moves: the turn clock shows, the AI plays each timed
// out turn for them, and after three in a row they are moved to spectator
// — from where "Take seat" puts them back at the table. Runs on the real
// 20s clock, so it takes a little over a minute.

test("an idle player is moved to spectator and can take a seat back", async ({
  page,
}) => {
  test.setTimeout(240_000);
  await createTableAndDeal(page, "Idle Tester", "e2e-idle");

  await expect(page.getByLabel(/seconds left to move/)).toBeVisible({
    timeout: 60_000,
  });
  await expect(
    page
      .getByText(/Idle Tester ran out of time; the AI played for them/)
      .first(),
  ).toBeVisible({ timeout: 60_000 });

  await expect(page.getByText(/^Watching/)).toBeVisible({ timeout: 150_000 });
  await expect(
    page.getByText(/Idle Tester missed 3 turns in a row/).first(),
  ).toBeVisible();

  await page
    .getByRole("button", { name: /^Take seat \d/ })
    .first()
    .click();
  await expect(page.getByText(/^Watching/)).toHaveCount(0);
});
