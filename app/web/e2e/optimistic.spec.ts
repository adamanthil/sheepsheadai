import { expect, test, type Page } from "@playwright/test";
import { createTableAndDeal } from "./helpers";

// A player's own move shows the moment it is made, not when the server
// answers; a move the server refuses goes back. Each test holds or refuses
// the action POST to tell the prediction from the server's state. The e2e
// server runs a 2s turn clock, so holds stay well inside it.

const ACTION_URL = "**/api/tables/*/action";
const HOLD_MS = 900;
// Shorter than the hold: what is checked within it can only be predicted.
const BEFORE_SERVER = { timeout: HOLD_MS - 400 };

async function holdActions(page: Page) {
  await page.route(ACTION_URL, async (route) => {
    await new Promise((resolve) => setTimeout(resolve, HOLD_MS));
    await route.continue();
  });
}

test("a pass shows before the server answers", async ({ page }) => {
  await createTableAndDeal(page, "Pass Tester", "e2e-predict-pass", {
    takeSeat: 1,
  });
  const pass = page.getByRole("button", { name: /Pass the buck/ });
  await expect(pass).toBeVisible();

  await holdActions(page);
  await pass.click();

  await expect(pass).toBeHidden(BEFORE_SERVER);
  await expect(page.getByText(/^Waiting on /)).toBeVisible(BEFORE_SERVER);
});

test("a played card leaves the hand before the server answers", async ({
  page,
}) => {
  await createTableAndDeal(page, "Play Tester", "e2e-predict-play", {
    takeSeat: 1,
  });
  // Pick, go alone and bury two: seat 1 then leads the first trick.
  await page.getByRole("button", { name: /Pick the blind/ }).click();
  await page.getByRole("button", { name: "Go alone" }).click();
  const playable = page.locator("[data-clickable='true']");
  await expect(page.getByText("Bury 2 cards")).toBeVisible();
  await playable.first().click();
  await playable.first().click();
  await page.getByRole("button", { name: "Confirm bury" }).click();
  await expect(page.getByText("Your play")).toBeVisible({ timeout: 15_000 });
  await expect(playable).toHaveCount(6);

  const card = playable.first();
  const name = await card.getAttribute("aria-label");
  expect(name).toBeTruthy();

  await holdActions(page);
  await card.click();

  await expect(page.getByRole("button", { name: name! })).toHaveCount(
    0,
    BEFORE_SERVER,
  );
  await expect(page.locator('[class*="cardSlot"]')).toHaveCount(
    5,
    BEFORE_SERVER,
  );
  await expect(page.getByText("Your play")).toBeHidden(BEFORE_SERVER);
});

test("a refused move goes back, with the reason", async ({ page }) => {
  await createTableAndDeal(page, "Refused Tester", "e2e-predict-refused", {
    takeSeat: 1,
  });
  const pass = page.getByRole("button", { name: /Pass the buck/ });
  await expect(pass).toBeVisible();

  await page.route(
    ACTION_URL,
    (route) =>
      route.fulfill({ status: 400, json: { detail: "not_your_turn" } }),
    { times: 1 },
  );
  await pass.click();

  await expect(page.getByText("Action failed: not_your_turn")).toBeVisible();
  await expect(pass).toBeVisible();
});
