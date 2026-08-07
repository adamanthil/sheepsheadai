import { expect, test } from "@playwright/test";
import { createTableAndDeal } from "./helpers";

// Staged bury flow: as the picker, tapping hand cards stages them face-up in
// the center (tappable to put back) without sending anything; the two BURY
// actions are only sent on Confirm bury, after which play starts.

test("staged bury: stage, put back, confirm", async ({ page }) => {
  // Sit in seat 1: it is always offered the first pick decision, so the AI
  // seats never get a chance to pick first.
  await createTableAndDeal(page, "Bury Tester", "e2e-bury", { takeSeat: 1 });

  await page.getByRole("button", { name: /Pick the blind/ }).click();

  // The engine asks for the partner call before the bury; go alone so every
  // hand card is a legal bury and no under-card step intervenes.
  await page.getByRole("button", { name: "Go alone" }).click();

  // Bury mode: all 8 cards (6 + blind) are stageable.
  await expect(page.getByText("Bury 2 cards")).toBeVisible();
  const clickableCards = page.locator("[data-clickable='true']");
  await expect(clickableCards).toHaveCount(8);

  // Going alone as picker shows your role badge immediately.
  await expect(page.getByText("Picker", { exact: true }).first()).toBeVisible();

  // Stage two cards: they leave the fan for the center, face-up, and the
  // confirm button appears — but nothing has been sent yet.
  await clickableCards.first().click();
  await clickableCards.first().click();
  const confirm = page.getByRole("button", { name: "Confirm bury" });
  await expect(confirm).toBeVisible();
  await expect(clickableCards).toHaveCount(6);

  // Put one back from the center; the confirm button retracts.
  const putBack = page.getByRole("button", {
    name: /Put .+ back in your hand/,
  });
  await putBack.first().click();
  await expect(confirm).toBeHidden();
  await expect(clickableCards).toHaveCount(7);

  // Re-stage and confirm: both BURY actions land and play starts.
  await clickableCards.first().click();
  await confirm.click();
  await expect(page.getByText("Play a card")).toBeVisible({ timeout: 15_000 });
});
