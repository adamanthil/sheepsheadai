import { expect, test } from "@playwright/test";
import { createTableAndDeal } from "./helpers";

// Happy path: create a table, fill the remaining seats with AI, deal, then
// act whenever the UI offers a decision until this player has played a card
// into a trick. Exercises the REST create/join/start flow, the table
// websocket, the AI turn loop, and hand persistence end to end.

test("create table, start with AI, play a card", async ({ page }) => {
  await createTableAndDeal(page, "Smoke Tester", "e2e-smoke");

  const actionButton = page
    .locator("button")
    .filter({
      hasText: /^(PICK|PASS|ALONE|JD PARTNER|CALL |BURY |UNDER |PLAY )/,
    })
    .first();
  const clickableCard = page.locator("[data-clickable='true']").first();

  let playedCard = false;
  for (let turn = 0; turn < 40 && !playedCard; turn++) {
    if (await actionButton.isVisible().catch(() => false)) {
      const label = (await actionButton.textContent()) ?? "";
      // A state frame can detach the button mid-click (AI turns resolve
      // fast); bound the click and let the loop re-evaluate instead of
      // retrying one click for the whole test timeout.
      const clicked = await actionButton
        .click({ timeout: 3000 })
        .then(() => true)
        .catch(() => false);
      if (clicked && label.trim().startsWith("PLAY ")) playedCard = true;
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
