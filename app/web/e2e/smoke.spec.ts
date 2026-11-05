import { expect, test } from "@playwright/test";
import {
  createTableAndDeal,
  handCards,
  pickAloneAndBury,
  playableCards,
} from "./helpers";

// Happy path: create a table (AIs hold the other seats), deal, pick, bury
// and lead a card into the first trick. Exercises the REST create/join/start
// flow, the table websocket, the AI turn loop, and hand persistence end to
// end.

test("create table, start with AI, play a card", async ({ page }) => {
  await createTableAndDeal(page, "Smoke Tester", "e2e-smoke", {
    takeSeat: 1,
  });
  await pickAloneAndBury(page);
  await expect(handCards(page)).toHaveCount(6);

  await playableCards(page).first().click();

  await expect(handCards(page)).toHaveCount(5);
  await expect(page.getByText(/^Waiting on /)).toBeVisible();
});
