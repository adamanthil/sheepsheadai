import { expect, test } from "@playwright/test";
import { createTable, joinInNewPage } from "./helpers";

// The host removes a player, from their seat card or from one of their
// chat messages; the removed player is sent home with a notice.

test("host removes a player from their seat", async ({ browser, page }) => {
  const tableId = await createTable(page, "Host", "e2e-kick-seat");
  const pest = await joinInNewPage(browser, tableId, "Pest");

  const seatCard = page
    .locator('[class*="card"]')
    .filter({ hasText: "Pest" })
    .first();
  await seatCard.getByRole("button", { name: "Remove" }).click();
  await seatCard.getByRole("button", { name: "Remove" }).click();

  await expect(seatCard).toHaveCount(0);
  await pest.waitForURL((url) => url.pathname === "/");
  await expect(
    pest.getByText("The host removed you from that table."),
  ).toBeVisible();
});

test("host removes a player from their chat message", async ({
  browser,
  page,
}) => {
  const tableId = await createTable(page, "Host", "e2e-kick-chat");
  const chatty = await joinInNewPage(browser, tableId, "Chatty");
  // Chat history arriving means Chatty's socket is up; a message typed
  // before that would be dropped.
  await expect(chatty.getByText(/Chatty joined/).first()).toBeVisible();
  await chatty.getByPlaceholder("Type a message...").fill("hello all");
  await chatty.getByPlaceholder("Type a message...").press("Enter");

  const chat = page.locator('[class*="chatPanel"]');
  await expect(chat.getByText("hello all")).toBeVisible();
  await chat.getByRole("button", { name: "Chatty:" }).click();
  await chat.getByRole("button", { name: "Remove" }).click();
  await expect(chat.getByText("Remove Chatty?")).toBeVisible();
  await chat.getByRole("button", { name: "Remove", exact: true }).click();

  await expect(
    page.getByText("Chatty was removed by the host").first(),
  ).toBeVisible();
  await chatty.waitForURL((url) => url.pathname === "/");
});
