import { expect, test } from "@playwright/test";
import { createTable, joinInNewPage } from "./helpers";

// Muting hides a player's chat for the viewer only, and can be undone
// from the chat header.

test("a player mutes and unmutes another player's chat", async ({
  browser,
  page,
}) => {
  const tableId = await createTable(page, "Reader", "e2e-mute");
  const chatty = await joinInNewPage(browser, tableId, "Chatty");
  // Chat history arriving means Chatty's socket is up.
  await expect(chatty.getByText(/Chatty joined/).first()).toBeVisible();
  const input = chatty.getByPlaceholder("Type a message...");
  await input.fill("first");
  await input.press("Enter");

  const chat = page.locator('[class*="chatPanel"]');
  await expect(chat.getByText("first")).toBeVisible();
  await chat.getByRole("button", { name: "Chatty:" }).click();
  await chat.getByRole("button", { name: "Mute" }).click();
  await expect(chat.getByText("first")).toHaveCount(0);

  // New messages stay hidden for the reader but not for their author.
  await input.fill("second");
  await input.press("Enter");
  await expect(chatty.getByText("second")).toBeVisible();
  await expect(chat.getByText("second")).toHaveCount(0);

  await chat.getByRole("button", { name: "Chatty muted ×" }).click();
  await expect(chat.getByText("first")).toBeVisible();
  await expect(chat.getByText("second")).toBeVisible();
});
