import { expect, test } from "@playwright/test";
import { createTable, joinInNewPage } from "./helpers";

// A player typing in the chat shows to the others, in the chat and on
// their seat, until the message is sent.

test("the others see a player typing until they send", async ({
  browser,
  page,
}) => {
  const tableId = await createTable(page, "Host", "e2e-typing");
  const kyle = await joinInNewPage(browser, tableId, "Kyle");
  // Kyle's join reaching both chats means both sockets are up: typing is
  // relayed live, never replayed to a socket that connects later.
  const chat = page.getByRole("region", { name: "Chat" });
  await expect(chat.getByText(/Kyle joined/)).toBeVisible();
  await expect(kyle.getByText(/Kyle joined/).first()).toBeVisible();

  const input = kyle.getByPlaceholder("Type a message...");
  await input.pressSequentially("one sec");
  await expect(chat.getByText("Kyle is typing")).toBeVisible();
  await expect(page.getByRole("img", { name: "Kyle is typing" })).toBeVisible();
  // Nobody is shown their own typing.
  await expect(kyle.getByText("Kyle is typing")).toHaveCount(0);

  await input.press("Enter");
  await expect(chat.getByText("one sec")).toBeVisible();
  await expect(chat.getByText("Kyle is typing")).toHaveCount(0);
  await expect(page.getByRole("img", { name: "Kyle is typing" })).toHaveCount(
    0,
  );
});
