import { expect, test, type Page } from "@playwright/test";
import { createTable } from "./helpers";

// Buttons that wait on the server answer the press at once: Deal cards
// shows it is busy, and a house rule switches before the server confirms
// it (and back, if the server refuses). Each test holds or refuses the
// request to tell the immediate response from the server's.

const HOLD_MS = 900;
// Shorter than the hold: what is checked within it came before the server.
const BEFORE_SERVER = { timeout: HOLD_MS - 400 };

async function hold(page: Page, url: string) {
  await page.route(url, async (route) => {
    await new Promise((resolve) => setTimeout(resolve, HOLD_MS));
    await route.continue();
  });
}

/** The rules PATCH needs the session the page hydrates after mount, and a
 * rule pressed before then does nothing: press until it takes. */
async function pressRule(page: Page, name: string) {
  const button = page.getByRole("button", { name, exact: true });
  await expect
    .poll(
      async () => {
        await button.click();
        return button.getAttribute("aria-pressed");
      },
      { timeout: 30_000 },
    )
    .toBe("true");
  return button;
}

test("Deal cards shows it is busy until the table opens", async ({ page }) => {
  await createTable(page, "Busy Host", "e2e-busy-deal");
  const deal = page.getByRole("button", { name: "Deal cards →" });
  await expect(deal).toBeEnabled();

  await hold(page, "**/api/tables/*/start");
  await deal.click();

  await expect(deal).toHaveAttribute("aria-busy", "true", BEFORE_SERVER);
  await expect(deal).toBeDisabled();
  await page.waitForURL(/\/table\//);
});

test("a house rule switches before the server answers", async ({ page }) => {
  await createTable(page, "Rules Host", "e2e-busy-rules");
  // Once the session is up (a first switch lands), hold the next one.
  await pressRule(page, "Doublers");

  await hold(page, "**/api/tables/*/rules");
  const leasters = page.getByRole("button", { name: "Leasters", exact: true });
  await leasters.click();
  await expect(leasters).toHaveAttribute("aria-pressed", "true", BEFORE_SERVER);
});

test("a refused house rule switches back", async ({ page }) => {
  await createTable(page, "Refused Host", "e2e-busy-refused");
  const doublers = await pressRule(page, "Doublers");

  await page.route(
    "**/api/tables/*/rules",
    (route) => route.fulfill({ status: 403, json: { detail: "not_host" } }),
    { times: 1 },
  );
  const leasters = page.getByRole("button", { name: "Leasters", exact: true });
  await leasters.click();

  await expect(page.getByText("not_host")).toBeVisible();
  await expect(doublers).toHaveAttribute("aria-pressed", "true");
  await expect(leasters).toHaveAttribute("aria-pressed", "false");
});
