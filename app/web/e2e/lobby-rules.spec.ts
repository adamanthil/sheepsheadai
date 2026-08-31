import { expect, test } from "@playwright/test";

// The lobby names each table's house rules, so a player knows what kind of
// game they are joining before they join it.

test("lobby lists each table's game mode", async ({ page, browser }) => {
  const suffix = Date.now();
  const api = await browser.newContext({ baseURL: "http://127.0.0.1:9100" });
  const seeded = [
    [`Doublers ${suffix}`, { partnerMode: 1, allPassMode: "doublers" }],
    [`Jack ${suffix}`, { partnerMode: 0, allPassMode: "leasters" }],
  ] as const;
  for (const [name, rules] of seeded) {
    const created = await api.request.post("/api/tables", {
      data: { name, fillWithAI: true, rules },
    });
    expect(created.ok()).toBeTruthy();
  }

  await page.goto("/");

  const row = (name: string) =>
    page.locator('div[class$="__row"]').filter({ hasText: name });

  await expect(row(`Doublers ${suffix}`)).toContainText("Called Ace");
  await expect(row(`Doublers ${suffix}`)).toContainText("Doublers");
  await expect(row(`Jack ${suffix}`)).toContainText("Jack of Diamonds");
  await expect(row(`Jack ${suffix}`)).toContainText("Leasters");
});
