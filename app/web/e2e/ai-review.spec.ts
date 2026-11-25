import { expect, test, type Page } from "@playwright/test";
import {
  GAME_ID,
  handReview,
  hands,
  me,
  queue,
  totals,
} from "./reviewFixtures";

// The AI review pages, drawn from a fixed hand. The account API is stubbed
// in the browser: reviewing a real hand needs a registered, verified
// account, a played-out hand and the background worker, all of which have
// their own server tests. What is checked here is that the pages put a
// review on screen and that the ways between them work.

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "*",
};

async function signIn(page: Page, { isAdmin = false } = {}) {
  const bodies: Record<string, unknown> = {
    "/api/account/me": me(isAdmin),
    "/api/account/hands": hands,
    [`/api/account/hands/${GAME_ID}/review`]: handReview,
    "/api/account/review": totals,
    "/api/admin/review-queue": queue,
  };
  await page.route(/\/api\/(account|admin)\//, (route) => {
    const request = route.request();
    if (request.method() === "OPTIONS") {
      return route.fulfill({ status: 204, headers: CORS });
    }
    const body = bodies[new URL(request.url()).pathname];
    return body === undefined
      ? route.fulfill({ status: 404, headers: CORS, json: { detail: "none" } })
      : route.fulfill({ headers: CORS, json: body });
  });
  await page.addInitScript(() => {
    window.localStorage.setItem("sheepshead_session_token", "fixture-token");
    window.localStorage.setItem(
      "sheepshead_player_id",
      "99999999-2222-4333-8444-555555555555",
    );
  });
}

test("hand history opens on results and leads into a hand's review", async ({
  page,
}) => {
  await signIn(page);
  await page.goto("/account/history");

  // Results first: the table's name is a column only there.
  await expect(page.getByRole("columnheader", { name: "Table" })).toBeVisible();

  await page.getByRole("button", { name: "AI comparison" }).click();
  await expect(
    page.getByRole("columnheader", { name: "Net vs AI" }),
  ).toBeVisible();

  await page.getByRole("button", { name: /Show decisions/ }).click();
  await expect(
    page.getByText("You led the Q♣. The AI leads the J♦."),
  ).toBeVisible();

  await page.getByRole("link", { name: "Open full review →" }).click();
  await page.waitForURL(`**/account/history/${GAME_ID}`);
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "You led the Q♣. The AI leads the J♦.",
    }),
  ).toBeVisible();
});

test("the hand page shows each decision against the AI's", async ({ page }) => {
  await signIn(page);
  await page.goto(`/account/history/${GAME_ID}?d=1`);
  const detail = page.getByRole("region", { name: "Selected decision" });

  // The call: the two cards from the blind are pointed out, and the deal
  // can be opened in the pick analyzer.
  await expect(
    detail.getByRole("heading", { name: "You called the A♥. So does the AI." }),
  ).toBeVisible();
  await expect(detail.getByText("Blind", { exact: true })).toHaveCount(2);
  await expect(
    detail.getByRole("link", { name: /Open in Analyze AI/ }),
  ).toHaveAttribute("href", /\/analyze\/pick\?/);

  // A card play, chosen on the chart: it says "played", not "led", for a
  // card that followed, shows the trick so far, and has no analyzer link.
  await page.getByRole("button", { name: /^Trick 2: expected score/ }).click();
  await expect(
    detail.getByRole("heading", {
      name: "You played the A♦. The AI plays the Q♦.",
    }),
  ).toBeVisible();
  await expect(detail.getByText("The trick")).toBeVisible();
  await expect(detail.getByText("+1.4")).toBeVisible();
  await expect(
    detail.getByRole("link", { name: /Open in Analyze AI/ }),
  ).toHaveCount(0);

  await detail.getByRole("button", { name: "← Previous" }).click();
  await expect(
    detail.getByRole("heading", {
      name: "You led the Q♣. The AI leads the J♦.",
    }),
  ).toBeVisible();

  // The way back lands on the AI comparison, not the results.
  await page.getByRole("link", { name: "← Hand history" }).click();
  await page.waitForURL("**/account/history?view=ai");
  await expect(
    page.getByRole("columnheader", { name: "Net vs AI" }),
  ).toBeVisible();
});

test("the AI review tab counts differences for the kind chosen", async ({
  page,
}) => {
  await signIn(page);
  await page.goto("/account/review");

  await expect(
    page.getByRole("button", { name: "Cost you most · 1" }),
  ).toBeVisible();
  const difference = page.getByRole("link", { name: /Review this decision/ });
  await expect(difference).toHaveCount(1);

  const kinds = page.getByRole("group", { name: "Kind of decision" });
  await kinds.getByRole("button", { name: "Bury" }).click();
  await expect(
    page.getByRole("button", { name: "Cost you most · 0" }),
  ).toBeVisible();
  await expect(page.getByText("Nothing here yet.")).toBeVisible();

  await kinds.getByRole("button", { name: "Play" }).click();
  await difference.click();
  await page.waitForURL(`**/account/history/${GAME_ID}?d=2`);
});

test("picking has its own view of the AI review", async ({ page }) => {
  await signIn(page);
  await page.goto("/account/review");
  const differences = page.getByRole("region", { name: "Largest differences" });
  const picking = page.getByRole("region", { name: "Picking" });
  await expect(differences).toBeVisible();
  await expect(picking).toHaveCount(0);

  await page.getByRole("button", { name: "Picking" }).click();
  await page.waitForURL("**/account/review?view=picking");
  await expect(picking).toBeVisible();
  await expect(differences).toHaveCount(0);

  // The address opens the same view.
  await page.reload();
  await expect(picking).toBeVisible();
});

test("only an administrator is offered the review queue", async ({ page }) => {
  await signIn(page);
  await page.goto("/account/history");
  const tabs = page.getByRole("navigation", { name: "Account" });
  await expect(tabs.getByRole("link", { name: "AI review" })).toBeVisible();
  await expect(tabs.getByRole("link", { name: "Admin" })).toHaveCount(0);
});

test("an administrator sees the review queue", async ({ page }) => {
  await signIn(page, { isAdmin: true });
  await page.goto("/account/history");
  await page
    .getByRole("navigation", { name: "Account" })
    .getByRole("link", { name: "Admin" })
    .click();
  await page.waitForURL("**/account/admin");
  await expect(page.getByText("Reviewing with")).toBeVisible();
  await expect(page.getByText("fixture-model")).toBeVisible();
});
