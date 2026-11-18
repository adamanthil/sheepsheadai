/** Reporting for errors a catch block absorbs on purpose.
 *
 * Many fetches degrade quietly instead of failing the page: an empty lobby,
 * a signed-out account line, a leaderboard notice. That's right for an
 * unreachable server, but the same catch also swallows a bug in the code
 * that handles the response, which would then look like a network blip
 * forever. reportCaught() sends the latter to Sentry and drops the expected
 * kinds: the network failing, a request aborted, and API errors
 * (ApiFailure). */
import * as Sentry from "@sentry/nextjs";

import { ApiFailure } from "./apiError";

// What fetch rejects with when no response arrives (Chrome, Safari,
// Firefox). Any other TypeError is a bug.
const NETWORK_FAILURE =
  /^(Failed to fetch|Load failed|NetworkError when attempting to fetch resource\.?)$/;

export function isExpectedFailure(err: unknown): boolean {
  if (err instanceof ApiFailure) return true;
  if (err instanceof DOMException && err.name === "AbortError") return true;
  return err instanceof TypeError && NETWORK_FAILURE.test(err.message);
}

/** Report `err` unless it's an expected failure. `where` names the call
 * site, as a tag to filter on. */
export function reportCaught(err: unknown, where: string): void {
  if (isExpectedFailure(err)) return;
  Sentry.captureException(err, { tags: { caught: where } });
}
