/** Human-readable message for a failed API response.
 *
 * Reads the FastAPI JSON `detail` when present. A 404 gets a specific
 * diagnosis: the analysis endpoints ship with this UI, so "route not
 * found" almost always means the API server is running older code than
 * the page (e.g. started from another checkout or before a pull) — not
 * that the user's inputs were wrong.
 */
export async function apiErrorMessage(
  res: Response,
  endpoint: string,
): Promise<string> {
  let detail: string | null = null;
  try {
    const data = await res.json();
    if (data && typeof data.detail === "string") {
      detail = data.detail;
    }
  } catch {
    // Non-JSON error body; fall through to the generic message.
  }

  if (res.status === 404) {
    return (
      `The API server has no ${endpoint} endpoint (404). ` +
      "It is likely running older code than this page — restart the " +
      "server from this branch and try again."
    );
  }

  return detail ?? `Request failed: ${res.status} ${res.statusText}`;
}

/** Message for a fetch that never produced a response (server down,
 * wrong port, CORS): the browser only reports "Failed to fetch". */
export function fetchFailureMessage(err: unknown, fallback: string): string {
  if (err instanceof TypeError) {
    return (
      "Cannot reach the analysis API server — check that it is running " +
      "(dev: port 9000) and restart it from this branch if needed."
    );
  }
  return err instanceof Error ? err.message : fallback;
}

/** A failed API call whose cause lies outside the browser's code: no
 * response, or one that wasn't ok. The api reports its own 5xx errors and a
 * 4xx is an answer, so lib/reportCaught.ts doesn't report these. */
export class ApiFailure extends Error {}

/** An API response that wasn't ok, thrown where a caller handles every
 * failure in one catch. */
export class HttpStatusError extends ApiFailure {
  constructor(readonly status: number) {
    super(`HTTP ${status}`);
    this.name = "HttpStatusError";
  }
}
