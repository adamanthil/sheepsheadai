import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAccount } from "../../lib/account";
import type { AccountPublic } from "../../lib/types";

/** The signed-in account, for a page that is only for one. A visitor
 * without an account, or with one that ``allow`` turns down, is sent to
 * the account page. ``account`` is null until it has loaded, and stays
 * null for a visitor who is being sent away. */
export function useRequiredAccount(
  allow: (account: AccountPublic) => boolean = () => true,
): { account: AccountPublic | null; refresh: () => Promise<void> } {
  const router = useRouter();
  const { me, refresh } = useAccount();
  const signedIn = me?.account ?? null;
  const account = signedIn && allow(signedIn) ? signedIn : null;
  const loaded = me !== undefined;

  useEffect(() => {
    if (loaded && !account) router.replace("/account");
  }, [loaded, account, router]);

  return { account, refresh };
}
