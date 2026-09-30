"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ds } from "../../lib/ds";
import { apiFetch } from "../../lib/api";
import {
  USERNAME_RE,
  accountErrorMessage,
  forgotPassword,
  login,
  logout,
  register,
  resendVerification,
  switchIdentity,
  useAccount,
  usernameAvailable,
} from "../../lib/account";
import type { AccountMe, AccountStats } from "../../lib/types";
import PageShell from "../components/PageShell";
import styles from "./account.module.css";

type Tab = "signin" | "create" | "forgot";

export default function AccountPage() {
  const { me, refresh } = useAccount();
  const account = me?.account ?? null;

  let body: React.ReactNode;
  if (me === undefined) {
    body = <p className={styles.note}>Loading…</p>;
  } else if (!account) {
    body = <AuthForms me={me} onSignedIn={refresh} />;
  } else if (!account.email_verified) {
    body = <Unverified email={account.email} onSignOut={refresh} />;
  } else {
    body = <Verified username={account.username} onSignOut={refresh} />;
  }

  return (
    <PageShell title={account ? `@${account.username}` : "Your account"}>
      {body}
    </PageShell>
  );
}

function AuthForms({
  me,
  onSignedIn,
}: {
  me: AccountMe | null;
  onSignedIn: () => Promise<void>;
}) {
  const [tab, setTab] = useState<Tab>("signin");
  return (
    <>
      <p className={styles.lede}>
        An account is optional: guests play exactly as before. Signing up
        carries your name to any device, keeps your stats, and puts you on the{" "}
        <Link href="/leaderboard" className={ds.link}>
          leaderboard
        </Link>
        .
      </p>
      <div className={styles.tabs} role="tablist">
        {(
          [
            ["signin", "Sign in"],
            ["create", "Create account"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            className={`${styles.tab} ${tab === key ? styles.tabActive : ""}`}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "signin" && (
        <SignIn onSignedIn={onSignedIn} onForgot={() => setTab("forgot")} />
      )}
      {tab === "create" && <Create me={me} onCreated={onSignedIn} />}
      {tab === "forgot" && <Forgot onBack={() => setTab("signin")} />}
    </>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <label className={styles.field}>
      <span className={ds.overline}>{label}</span>
      {children}
      {hint && <span className={styles.hint}>{hint}</span>}
    </label>
  );
}

function SignIn({
  onSignedIn,
  onForgot,
}: {
  onSignedIn: () => Promise<void>;
  onForgot: () => void;
}) {
  const [loginName, setLoginName] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      switchIdentity(await login(loginName.trim(), password));
      await onSignedIn();
    } catch (err) {
      setError(accountErrorMessage(err));
      setBusy(false);
    }
  };

  return (
    <form className={styles.form} onSubmit={submit}>
      <Field label="Email or username">
        <input
          className={ds.input}
          value={loginName}
          onChange={(e) => setLoginName(e.target.value)}
          autoComplete="username"
          autoCapitalize="none"
          required
        />
      </Field>
      <Field label="Password">
        <input
          className={ds.input}
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
          required
        />
      </Field>
      {error && <p className={styles.error}>{error}</p>}
      <div className={styles.actions}>
        <button
          className={`${ds.btn} ${ds.btnAccent}`}
          disabled={busy || !loginName.trim() || !password}
        >
          {busy ? "Signing in…" : "Sign in →"}
        </button>
        <button type="button" className={styles.textButton} onClick={onForgot}>
          Forgot password?
        </button>
      </div>
    </form>
  );
}

type Availability = "idle" | "checking" | "available" | "taken" | "invalid";

function useUsernameAvailability(username: string): Availability {
  const [state, setState] = useState<Availability>("idle");
  useEffect(() => {
    const name = username.trim();
    if (!name) {
      setState("idle");
      return;
    }
    if (!USERNAME_RE.test(name)) {
      setState("invalid");
      return;
    }
    setState("checking");
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const ok = await usernameAvailable(name);
        if (!cancelled) setState(ok ? "available" : "taken");
      } catch {
        if (!cancelled) setState("idle");
      }
    }, 400);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [username]);
  return state;
}

const AVAILABILITY_HINT: Record<Availability, string> = {
  idle: "3–20 letters, digits, _ or -. Shown on the leaderboard.",
  checking: "Checking…",
  available: "Available.",
  taken: "Taken, or reserved. Try another.",
  invalid: "Use 3–20 letters, digits, _ or -.",
};

function Create({
  me,
  onCreated,
}: {
  me: AccountMe | null;
  onCreated: () => Promise<void>;
}) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const availability = useUsernameAvailability(username);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      switchIdentity(await register(username.trim(), email.trim(), password));
      await onCreated();
    } catch (err) {
      setError(accountErrorMessage(err));
      setBusy(false);
    }
  };

  return (
    <form className={styles.form} onSubmit={submit}>
      {me?.name && (
        <p className={styles.note}>
          Hands you&rsquo;ve played here as <strong>{me.name}</strong> will
          count toward your new account.
        </p>
      )}
      <Field
        label="Username"
        hint={
          <span data-state={availability}>
            {AVAILABILITY_HINT[availability]}
          </span>
        }
      >
        <input
          className={ds.input}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
          autoCapitalize="none"
          maxLength={20}
          required
        />
      </Field>
      <Field label="Email" hint="Only for confirming it's you and resets.">
        <input
          className={ds.input}
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="email"
          required
        />
      </Field>
      <Field label="Password" hint="At least 8 characters.">
        <input
          className={ds.input}
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="new-password"
          minLength={8}
          maxLength={128}
          required
        />
      </Field>
      {error && <p className={styles.error}>{error}</p>}
      <div className={styles.actions}>
        <button
          className={`${ds.btn} ${ds.btnAccent}`}
          disabled={
            busy ||
            availability === "taken" ||
            availability === "invalid" ||
            !email.trim() ||
            password.length < 8
          }
        >
          {busy ? "Creating…" : "Create account →"}
        </button>
      </div>
    </form>
  );
}

function Forgot({ onBack }: { onBack: () => void }) {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await forgotPassword(email.trim());
      setSent(true);
    } catch (err) {
      setError(accountErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  if (sent) {
    return (
      <div className={styles.form}>
        <p className={styles.note}>
          If an account uses <strong>{email.trim()}</strong>, a reset link is on
          its way. It expires in an hour.
        </p>
        <div className={styles.actions}>
          <button type="button" className={styles.textButton} onClick={onBack}>
            ← Back to sign in
          </button>
        </div>
      </div>
    );
  }
  return (
    <form className={styles.form} onSubmit={submit}>
      <Field label="Email">
        <input
          className={ds.input}
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="email"
          required
        />
      </Field>
      {error && <p className={styles.error}>{error}</p>}
      <div className={styles.actions}>
        <button
          className={`${ds.btn} ${ds.btnAccent}`}
          disabled={busy || !email.trim()}
        >
          {busy ? "Sending…" : "Email me a reset link →"}
        </button>
        <button type="button" className={styles.textButton} onClick={onBack}>
          Back
        </button>
      </div>
    </form>
  );
}

function SignOut({ onSignOut }: { onSignOut: () => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  return (
    <button
      type="button"
      className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        await logout();
        await onSignOut();
      }}
    >
      Sign out
    </button>
  );
}

function Unverified({
  email,
  onSignOut,
}: {
  email: string;
  onSignOut: () => Promise<void>;
}) {
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <div className={styles.form}>
      <p className={styles.lede}>
        Almost there. We sent a confirmation link to <strong>{email}</strong>.
        Confirm it to unlock your stats, the leaderboard, and your account badge
        at the table. You can keep playing in the meantime.
      </p>
      <p className={styles.note}>
        Unconfirmed accounts are removed after 7 days (your hands stay).
      </p>
      {status && <p className={styles.note}>{status}</p>}
      <div className={styles.actions}>
        <button
          type="button"
          className={`${ds.btn} ${ds.btnSm}`}
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            try {
              await resendVerification();
              setStatus("Sent. Check your inbox (and spam folder).");
            } catch (err) {
              setStatus(accountErrorMessage(err));
            } finally {
              setBusy(false);
            }
          }}
        >
          Resend the email
        </button>
        <SignOut onSignOut={onSignOut} />
      </div>
    </div>
  );
}

function Verified({
  username,
  onSignOut,
}: {
  username: string;
  onSignOut: () => Promise<void>;
}) {
  const [stats, setStats] = useState<AccountStats | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiFetch("/api/account/stats");
        if (!res.ok) throw new Error(String(res.status));
        const data = (await res.json()) as AccountStats;
        if (!cancelled) setStats(data);
      } catch {
        if (!cancelled) setFailed(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [username]);

  return (
    <div className={styles.form}>
      {failed && <p className={styles.error}>Couldn&rsquo;t load stats.</p>}
      {stats && <StatsPanel stats={stats} />}
      <div className={styles.actions}>
        <Link
          href="/leaderboard"
          className={`${ds.btn} ${ds.btnSm} ${styles.buttonLink}`}
        >
          Leaderboard →
        </Link>
        <SignOut onSignOut={onSignOut} />
      </div>
    </div>
  );
}

const pct = (x: number | null | undefined) =>
  x == null ? "—" : `${(x * 100).toFixed(1)}%`;
const signed = (x: number) => (x > 0 ? `+${x}` : String(x));

function StatsPanel({ stats }: { stats: AccountStats }) {
  const tiles: [string, string][] = [
    ["Hands", String(stats.hands)],
    ["Total score", signed(stats.total)],
    ["Score / hand", stats.sph == null ? "—" : stats.sph.toFixed(2)],
    ["Win rate", pct(stats.win_pct)],
    ["Pick rate", pct(stats.pick_pct)],
    ["Leasters", String(stats.leaster_hands)],
  ];
  return (
    <section aria-label="Your stats">
      <div className={`${ds.headRule} ${styles.statsHead}`}>
        <span className={ds.overline}>Your stats</span>
        <span className={ds.overline}>
          {stats.rank != null
            ? `Rank #${stats.rank} by total`
            : `${stats.qualifies_in} more hand${stats.qualifies_in === 1 ? "" : "s"} to qualify`}
        </span>
      </div>
      <dl className={styles.stats}>
        {tiles.map(([label, value]) => (
          <div key={label} className={styles.stat}>
            <dt className={ds.overline}>{label}</dt>
            <dd className={styles.statValue}>{value}</dd>
          </div>
        ))}
      </dl>
      <p className={styles.note}>
        The leaderboard lists confirmed accounts with at least {stats.min_hands}{" "}
        finished hands.
      </p>
    </section>
  );
}
