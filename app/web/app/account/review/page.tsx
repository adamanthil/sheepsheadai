"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import AccountShell from "../AccountShell";
import accountStyles from "../account.module.css";
import {
  accountErrorMessage,
  fetchAccountReview,
  useAccount,
} from "../../../lib/account";
import { PlayingCard, ds } from "../../../lib/ds";
import {
  decisionLabel,
  differenceLines,
  signed,
  tone,
} from "../../../lib/review";
import type {
  AccountReview,
  ReviewDifference,
  ReviewPick,
} from "../../../lib/types";
import SuitText from "../components/SuitText";
import styles from "./review.module.css";

const DAY = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
});

const KIND_LABEL = {
  pick: "Pick or pass",
  call: "Partner call",
  bury: "Bury",
  play: "Card play",
};

type KindFilter = "all" | "pick" | "call" | "bury" | "play";
const FILTERS: { key: KindFilter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "pick", label: "Pick" },
  { key: "call", label: "Call" },
  { key: "bury", label: "Bury" },
  { key: "play", label: "Play" },
];

// Differences listed before "Show more".
const FIRST_SHOWN = 8;

function percent(part: number, whole: number): string {
  return whole ? `${Math.round((part / whole) * 100)}%` : "—";
}

function toneClass(value: number): string {
  const kind = tone(value);
  return kind === "cost" ? styles.cost : kind === "gain" ? styles.gain : "";
}

/** Every reviewed hand, totalled: how often the player made the AI's
 * choice, where they differ most, and their picking. */
export default function AccountReviewPage() {
  const router = useRouter();
  const { me } = useAccount();
  const account = me?.account ?? null;

  useEffect(() => {
    if (me !== undefined && !account) router.replace("/account");
  }, [me, account, router]);

  let body: React.ReactNode;
  if (!account) {
    body = <p className={accountStyles.note}>Loading…</p>;
  } else if (!account.email_verified) {
    body = (
      <p className={accountStyles.note}>
        Confirm your email to see your AI review.{" "}
        <Link href="/account" className={ds.link}>
          Back to your account
        </Link>
      </p>
    );
  } else {
    body = <Totals />;
  }

  return (
    <AccountShell
      title={account ? `@${account.username}` : "Your account"}
      tab="review"
      account={account}
      wide
    >
      {body}
    </AccountShell>
  );
}

function Totals() {
  const [review, setReview] = useState<AccountReview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAccountReview().then(setReview, (err) =>
      setError(accountErrorMessage(err)),
    );
  }, []);

  if (error) return <p className={accountStyles.error}>{error}</p>;
  if (!review) return <p className={accountStyles.note}>Loading…</p>;

  const lede = (
    <p className={styles.lede}>
      Your decisions, held up against what the AI model would have done in the
      same seat with the same cards. Every hand is reviewed once it ends.
    </p>
  );
  if (review.hands_reviewed === 0) {
    return (
      <>
        {lede}
        <p className={accountStyles.note}>
          {review.hands_waiting > 0
            ? `${review.hands_waiting} of your hands are waiting for review. They are worked through in the background, so check back soon.`
            : "No reviewed hands yet. Play a hand and it will show up here."}
        </p>
      </>
    );
  }

  const perHand = review.net / review.hands_reviewed;
  return (
    <>
      {lede}

      <section aria-label="You and the AI">
        <div className={styles.sectionHead}>
          <span>You and the AI</span>
          <span>
            {review.hands_reviewed} hands reviewed
            {review.hands_waiting > 0 && ` · ${review.hands_waiting} waiting`}
          </span>
        </div>
        <dl className={styles.figures}>
          <div>
            <dt>Agreed</dt>
            <dd>{percent(review.agreed, review.decisions)}</dd>
            <dd className={styles.figureNote}>
              of {review.decisions} decisions with a choice
            </dd>
          </div>
          <div>
            <dt>Cost you</dt>
            <dd>{review.costly}</dd>
            <dd className={styles.figureNote}>a point or more</dd>
          </div>
          <div>
            <dt>Beat its choice</dt>
            <dd>{review.better}</dd>
            <dd className={styles.figureNote}>a point or more</dd>
          </div>
          <div>
            <dt>Net vs. AI</dt>
            <dd className={toneClass(review.net)}>{signed(review.net)}</dd>
            <dd className={styles.figureNote}>{signed(perHand, 2)} a hand</dd>
          </div>
        </dl>

        <h2 className={styles.subHead}>Agreed, by kind of decision</h2>
        <ul className={styles.kinds}>
          {review.by_kind.map((kind) => (
            <li key={kind.kind}>
              <span>{KIND_LABEL[kind.kind]}</span>
              <span className={styles.kindTrack} aria-hidden="true">
                <span
                  style={{
                    width: `${(kind.agreed / kind.decisions) * 100}%`,
                  }}
                />
              </span>
              <span className={styles.mono}>
                {percent(kind.agreed, kind.decisions)}
              </span>
              <span className={styles.kindCount}>
                {kind.decisions} decisions
              </span>
              <span className={styles.mono}>{signed(kind.net)}</span>
            </li>
          ))}
        </ul>
        <p className={styles.footnote}>
          The last column is your net expected score against the AI’s choices
          for that kind of decision.
        </p>
      </section>

      <Differences review={review} />
      <Picking review={review} />
    </>
  );
}

function Differences({ review }: { review: AccountReview }) {
  const [which, setWhich] = useState<"cost" | "gained">("cost");
  const [filter, setFilter] = useState<KindFilter>("all");
  const [all, setAll] = useState(false);
  const list = which === "cost" ? review.cost_most : review.gained_most;
  // The two counts follow the kind chosen, as the list under them does.
  const ofKind = review.by_kind.find((kind) => kind.kind === filter);
  const costly = filter === "all" ? review.costly : (ofKind?.costly ?? 0);
  const better = filter === "all" ? review.better : (ofKind?.better ?? 0);
  const shown = list.filter(
    (d) =>
      filter === "all" || (d.kind === "under" ? "call" : d.kind) === filter,
  );

  return (
    <section aria-label="Largest differences" className={styles.section}>
      <div className={styles.sectionHead}>
        <span>Largest differences</span>
        <span>Points of expected score</span>
      </div>
      <div className={styles.controls}>
        <div className={styles.segmented} role="group" aria-label="Which way">
          <button
            type="button"
            aria-pressed={which === "cost"}
            onClick={() => setWhich("cost")}
          >
            Cost you most · {costly}
          </button>
          <button
            type="button"
            aria-pressed={which === "gained"}
            onClick={() => setWhich("gained")}
          >
            Beat the AI’s choice · {better}
          </button>
        </div>
        <div
          className={styles.segmented}
          role="group"
          aria-label="Kind of decision"
        >
          {FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              aria-pressed={filter === f.key}
              onClick={() => setFilter(f.key)}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>
      {shown.length === 0 ? (
        <p className={accountStyles.note}>Nothing here yet.</p>
      ) : (
        <ol className={styles.differences}>
          {(all ? shown : shown.slice(0, FIRST_SHOWN)).map((d) => (
            <DifferenceRow key={`${d.game_id}-${d.index}`} difference={d} />
          ))}
        </ol>
      )}
      {!all && shown.length > FIRST_SHOWN && (
        <button
          type="button"
          className={`${ds.btn} ${ds.btnSm} ${ds.btnGhost} ${styles.more}`}
          onClick={() => setAll(true)}
        >
          Show {shown.length - FIRST_SHOWN} more
        </button>
      )}
      <p className={styles.footnote}>
        A difference is the AI’s expected score after your choice, less its
        expected score after its own, in game points. Each is played forward to
        your next decision with the cards as dealt; a pick is averaged over
        every possible blind. Only differences of a full point or more are
        listed: {review.slight} smaller ones and {review.even} that were about
        even are left off.
      </p>
    </section>
  );
}

function DifferenceRow({ difference }: { difference: ReviewDifference }) {
  const lines = differenceLines(difference);
  return (
    <li>
      <Link
        href={`/account/history/${difference.game_id}?d=${difference.index}`}
        className={styles.difference}
        aria-label={`${lines.yours}, ${signed(difference.difference)}. Review this decision`}
      >
        <span
          className={`${styles.differenceValue} ${toneClass(difference.difference)}`}
        >
          {signed(difference.difference)}
        </span>
        <span className={styles.differenceText}>
          <span className={styles.differenceYours}>
            <SuitText>{lines.yours}</SuitText>
          </span>
          <span className={styles.differenceAi}>
            <SuitText>{lines.ai}</SuitText>,{" "}
            {Math.round(difference.ai_probability * 100)}% ·{" "}
            <span className={styles.mono}>
              {DAY.format(new Date(difference.time_closed))} ·{" "}
              {difference.table_name}
            </span>
          </span>
        </span>
        <span className={`${ds.badge} ${ds.badgeQuiet}`}>
          {decisionLabel(difference)}
        </span>
        <span aria-hidden="true" className={styles.arrow}>
          →
        </span>
      </Link>
    </li>
  );
}

function Picking({ review }: { review: AccountReview }) {
  const [which, setWhich] = useState<"picked" | "passed">("picked");
  if (review.pick_decisions === 0) return null;
  const list =
    which === "picked" ? review.picked_ai_passes : review.passed_ai_picks;
  const picks = [...review.picked_ai_passes, ...review.passed_ai_picks];
  const scored = review.picked_ai_passes.reduce((sum, p) => sum + p.score, 0);

  return (
    <section aria-label="Picking" className={styles.section}>
      <div className={styles.sectionHead}>
        <span>Picking</span>
        <span>{review.pick_decisions} pick decisions</span>
      </div>
      <dl className={styles.figures}>
        <div>
          <dt>Agreed</dt>
          <dd>{percent(review.pick_agreed, review.pick_decisions)}</dd>
        </div>
        <div>
          <dt>You picked, it passes</dt>
          <dd>{review.picked_ai_passes.length}</dd>
        </div>
        <div>
          <dt>You passed, it picks</dt>
          <dd>{review.passed_ai_picks.length}</dd>
        </div>
        <div>
          <dt>Score on those picks</dt>
          <dd className={scored < 0 ? styles.cost : styles.gain}>
            {signed(scored, 0)}
          </dd>
        </div>
      </dl>
      {picks.length > 0 && (
        <>
          <div className={styles.controls}>
            <div
              className={styles.segmented}
              role="group"
              aria-label="Which picks"
            >
              <button
                type="button"
                aria-pressed={which === "picked"}
                onClick={() => setWhich("picked")}
              >
                Picked, AI passes · {review.picked_ai_passes.length}
              </button>
              <button
                type="button"
                aria-pressed={which === "passed"}
                onClick={() => setWhich("passed")}
              >
                Passed, AI picks · {review.passed_ai_picks.length}
              </button>
            </div>
          </div>
          <ul className={styles.picks}>
            {list.map((pick) => (
              <PickCard key={pick.game_id} pick={pick} />
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function PickCard({ pick }: { pick: ReviewPick }) {
  return (
    <li className={styles.pick}>
      <div className={styles.pickHead}>
        <span className={styles.mono}>
          {DAY.format(new Date(pick.time_closed))} · {pick.table_name}
        </span>
        <span
          className={`${styles.mono} ${pick.score < 0 ? styles.cost : styles.gain}`}
        >
          {signed(pick.score, 0)}
        </span>
      </div>
      <div className={styles.pickCards}>
        {pick.hand.map((card) => (
          <PlayingCard key={card} code={card} largeIndex />
        ))}
      </div>
      <div className={styles.pickChance}>
        <span>Seat {pick.seat} · AI picks</span>
        <span className={styles.mono}>
          {Math.round(pick.ai_pick_probability * 100)}%
        </span>
      </div>
      <span className={styles.kindTrack} aria-hidden="true">
        <span style={{ width: `${pick.ai_pick_probability * 100}%` }} />
      </span>
      <div className={styles.pickFoot}>
        <span>
          Expected score{" "}
          <strong className={styles.mono}>
            pick {signed(pick.pick_score)} · pass {signed(pick.pass_score)}
          </strong>
        </span>
        <Link href={`/account/history/${pick.game_id}?d=0`} className={ds.link}>
          Review hand →
        </Link>
      </div>
    </li>
  );
}
