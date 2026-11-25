"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import AccountShell from "../AccountShell";
import VerifiedOnly from "../components/VerifiedOnly";
import { useLinkedView } from "../useLinkedView";
import { useRequiredAccount } from "../useRequiredAccount";
import accountStyles from "../account.module.css";
import { accountErrorMessage, fetchAccountReview } from "../../../lib/account";
import { PlayingCard, ds } from "../../../lib/ds";
import {
  decisionLabel,
  differenceLines,
  tone,
  totalTone,
} from "../../../lib/review";
import { shortDate, signed } from "../../../lib/format";
import {
  REVIEW_VIEWS,
  handReviewHref,
  reviewHref,
} from "../../../lib/reviewLinks";
import type {
  AccountReview,
  ReviewDifference,
  ReviewPick,
} from "../../../lib/types";
import { BigFigure, BigFigures } from "../components/BigFigures";
import SectionHead from "../components/SectionHead";
import Segmented from "../components/Segmented";
import SuitText from "../components/SuitText";
import styles from "./review.module.css";

const KIND_LABEL = {
  pick: "Pick or pass",
  call: "Partner call",
  bury: "Bury",
  play: "Card play",
};

type KindFilter = "all" | "pick" | "call" | "bury" | "play";
const FILTERS: { value: KindFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "pick", label: "Pick" },
  { value: "call", label: "Call" },
  { value: "bury", label: "Bury" },
  { value: "play", label: "Play" },
];

// Differences listed before "Show more".
const FIRST_SHOWN = 8;

function percent(part: number, whole: number): string {
  return whole ? `${Math.round((part / whole) * 100)}%` : "—";
}

/** Every reviewed hand, totalled: how often the player made the AI's
 * choice, where they differ most, and their picking. */
export default function AccountReviewPage() {
  const { account } = useRequiredAccount();

  return (
    <AccountShell tab="review" account={account} wide>
      <VerifiedOnly account={account} to="see your AI review">
        <Totals />
      </VerifiedOnly>
    </AccountShell>
  );
}

function Totals() {
  const [review, setReview] = useState<AccountReview | null>(null);
  const [view, show] = useLinkedView(REVIEW_VIEWS, reviewHref);
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
  const toggle = review.pick_decisions > 0 && (
    <Segmented
      size="large"
      className={styles.viewToggle}
      label="Review of"
      options={[
        { value: "decisions", label: "Overview" },
        { value: "picking", label: "Picking" },
      ]}
      value={view}
      onChange={show}
    />
  );
  if (view === "picking" && review.pick_decisions > 0) {
    return (
      <>
        {lede}
        {toggle}
        <Picking review={review} />
      </>
    );
  }
  return (
    <>
      {lede}
      {toggle}

      <section aria-label="You and the AI">
        <SectionHead
          title="You and the AI"
          aside={
            <>
              {review.hands_reviewed} hands reviewed
              {review.hands_waiting > 0 && ` · ${review.hands_waiting} waiting`}
            </>
          }
        />
        <BigFigures>
          <BigFigure
            label="Agreed"
            note={`of ${review.decisions} decisions with a choice`}
          >
            {percent(review.agreed, review.decisions)}
          </BigFigure>
          <BigFigure
            label="AI chose better"
            note={`${review.costly === 1 ? "time" : "times"}, by 1+ score`}
          >
            {review.costly}
          </BigFigure>
          <BigFigure
            label="You chose better"
            note={`${review.better === 1 ? "time" : "times"}, by 1+ score`}
          >
            {review.better}
          </BigFigure>
          <BigFigure
            label="Net vs. AI"
            note={`${signed(perHand, 2)} score per hand`}
            tone={totalTone(review.net)}
          >
            {signed(review.net, 1)}
          </BigFigure>
        </BigFigures>

        <h2 className={accountStyles.subHead}>Agreed, by kind of decision</h2>
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
              <span className={styles.mono}>{signed(kind.net, 1)}</span>
            </li>
          ))}
        </ul>
        <p className={accountStyles.footnote}>
          The last column is your net expected score against the AI’s choices
          for that kind of decision.
        </p>
      </section>

      <Differences review={review} />
    </>
  );
}

function Differences({ review }: { review: AccountReview }) {
  const [which, setWhich] = useState<"cost" | "gained">("cost");
  const [filter, setFilter] = useState<KindFilter>("all");
  const [all, setAll] = useState(false);
  // The server keeps a list for each kind, so narrowing to one shows that
  // kind's largest rather than what is left of everything's.
  const shown = (which === "cost" ? review.cost_most : review.gained_most)[
    filter
  ];
  // The two counts follow the kind chosen, as the list under them does.
  const ofKind = review.by_kind.find((kind) => kind.kind === filter);
  const costly = filter === "all" ? review.costly : (ofKind?.costly ?? 0);
  const better = filter === "all" ? review.better : (ofKind?.better ?? 0);
  // A long history has more differences than the server lists.
  const cut = (which === "cost" ? costly : better) > shown.length;

  return (
    <section aria-label="Largest differences" className={styles.section}>
      <SectionHead
        title="Largest differences"
        aside="Points of expected score"
      />
      <div className={styles.controls}>
        <Segmented
          label="Which way"
          options={[
            { value: "cost", label: `Cost you most · ${costly}` },
            { value: "gained", label: `Beat the AI’s choice · ${better}` },
          ]}
          value={which}
          onChange={setWhich}
        />
        <Segmented
          label="Kind of decision"
          options={FILTERS}
          value={filter}
          onChange={setFilter}
        />
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
      <p className={accountStyles.footnote}>
        A difference is the AI’s expected score after your choice, less its
        expected score after its own, in game points. Each is played forward to
        your next decision with the cards as dealt; a pick is averaged over
        every possible blind. Only differences of a full point or more are
        listed: {review.slight} smaller ones and {review.even} that were about
        even are left off.
        {cut && ` The list stops at the ${shown.length} largest.`}
      </p>
    </section>
  );
}

function DifferenceRow({ difference }: { difference: ReviewDifference }) {
  const lines = differenceLines(difference);
  return (
    <li>
      <Link
        href={handReviewHref(difference.game_id, difference.index)}
        className={styles.difference}
        aria-label={`${lines.yours}, ${signed(difference.difference, 1)}. Review this decision`}
      >
        <span
          className={styles.differenceValue}
          data-tone={tone(difference.standing)}
        >
          {signed(difference.difference, 1)}
        </span>
        <span className={styles.differenceText}>
          <span className={styles.differenceYours}>
            <SuitText>{lines.yours}</SuitText>
          </span>
          <span className={styles.differenceAi}>
            <SuitText>{lines.ai}</SuitText>,{" "}
            {Math.round(difference.ai_probability * 100)}% ·{" "}
            <span className={styles.mono}>
              {shortDate(difference.time_closed)} · {difference.table_name}
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
  const { picked_ai_passes: picked, passed_ai_picks: passed } = review;
  const shown = which === "picked" ? picked : passed;

  return (
    <section aria-label="Picking">
      <SectionHead
        title="Picking"
        aside={`${review.pick_decisions} pick decisions`}
      />
      <BigFigures>
        <BigFigure label="Agreed">
          {percent(review.pick_agreed, review.pick_decisions)}
        </BigFigure>
        <BigFigure label="You picked, it passes">{picked.count}</BigFigure>
        <BigFigure label="You passed, it picks">{passed.count}</BigFigure>
        <BigFigure
          label="Score on those picks"
          tone={picked.score < 0 ? "cost" : "gain"}
        >
          {signed(picked.score)}
        </BigFigure>
      </BigFigures>
      {picked.count + passed.count > 0 && (
        <>
          <div className={`${styles.controls} ${styles.underFigures}`}>
            <Segmented
              label="Which picks"
              options={[
                {
                  value: "picked",
                  label: `Picked, AI passes · ${picked.count}`,
                },
                {
                  value: "passed",
                  label: `Passed, AI picks · ${passed.count}`,
                },
              ]}
              value={which}
              onChange={setWhich}
            />
          </div>
          <ul className={styles.picks}>
            {shown.picks.map((pick) => (
              <PickCard key={pick.game_id} pick={pick} />
            ))}
          </ul>
          {shown.count > shown.picks.length && (
            <p className={accountStyles.footnote}>
              The {shown.picks.length} most recent are shown.
            </p>
          )}
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
          {shortDate(pick.time_closed)} · {pick.table_name}
        </span>
        <span
          className={styles.mono}
          data-tone={pick.score < 0 ? "cost" : "gain"}
        >
          {signed(pick.score)}
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
            pick {signed(pick.pick_score, 1)} · pass{" "}
            {signed(pick.pass_score, 1)}
          </strong>
        </span>
        <Link href={handReviewHref(pick.game_id, 0)} className={ds.link}>
          Review hand →
        </Link>
      </div>
    </li>
  );
}
