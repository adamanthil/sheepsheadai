"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import PageShell from "../../../components/PageShell";
import BetaTag from "../../components/BetaTag";
import SectionHead from "../../components/SectionHead";
import VerifiedOnly from "../../components/VerifiedOnly";
import { useRequiredAccount } from "../../useRequiredAccount";
import SuitText from "../../components/SuitText";
import accountStyles from "../../account.module.css";
import {
  AccountError,
  accountErrorMessage,
  fetchHandReview,
} from "../../../../lib/account";
import { CardText, ds } from "../../../../lib/ds";
import {
  actionCodes,
  cardLabel,
  decisionCounts,
  decisionLabel,
  decisionNote,
  handHeadline,
  historyHref,
  netTone,
  partnerLine,
  tone,
} from "../../../../lib/review";
import { signed } from "../../../../lib/format";
import { pickScenarioHref } from "../../../../lib/pickScenario";
import type { HandReview } from "../../../../lib/types";
import DecisionDetail from "./DecisionDetail";
import ExpectedScoreChart from "./ExpectedScoreChart";
import styles from "./handReview.module.css";

const WHEN = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

const ROLE = {
  picker: "Picker",
  partner: "Partner",
  defender: "Defender",
  leaster: "Leaster",
};

/** One hand, decision by decision, against the AI. */
export default function HandReviewPage() {
  const { gameId } = useParams<{ gameId: string }>();
  const { account } = useRequiredAccount();

  return (
    <PageShell wide>
      <VerifiedOnly account={account} to="review your hands">
        {/* Keyed by hand, so moving to another one starts from its own
            headline. */}
        <Review key={gameId} gameId={gameId} />
      </VerifiedOnly>
    </PageShell>
  );
}

/** The decision a link asked for (?d=<index>), if any. */
function linkedDecision(): number | null {
  const asked = new URLSearchParams(window.location.search).get("d");
  return asked !== null && /^\d+$/.test(asked) ? Number(asked) : null;
}

function Review({ gameId }: { gameId: string }) {
  const [review, setReview] = useState<HandReview | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The decision whose detail is showing; null until one is picked, when
  // the page shows the one its headline is about.
  const [picked, setPicked] = useState<number | null>(linkedDecision);
  // On a phone the detail opens in place and can be closed again.
  const [shut, setShut] = useState(false);
  // The selected decision's row, and whether the selection was just moved
  // from somewhere other than the row itself (the step buttons, the
  // chart).
  const selectedRow = useRef<HTMLButtonElement>(null);
  const moved = useRef(false);

  // On a phone the detail opens under its row, and the one that was open
  // closes, so the page shifts under the reader: a step from the foot of
  // a long detail would leave the next one mostly out of view. Bring the
  // new row to the top. Where the detail sits beside the list instead, it
  // has not moved and nothing is scrolled.
  useEffect(() => {
    if (!moved.current) return;
    moved.current = false;
    const row = selectedRow.current;
    const detail = row?.nextElementSibling;
    if (!row || !detail) return;
    const inPlace =
      detail.getBoundingClientRect().top >=
      row.getBoundingClientRect().bottom - 2;
    if (inPlace) row.scrollIntoView({ block: "start" });
  }, [picked]);

  useEffect(() => {
    let live = true;
    fetchHandReview(gameId).then(
      (data) => live && setReview(data),
      (err) =>
        live &&
        setError(
          err instanceof AccountError && err.status === 404
            ? "That hand isn't in your history."
            : accountErrorMessage(err),
        ),
    );
    return () => {
      live = false;
    };
  }, [gameId]);

  if (error) return <p className={accountStyles.error}>{error}</p>;
  if (!review) return <p className={accountStyles.note}>Loading…</p>;

  const { hand, decisions, summary } = review;
  const headline = handHeadline(review);
  // A link may name a decision the hand does not have; fall back then.
  const known = decisions.some((d) => d.index === picked) ? picked : null;
  const selected = known ?? headline.index ?? decisions[0]?.index ?? 0;
  const position = decisions.findIndex((d) => d.index === selected);
  const select = (index: number) => {
    setShut(index === selected ? !shut : false);
    setPicked(index);
  };
  const step = (by: number) => () => {
    moved.current = true;
    setPicked(decisions[position + by].index);
    setShut(false);
  };

  const roleLine =
    hand.role === "picker"
      ? hand.alone
        ? "Picker, alone"
        : review.called_card
          ? `Picker, called ${cardLabel(review.called_card)}`
          : "Picker"
      : ROLE[hand.role];
  const result = hand.score > 0 ? "Won" : hand.score < 0 ? "Lost" : "Even";
  const analyzeHref = pickScenarioHref({
    seat: review.seat,
    partnerMode: review.partner_mode,
    hand: review.dealt_hand,
    blind: review.blind,
  });
  const humans = review.seats.filter((s) => !s.is_ai);
  const ai = review.seats.length - humans.length;

  return (
    <>
      <div className={styles.topRow}>
        <Link href={historyHref("ai")} className={ds.link}>
          ← Hand history
        </Link>
        <span className={styles.neighbours}>
          {review.newer_game_id && (
            <Link
              href={`/account/history/${review.newer_game_id}`}
              className={ds.link}
            >
              ← Newer hand
            </Link>
          )}
          {review.older_game_id && (
            <Link
              href={`/account/history/${review.older_game_id}`}
              className={ds.link}
            >
              Older hand →
            </Link>
          )}
        </span>
      </div>

      <div className={`${ds.overline} ${styles.kicker}`}>
        <span>
          AI review
          <BetaTag />
        </span>
        <span>
          {WHEN.format(new Date(hand.time_closed))} · {hand.table_name}
        </span>
      </div>
      <h1 className={styles.headline}>
        <SuitText>{headline.text}</SuitText>
      </h1>
      <div className={styles.meta}>
        <span>{roleLine}</span>
        <span data-tone={hand.score < 0 ? "cost" : undefined}>
          {result} {signed(hand.score)}
        </span>
        <span>
          {hand.points_taken} {hand.points_scope} points
        </span>
        {summary && (
          <>
            <span>
              Net vs. AI{" "}
              <strong
                className={`${styles.mono} ${styles.net}`}
                data-tone={netTone(summary.net, summary.notable)}
              >
                {signed(summary.net, 1)}
              </strong>{" "}
              <span className={styles.muted}>
                (cost {summary.cost.toFixed(1)}, gained{" "}
                {summary.gained.toFixed(1)})
              </span>
            </span>
            <span>{decisionCounts(summary)}</span>
          </>
        )}
      </div>

      {summary && decisions.length > 0 && (
        <>
          <SectionHead
            className={styles.chartHead}
            title="Expected score through the hand"
            aside={`Final score ${signed(hand.score)}`}
          />
          <ExpectedScoreChart
            decisions={decisions}
            finalScore={hand.score}
            selected={selected}
            onSelect={(index) => {
              moved.current = true;
              setPicked(index);
              setShut(false);
            }}
          />

          <div
            className={styles.decisions}
            style={{ ["--rows" as string]: decisions.length }}
          >
            <SectionHead
              className={styles.listHead}
              title="Your decisions"
              aside="vs. AI"
            />
            {decisions.map((d) => {
              const on = d.index === selected;
              const open = on && !shut;
              return (
                <React.Fragment key={d.index}>
                  <button
                    type="button"
                    ref={on ? selectedRow : undefined}
                    className={`${styles.row} ${on ? styles.rowOn : ""}`}
                    aria-expanded={open}
                    onClick={() => select(d.index)}
                  >
                    <span className={styles.rowLabel}>{decisionLabel(d)}</span>
                    <span className={styles.rowAction}>
                      <CardText className={styles.rowCards}>
                        {actionCodes(d.action)}
                      </CardText>
                      <span
                        className={styles.rowNote}
                        data-tone={tone(d.standing)}
                      >
                        {decisionNote(d)}
                      </span>
                    </span>
                    <span
                      className={styles.rowDifference}
                      data-tone={tone(d.standing)}
                    >
                      {d.difference != null ? signed(d.difference, 1) : "—"}
                      <span className={styles.caret} aria-hidden="true">
                        {open ? "▴" : "▾"}
                      </span>
                    </span>
                  </button>
                  {on && (
                    <section
                      className={`${styles.detail} ${shut ? styles.detailShut : ""}`}
                      aria-label="Selected decision"
                    >
                      <DecisionDetail
                        decision={d}
                        seat={review.seat}
                        seats={review.seats}
                        blind={review.blind}
                        partner={partnerLine(review, d)}
                        analyzeHref={analyzeHref}
                        onPrevious={position > 0 ? step(-1) : null}
                        onNext={
                          position < decisions.length - 1 ? step(1) : null
                        }
                      />
                    </section>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </>
      )}

      <p className={styles.table}>
        <span className={ds.overline}>At the table</span>
        {humans.map((s) => (
          <span key={s.seat}>
            <span className={styles.tableName}>{s.name}</span>{" "}
            <span className={styles.muted}>
              {s.is_you ? "(you) · " : ""}seat {s.seat} · {ROLE[s.role]}
            </span>
          </span>
        ))}
        {ai > 0 && (
          <span className={styles.muted}>
            and {ai} AI {ai === 1 ? "player" : "players"}
          </span>
        )}
      </p>
    </>
  );
}
