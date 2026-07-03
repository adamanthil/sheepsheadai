import React from "react";
import { PlayingCard, ds } from "../../../../../lib/ds";
import type { SeatView, StageProps } from "./types";
import styles from "../Stage.module.css";

// ---------- Card / chit content for a seat ----------
export function seatCardContent(props: StageProps, seat: SeatView, w: number) {
  const h = Math.round(w * 1.45);
  const { phase } = props;
  if (phase === "play" || phase === "done") {
    const played = props.displayCards[seat.absSeat - 1] || "";
    if (played) {
      const highlight = props.showPrev && props.winnerSeat === seat.absSeat;
      return (
        <PlayingCard
          code={played}
          w={w}
          className={highlight ? styles.winnerCard : undefined}
        />
      );
    }
    return <div className={styles.emptyCard} style={{ width: w, height: h }} />;
  }
  // pick / interlude → chits, driven by the seat's role so seats that
  // haven't acted yet don't falsely read "passed".
  let text: string;
  if (phase === "pick") {
    text =
      seat.role === "PASS"
        ? "passed"
        : seat.role === "PENDING"
          ? "deciding"
          : "waiting";
  } else {
    text = "waiting";
  }
  return (
    <div className={styles.chit} style={{ width: w, height: h }}>
      <span className={styles.chitText} style={{ fontSize: w > 90 ? 11 : 9 }}>
        {text}
      </span>
    </div>
  );
}

// ---------- Center content router ----------
export function CenterContent({
  props,
  mobile,
}: {
  props: StageProps;
  mobile?: boolean;
}) {
  const { phase, isYourTurn, yourMode, handLen, trickIndex, totalTricks } =
    props;
  const scale = mobile ? 1 : (props.uiScale ?? 1);

  if (phase === "pick") {
    // Your decision → the blind itself is the Pick button, with a Pass card
    // beside it (mirrors the call-decision layout). Otherwise the blind shows
    // as non-interactive context while others decide.
    const decide = props.pickActionId != null || props.passActionId != null;
    if (decide) {
      const w = mobile ? 60 : Math.round(96 * scale);
      const h = Math.round(w * 1.45);
      return (
        <>
          {!mobile && (
            <div
              className={ds.overline}
              style={{ fontSize: 10, marginBottom: 12 }}
            >
              Pick or pass
            </div>
          )}
          <div className={styles.callRow}>
            {props.pickActionId != null && (
              <button
                className={styles.callOption}
                onClick={() => props.onAction(props.pickActionId!)}
                style={{ pointerEvents: "auto" }}
              >
                <div
                  className={styles.blindStack}
                  style={{ width: Math.round(w * 1.55), height: h }}
                >
                  <div
                    className={styles.blindCard}
                    style={{ left: 0, transform: "rotate(-4deg)" }}
                  >
                    <PlayingCard code="__" w={w} />
                  </div>
                  <div
                    className={styles.blindCard}
                    style={{
                      left: Math.round(w * 0.55),
                      transform: "rotate(3deg)",
                    }}
                  >
                    <PlayingCard code="__" w={w} />
                  </div>
                </div>
                <div className={styles.callLabel}>Pick the blind</div>
              </button>
            )}
            {props.passActionId != null && (
              <button
                className={styles.callOption}
                onClick={() => props.onAction(props.passActionId!)}
                style={{ pointerEvents: "auto" }}
              >
                <div
                  className={styles.alonePanel}
                  style={{ width: w, height: h }}
                >
                  <div
                    className={styles.aloneTitle}
                    style={{ fontSize: Math.round(w * 0.28) }}
                  >
                    Pass
                  </div>
                </div>
                <div className={styles.callLabel}>Pass the buck</div>
              </button>
            )}
          </div>
          {mobile && (
            <div className={ds.overline} style={{ fontSize: 8, marginTop: 8 }}>
              Pick or pass
            </div>
          )}
        </>
      );
    }
    const w = mobile ? 56 : Math.round(104 * scale);
    return (
      <>
        <div
          className={styles.blindStack}
          style={{ width: w * 2 + 8, height: Math.round(w * 1.45) }}
        >
          <div
            className={styles.blindCard}
            style={{ left: 0, transform: "rotate(-4deg)" }}
          >
            <PlayingCard code="__" w={w} />
          </div>
          <div
            className={styles.blindCard}
            style={{ left: w, transform: "rotate(3deg)" }}
          >
            <PlayingCard code="__" w={w} />
          </div>
        </div>
        <div
          className={ds.overline}
          style={{ fontSize: mobile ? 8 : 10, marginTop: mobile ? 6 : 12 }}
        >
          The blind
        </div>
        {!mobile && <div className={styles.centerSub}>two cards face-down</div>}
      </>
    );
  }

  if (phase === "interlude" && yourMode === "bury") {
    const chosen = Math.max(0, 8 - handLen);
    const w = mobile ? 48 : Math.round(104 * scale);
    return (
      <>
        <div className={styles.slotRow} style={{ gap: mobile ? 6 : 14 }}>
          {[0, 1].map((i) =>
            i < chosen ? (
              <PlayingCard key={i} code="__" w={w} />
            ) : (
              <div
                key={i}
                className={styles.slot}
                style={{ width: w, height: Math.round(w * 1.45) }}
              >
                <span
                  className={styles.slotLabel}
                  style={{ fontSize: mobile ? 8 : 10 }}
                >
                  slot {i + 1}
                </span>
              </div>
            ),
          )}
        </div>
        <div
          className={ds.overline}
          style={{ fontSize: mobile ? 8 : 10, marginTop: mobile ? 6 : 12 }}
        >
          Burying
        </div>
        {!mobile && (
          <div className={styles.centerSub}>
            {chosen} of 2 chosen · tap a hand card to bury
          </div>
        )}
      </>
    );
  }

  if (phase === "interlude" && yourMode === "call") {
    const w = mobile ? 60 : Math.round(96 * scale);
    return (
      <>
        {!mobile && (
          <div
            className={ds.overline}
            style={{ fontSize: 10, marginBottom: 12 }}
          >
            Choose your partner
          </div>
        )}
        <div className={styles.callRow}>
          {props.callOptions.map((o) => {
            const sel = props.selectedCall === o.label;
            return (
              <button
                key={o.actionId}
                className={styles.callOption}
                onClick={() => props.onAction(o.actionId)}
                style={{ pointerEvents: "auto" }}
              >
                {o.code ? (
                  <PlayingCard code={o.code} w={w} playable={sel} />
                ) : (
                  <div
                    className={styles.alonePanel}
                    style={{
                      width: w,
                      height: Math.round(w * 1.45),
                      ...(sel
                        ? {
                            boxShadow:
                              "0 0 0 2px var(--accent-2), var(--shadow-2)",
                            transform: "translateY(-6px)",
                            borderColor: "var(--accent-2)",
                          }
                        : {}),
                    }}
                  >
                    <div
                      className={styles.aloneTitle}
                      style={{ fontSize: Math.round(w * 0.28) }}
                    >
                      Alone
                    </div>
                  </div>
                )}
                <div
                  className={`${styles.callLabel} ${sel ? styles.callLabelSel : ""}`}
                >
                  {o.display}
                </div>
              </button>
            );
          })}
        </div>
        {mobile && (
          <div className={ds.overline} style={{ fontSize: 8, marginTop: 8 }}>
            Call partner
          </div>
        )}
      </>
    );
  }

  if (phase === "interlude") {
    return <div className={styles.centerSub}>Setting up the hand…</div>;
  }

  // play / done
  const label = props.isLeaster
    ? "Leaster"
    : `Trick ${Math.min(trickIndex + 1, totalTricks)} of ${totalTricks}`;
  return (
    <>
      <div className={ds.overline} style={{ fontSize: mobile ? 8 : 10 }}>
        {label}
      </div>
      {isYourTurn && (
        <div className={styles.turnPill} style={{ marginTop: mobile ? 4 : 8 }}>
          <span className={styles.turnDot} />
          <span
            className={styles.turnText}
            style={{ fontSize: mobile ? 9 : 10 }}
          >
            your turn
          </span>
        </div>
      )}
    </>
  );
}
