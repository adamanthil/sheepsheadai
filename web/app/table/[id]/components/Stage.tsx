import React from "react";
import { PlayingCard, SeatAvatar, ds } from "../../../../lib/ds";
import { RING_ANCHORS, MOBILE_RING_ANCHORS } from "../lib/seatLayout";
import type { TablePhase, SeatRole, InterludeMode } from "../lib/phase";
import CollectOverlay from "./CollectOverlay";
import type { AnimTrick } from "../hooks/useTrickAnimation";
import styles from "./Stage.module.css";

export interface SeatView {
  absSeat: number;
  rel: number;
  name: string;
  isAI: boolean;
  role: SeatRole;
  you: boolean;
}

export interface CallOption {
  actionId: number;
  label: string; // raw action label, e.g. "CALL AC" / "ALONE"
  code: string | null; // card code for ace options, null for ALONE/JD
  display: string; // human label
}

interface StageProps {
  seats: SeatView[];
  yourSeat: number;
  phase: TablePhase;
  isLeaster: boolean;
  yourMode: InterludeMode;
  isYourTurn: boolean;
  handLen: number;
  trickIndex: number;
  totalTricks: number;
  displayCards: string[]; // current_trick, or last_trick when showing prev
  winnerSeat: number | null;
  showPrev: boolean;
  prevText: string | null;
  callOptions: CallOption[];
  selectedCall: string | null;
  onAction: (actionId: number) => void;
  isMobile: boolean;
  trickBoxRef: React.RefObject<HTMLDivElement | null>;
  animTrick: AnimTrick | null;
  callout: string | null;
}

function roleBadge(role: SeatRole, small?: boolean) {
  const fs = small ? { fontSize: 8, padding: "1px 5px" } : undefined;
  if (role === "PICKER")
    return (
      <span className={`${ds.badge} ${ds.badgeAccent}`} style={fs}>
        Picker
      </span>
    );
  if (role === "PARTNER")
    return (
      <span className={`${ds.badge} ${ds.badgeGold}`} style={fs}>
        Partner
      </span>
    );
  if (role === "PASS")
    return (
      <span className={`${ds.badge} ${ds.badgeQuiet}`} style={fs}>
        Pass
      </span>
    );
  if (role === "PENDING")
    return (
      <span
        className={`${ds.badge} ${ds.badgeQuiet}`}
        style={{ ...fs, animation: "ssPulse 1.4s ease-in-out infinite" }}
      >
        Deciding
      </span>
    );
  return null;
}

function tone(role: SeatRole): "default" | "picker" | "partner" {
  return role === "PICKER"
    ? "picker"
    : role === "PARTNER"
      ? "partner"
      : "default";
}

export default function Stage(props: StageProps) {
  return props.isMobile ? (
    <MobileStage {...props} />
  ) : (
    <DesktopStage {...props} />
  );
}

// ---------- Card / chit content for a seat ----------
function seatCardContent(props: StageProps, seat: SeatView, w: number) {
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
function CenterContent({
  props,
  mobile,
}: {
  props: StageProps;
  mobile?: boolean;
}) {
  const { phase, isYourTurn, yourMode, handLen, trickIndex, totalTricks } =
    props;

  if (phase === "pick") {
    const w = mobile ? 56 : 104;
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
    const w = mobile ? 48 : 104;
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
    const w = mobile ? 60 : 96;
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

// ---------- Desktop ----------
function DesktopStage(props: StageProps) {
  const { seats, yourSeat } = props;
  const you = seats.find((s) => s.you);
  const youPlayed =
    (props.phase === "play" || props.phase === "done") && you
      ? props.displayCards[you.absSeat - 1] || ""
      : "";

  return (
    <div
      className={styles.deskStage}
      ref={props.trickBoxRef as React.RefObject<HTMLDivElement>}
    >
      <div className={styles.deskInner}>
        <svg
          className={styles.ellipse}
          viewBox="0 0 560 480"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <ellipse
            cx="280"
            cy="260"
            rx="260"
            ry="200"
            fill="none"
            stroke="var(--rule)"
            strokeDasharray="2 5"
            strokeWidth="1"
          />
        </svg>

        <div className={styles.center}>
          <CenterContent props={props} />
        </div>

        {seats
          .filter((s) => !s.you)
          .map((seat) => {
            const anchor = RING_ANCHORS[seat.rel];
            if (!anchor) return null;
            return (
              <div
                key={seat.absSeat}
                className={styles.ringSeat}
                style={{ left: `${anchor.cardX}%`, top: `${anchor.cardY}%` }}
              >
                <RingChip seat={seat} plate={anchor.plate} />
                <div>{seatCardContent(props, seat, 104)}</div>
              </div>
            );
          })}

        {youPlayed && (
          <div
            className={styles.ringSeat}
            style={{
              left: `${RING_ANCHORS[0].cardX}%`,
              top: `${RING_ANCHORS[0].cardY}%`,
            }}
          >
            <PlayingCard code={youPlayed} w={104} />
            <span className={styles.youPlate}>You</span>
          </div>
        )}

        {props.showPrev && props.prevText && !props.animTrick && (
          <div className={styles.prevBanner}>{props.prevText}</div>
        )}
        {props.animTrick && (
          <CollectOverlay
            containerRef={props.trickBoxRef}
            yourSeat={yourSeat}
            winner={props.animTrick.winner}
            cards={props.animTrick.cards}
            cardW={104}
          />
        )}
        {props.callout && <div className={styles.callout}>{props.callout}</div>}
      </div>
    </div>
  );
}

// Compact name-plate that floats off a seat's played card, toward the table
// rim: mid seats sit above their card, top seats below. `compact` trims it for
// the tighter mobile ring (smaller avatar, no seat number).
function RingChip({
  seat,
  plate,
  compact,
}: {
  seat: SeatView;
  plate: "above" | "below";
  compact?: boolean;
}) {
  const plateClass = plate === "above" ? styles.chipAbove : styles.chipBelow;
  return (
    <div className={`${styles.chip} ${plateClass}`}>
      <SeatAvatar
        name={seat.name}
        isAI={seat.isAI}
        tone={tone(seat.role)}
        size={compact ? 26 : 32}
      />
      <div className={styles.chipText}>
        <div
          className={`${styles.chipName} ${compact ? styles.chipNameSm : ""}`}
        >
          {seat.name}
        </div>
        {!compact && (
          <div className={ds.overline} style={{ fontSize: 9 }}>
            Seat {seat.absSeat}
          </div>
        )}
      </div>
      {roleBadge(seat.role, compact) && (
        <div className={styles.chipBadges}>{roleBadge(seat.role, compact)}</div>
      )}
    </div>
  );
}

// ---------- Mobile ----------
function MobileStage(props: StageProps) {
  const { seats } = props;
  const MOB_CARD = 66;
  const you = seats.find((s) => s.you);
  const youPlayed =
    (props.phase === "play" || props.phase === "done") && you
      ? props.displayCards[you.absSeat - 1] || ""
      : "";
  return (
    <div
      className={styles.mobStage}
      ref={props.trickBoxRef as React.RefObject<HTMLDivElement>}
    >
      <svg
        className={styles.mobEllipse}
        preserveAspectRatio="none"
        viewBox="0 0 100 100"
        aria-hidden="true"
      >
        <ellipse
          cx="50"
          cy="50"
          rx="48"
          ry="46"
          fill="none"
          stroke="var(--rule)"
          strokeDasharray="0.6 1.4"
          strokeWidth="0.5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>

      <div className={styles.mobCenter}>
        <CenterContent props={props} mobile />
      </div>

      {seats
        .filter((s) => !s.you)
        .map((seat) => {
          const anchor = MOBILE_RING_ANCHORS[seat.rel];
          if (!anchor) return null;
          return (
            <div
              key={seat.absSeat}
              className={styles.ringSeat}
              style={{ left: `${anchor.cardX}%`, top: `${anchor.cardY}%` }}
            >
              <RingChip seat={seat} plate={anchor.plate} compact />
              <div>{seatCardContent(props, seat, MOB_CARD)}</div>
            </div>
          );
        })}

      {youPlayed && (
        <div
          className={styles.ringSeat}
          style={{
            left: `${MOBILE_RING_ANCHORS[0].cardX}%`,
            top: `${MOBILE_RING_ANCHORS[0].cardY}%`,
          }}
        >
          <PlayingCard code={youPlayed} w={MOB_CARD} />
          <span className={styles.youPlate}>You</span>
        </div>
      )}

      {props.callout && <div className={styles.callout}>{props.callout}</div>}
    </div>
  );
}
