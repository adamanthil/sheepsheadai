import React from "react";
import { PlayingCard, ds } from "../../../../../lib/ds";
import type { SeatView, StageProps } from "./types";
import styles from "../Stage.module.css";

// ---------- Card / chit content for a seat ----------
// `w` is the card width in px; omit it (mobile) to let the CSS-driven --pc-w
// size the card, chit and empty slot alike.
export function seatCardContent(props: StageProps, seat: SeatView, w?: number) {
  const sizeVars =
    w != null
      ? ({ ["--pc-w" as string]: `${w}px` } as React.CSSProperties)
      : undefined;
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
    return <div className={styles.emptyCard} style={sizeVars} />;
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
    <div className={styles.chit} style={sizeVars}>
      <span className={styles.chitText}>{text}</span>
    </div>
  );
}

// ---------- Staging pieces shared by the bury and under branches ----------
function EmptySlot({
  label,
  w,
  mobile,
}: {
  label: string;
  w: number;
  mobile?: boolean;
}) {
  return (
    <div
      className={styles.slot}
      style={{ width: w, height: Math.round(w * 1.45) }}
    >
      <span className={styles.slotLabel} style={{ fontSize: mobile ? 8 : 10 }}>
        {label}
      </span>
    </div>
  );
}

function PutBackButton({
  card,
  w,
  onClick,
}: {
  card: string;
  w: number;
  onClick: () => void;
}) {
  return (
    <button
      className={styles.callOption}
      style={{ pointerEvents: "auto" }}
      onClick={onClick}
      aria-label={`Put ${card} back in your hand`}
    >
      <PlayingCard code={card} w={w} playable />
    </button>
  );
}

function ConfirmButton({
  label,
  busyLabel,
  busy,
  mobile,
  onClick,
}: {
  label: string;
  busyLabel: string;
  busy: boolean;
  mobile?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className={`${ds.btn} ${ds.btnAccent}`}
      style={{ pointerEvents: "auto", marginTop: mobile ? 8 : 12 }}
      disabled={busy}
      onClick={onClick}
    >
      {busy ? busyLabel : label}
    </button>
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
  const { phase, isYourTurn, yourMode, trickIndex, totalTricks } = props;
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
    const w = mobile ? 64 : Math.round(104 * scale);
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
    // Staged bury: cards committed on the server (reconnect / mid-confirm)
    // render locked; the local selection renders face-up and tappable to put
    // back. Nothing is sent until Confirm.
    const { serverBury, burySelection, busy } = props.staging;
    const staged = [
      ...serverBury.map((card) => ({ card, locked: true })),
      ...burySelection.map((card) => ({ card, locked: false })),
    ].slice(0, 2);
    const w = mobile ? 56 : Math.round(104 * scale);
    const ready = staged.length === 2 && burySelection.length > 0;
    return (
      <>
        <div className={styles.slotRow} style={{ gap: mobile ? 6 : 14 }}>
          {[0, 1].map((i) => {
            const s = staged[i];
            if (!s) {
              return (
                <EmptySlot
                  key={i}
                  label={`slot ${i + 1}`}
                  w={w}
                  mobile={mobile}
                />
              );
            }
            if (s.locked) return <PlayingCard key={i} code={s.card} w={w} />;
            return (
              <PutBackButton
                key={i}
                card={s.card}
                w={w}
                onClick={() => props.staging.onDeselectBury(s.card)}
              />
            );
          })}
        </div>
        <div
          className={ds.overline}
          style={{ fontSize: mobile ? 8 : 10, marginTop: mobile ? 6 : 12 }}
        >
          Burying
        </div>
        {!mobile && (
          <div className={styles.centerSub}>
            {staged.length === 2
              ? "tap a card to put it back"
              : `${staged.length} of 2 chosen · tap a hand card to bury`}
          </div>
        )}
        {ready && (
          <ConfirmButton
            label="Confirm bury"
            busyLabel="Burying…"
            busy={busy}
            mobile={mobile}
            onClick={props.staging.onConfirmBury}
          />
        )}
      </>
    );
  }

  if (phase === "interlude" && yourMode === "under") {
    // Called under: the picker must tuck one card face down for the called
    // suit. Same stage-then-confirm flow as burying, so a stray tap can't
    // lock in the wrong card.
    const w = mobile ? 56 : Math.round(104 * scale);
    const sel = props.staging.underSelection;
    return (
      <>
        {!mobile && (
          <div
            className={ds.overline}
            style={{ fontSize: 10, marginBottom: 12 }}
          >
            {props.calledCardDisplay
              ? `Called ${props.calledCardDisplay} under`
              : "Called under"}
          </div>
        )}
        <div className={styles.slotRow}>
          {sel ? (
            <PutBackButton
              card={sel}
              w={w}
              onClick={props.staging.onDeselectUnder}
            />
          ) : (
            <EmptySlot label="under" w={w} mobile={mobile} />
          )}
        </div>
        {mobile && (
          <div className={ds.overline} style={{ fontSize: 8, marginTop: 6 }}>
            Under card
          </div>
        )}
        {!mobile && (
          <div className={styles.centerSub}>
            {sel
              ? "tap the card to put it back"
              : "tap a hand card to tuck under · it plays face down for the called suit"}
          </div>
        )}
        {sel && (
          <ConfirmButton
            label="Confirm under"
            busyLabel="Tucking under…"
            busy={props.staging.busy}
            mobile={mobile}
            onClick={props.staging.onConfirmUnder}
          />
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
