import React, { useMemo } from 'react';
import { CardText, SeatAvatar, ds } from '../../../../lib/ds';
import styles from './ActionBar.module.css';

interface ActionBarProps {
  yourName: string;
  yourSeat: number;
  isYourTurn: boolean;
  actorName: string;
  helper: string;
  validActions: number[];
  actionLookup: Record<string, string>;
  onTakeAction: (id: number) => void;
  hasLastTrick: boolean;
  showPrev: boolean;
  onTogglePrev: () => void;
  onShowScores: () => void;
  isHost: boolean;
  confirmClose: boolean;
  onConfirmClose: (v: boolean) => void;
  onCloseTable: () => void;
  isMobile: boolean;
}

export default function ActionBar(props: ActionBarProps) {
  const { validActions, actionLookup } = props;

  // Primary action buttons: everything that isn't a card PLAY/BURY (those are
  // taken by tapping cards), plus the explicit PLAY UNDER.
  const actionButtons = useMemo(() => (
    validActions
      .map((aid) => ({ id: aid, label: actionLookup[String(aid)] }))
      .filter((a) => a.label && ((!a.label.startsWith('PLAY') && !a.label.startsWith('BURY')) || a.label === 'PLAY UNDER'))
  ), [validActions, actionLookup]);

  const utils = (
    <div className={styles.utilRow}>
      {props.hasLastTrick && (
        <button className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`} onClick={props.onTogglePrev}>
          {props.showPrev ? 'Hide prev' : 'Show prev'}
        </button>
      )}
      <button className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`} onClick={props.onShowScores}>Scores</button>
      {props.isHost && (
        props.confirmClose ? (
          <>
            <button className={`${ds.btn} ${ds.btnSm}`} onClick={props.onCloseTable}>Confirm close</button>
            <button className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`} onClick={() => props.onConfirmClose(false)}>Cancel</button>
          </>
        ) : (
          <button className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm} ${styles.dangerLink}`} onClick={() => props.onConfirmClose(true)}>Close table</button>
        )
      )}
    </div>
  );

  const primaries = actionButtons.map((a) => {
    const accent = a.label === 'PICK' || a.label.startsWith('CALL') || a.label === 'ALONE' || a.label === 'PLAY UNDER' || a.label.startsWith('CONFIRM');
    return (
      <button key={a.id} className={`${ds.btn} ${accent ? ds.btnAccent : ds.btnGhost}`} onClick={() => props.onTakeAction(a.id)}>
        <CardText>{a.label}</CardText>
      </button>
    );
  });

  if (props.isMobile) {
    return (
      <div className={styles.mob}>
        <div className={styles.mobHelper}>
          {props.isYourTurn ? props.helper : `Waiting for ${props.actorName}…`}
        </div>
        <div className={styles.mobRow}>{primaries}</div>
        <div className={styles.mobRow}>{utils}</div>
      </div>
    );
  }

  return (
    <div className={styles.desk}>
      <div className={styles.deskWho}>
        <SeatAvatar name={props.yourName} size={32} tone="you" />
        <div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <div className={styles.whoName}>{props.yourName}</div>
            <span className={ds.badge} style={{ fontSize: 9 }}>You</span>
          </div>
          <div className={styles.whoSub}>seat {props.yourSeat}{props.isYourTurn ? ' · your move' : ''}</div>
        </div>
      </div>
      <div className={styles.deskRight}>
        {props.isYourTurn
          ? <span className={styles.helper}>{props.helper}</span>
          : <span className={styles.waiting}>Waiting for {props.actorName}…</span>}
        {primaries}
        {utils}
      </div>
    </div>
  );
}
