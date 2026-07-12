import React, { useState } from "react";
import { ChatPanel, type ChatMessage } from "../../../components/chat";
import { ds } from "../../../../lib/ds";
import type { TableView } from "../../../../lib/types";
import Scoreboard from "./Scoreboard";
import styles from "./MobileLogScreen.module.css";

interface MobileLogScreenProps {
  table: TableView;
  yourSeat: number;
  chatMessages: ChatMessage[];
  onSendMessage: (msg: string) => void;
  onClose: () => void;
}

type Tab = "scores" | "chat";

export default function MobileLogScreen({
  table,
  yourSeat,
  chatMessages,
  onSendMessage,
  onClose,
}: MobileLogScreenProps) {
  const [tab, setTab] = useState<Tab>("chat");
  const tabs: Array<{ key: Tab; label: string }> = [
    { key: "scores", label: "Scores" },
    { key: "chat", label: "Chat" },
  ];

  return (
    <div className={styles.screen}>
      <div className={styles.header}>
        <a className={styles.back} onClick={onClose}>
          ← Table
        </a>
        <div className={styles.title}>Log &amp; Chat</div>
        <span style={{ width: 56 }} />
      </div>
      <div className={styles.tabs}>
        {tabs.map((t) => (
          <button
            key={t.key}
            className={`${styles.tab} ${tab === t.key ? styles.tabActive : ""}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className={styles.body}>
        {tab === "scores" ? (
          <div className={styles.scoresPane}>
            <Scoreboard table={table} yourSeat={yourSeat} compact />
          </div>
        ) : (
          <ChatPanel messages={chatMessages} onSendMessage={onSendMessage} />
        )}
      </div>
    </div>
  );
}
