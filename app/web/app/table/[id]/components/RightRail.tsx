import React from "react";
import { ChatPanel, type ChatMessage } from "../../../components/chat";
import type { TableView } from "../../../../lib/types";
import Scoreboard from "./Scoreboard";
import styles from "./RightRail.module.css";

interface RightRailProps {
  table: TableView;
  yourSeat: number | null;
  chatMessages: ChatMessage[];
  onSendMessage: (msg: string) => void;
  seatControls?: (seat: number) => React.ReactNode;
  authorActions?: (msg: ChatMessage) => React.ReactNode;
}

/** Desktop right rail: scoreboard on top, chat filling the rest. */
export default function RightRail({
  table,
  yourSeat,
  chatMessages,
  onSendMessage,
  seatControls,
  authorActions,
}: RightRailProps) {
  return (
    <div className={styles.rail}>
      <div className={styles.scores}>
        <Scoreboard
          table={table}
          yourSeat={yourSeat}
          seatControls={seatControls}
        />
      </div>
      <div className={styles.chat}>
        <ChatPanel
          messages={chatMessages}
          onSendMessage={onSendMessage}
          authorActions={authorActions}
        />
      </div>
    </div>
  );
}
