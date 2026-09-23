import React from "react";
import { ChatPanel, type ChatPanelProps } from "../../../components/chat";
import type { TableView } from "../../../../lib/types";
import Scoreboard from "./Scoreboard";
import styles from "./RightRail.module.css";

interface RightRailProps {
  table: TableView;
  yourSeat: number | null;
  chat: ChatPanelProps;
  seatControls?: (seat: number) => React.ReactNode;
}

/** Desktop right rail: scoreboard on top, chat filling the rest. */
export default function RightRail({
  table,
  yourSeat,
  chat,
  seatControls,
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
        <ChatPanel {...chat} />
      </div>
    </div>
  );
}
