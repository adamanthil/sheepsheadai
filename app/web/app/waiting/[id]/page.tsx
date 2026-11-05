"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import type { Rules, TableSummary } from "../../../lib/types";
import { apiFetch } from "../../../lib/api";
import { useTableSocket } from "../../../lib/hooks/useTableSocket";
import { useBusy } from "../../../lib/hooks/useBusy";
import { useChatMutes } from "../../../lib/hooks/useChatMutes";
import { useChatUnread } from "../../../lib/hooks/useChatUnread";
import { STORAGE_KEYS } from "../../../lib/storage";
import styles from "./page.module.css";
import { ChatPanel } from "../../components/chat";
import { RemovePlayerButton } from "../../components/RemovePlayerButton";
import type { SeatInfo } from "./components/Seat";
import { ds, useIsMobile } from "../../../lib/ds";
import {
  DesktopWaitingLayout,
  MobileWaitingLayout,
} from "./components/WaitingLayouts";
import RulesPanel, { type AllPassMode } from "./components/RulesPanel";

type TableInfo = TableSummary & {
  seats: Record<string, string | null>;
};

export default function WaitingRoom() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [clientId, setClientId] = useState<string>("");

  // Hydrate client_id from localStorage (Phase 2: no more ?client_id in URLs).
  useEffect(() => {
    if (typeof window === "undefined" || !params?.id) return;
    const stored = window.localStorage.getItem(
      STORAGE_KEYS.clientId(params.id),
    );
    if (stored) {
      setClientId(stored);
    } else {
      // No session for this table — send them back to the lobby.
      router.replace("/");
    }
  }, [params?.id, router]);

  const [table, setTable] = useState<TableInfo | null>(null);
  const [isHost, setIsHost] = useState(false);
  // Set during load() but never rendered; kept so the fetch path stays
  // ready for a spinner without reintroducing an unused binding.
  const [, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [partnerMode, setPartnerMode] = useState<number>(1); // 1 = Called Ace (default), 0 = Jack of Diamonds
  const [allPassMode, setAllPassMode] = useState<AllPassMode>("leasters"); // "leasters" (default) or "doublers"
  const [scoringMode, setScoringMode] = useState<number>(1); // 1 = Double on the Bump (default), 0 = Symmetric
  const [callout, setCallout] = useState<string | null>(null);
  // Mobile only: the chat is a full screen opened from the menu.
  const [chatOpen, setChatOpen] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch("/api/tables");
      const list: TableInfo[] = await res.json();
      const t = list.find((t) => t.id === params?.id);
      if (t) {
        setTable(t);
        // Initialize modes from table rules
        const rules = t.rules ?? {};
        const pMode = rules.partnerMode !== undefined ? rules.partnerMode : 1; // Default to Called Ace
        setPartnerMode(pMode);
        setAllPassMode(
          rules.allPassMode === "doublers" ? "doublers" : "leasters",
        );
        const sMode =
          rules.doubleOnTheBump !== undefined
            ? rules.doubleOnTheBump
              ? 1
              : 0
            : 1; // Default to Double on the Bump
        setScoringMode(sMode);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, [params?.id]);

  // Shared socket hook: lobby updates, chat, auto-navigation on game start,
  // and reconnect-with-backoff all come from useTableSocket.
  const { lastState, chatMessages, sendChatMessage, kickPlayer } =
    useTableSocket(params?.id, clientId, {
      onTableUpdate: (t, hostFlag) => {
        setTable(t as TableInfo);
        if (typeof hostFlag === "boolean") setIsHost(hostFlag);
        const rules = t.rules ?? {};
        if (rules.partnerMode !== undefined) setPartnerMode(rules.partnerMode);
        if (rules.allPassMode !== undefined) setAllPassMode(rules.allPassMode);
        if (rules.doubleOnTheBump !== undefined)
          setScoringMode(rules.doubleOnTheBump ? 1 : 0);
        if (t.status === "playing") {
          router.push(`/table/${params?.id}`);
        }
      },
      onLobbyEvent: (msg) => {
        setCallout(msg);
        setTimeout(() => setCallout(null), 1800);
      },
      onTableClosed: () => setCallout("Table closed"),
    });

  // A state message while in the waiting room means the game has started.
  useEffect(() => {
    if (lastState?.table?.status === "playing" && params?.id) {
      router.push(`/table/${params.id}`);
    }
  }, [lastState, params?.id, router]);

  // The table page's code is fetched now, so Deal cards waits only on the
  // server, not also on the next page.
  useEffect(() => {
    if (params?.id) router.prefetch(`/table/${params.id}`);
  }, [params?.id, router]);

  const seating = useBusy();
  const [seatPressed, setSeatPressed] = useState<number | null>(null);
  function chooseSeat(seat: number) {
    setSeatPressed(seat);
    void seating.run(async () => {
      const res = await apiFetch(`/api/tables/${params?.id}/seat`, {
        method: "POST",
        body: JSON.stringify({ client_id: clientId, seat }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        setError(j?.detail || "Seat selection failed");
        return false;
      }
      setTable(await res.json());
      return true;
    });
  }

  /** PATCH some of the table's rules; false (with the error shown) when the
   * server refused. */
  async function patchRules(
    rules: Partial<Rules>,
    what: string,
  ): Promise<boolean> {
    try {
      const res = await apiFetch(`/api/tables/${params?.id}/rules`, {
        method: "PATCH",
        body: JSON.stringify({ client_id: clientId, rules }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        setError(j?.detail || `Failed to update ${what}`);
        return false;
      }
      return true;
    } catch {
      setError(`Failed to update ${what}`);
      return false;
    }
  }

  // A rule switches the moment it is pressed and switches back if the
  // server refuses. Nothing is sent before the session has loaded.
  function updatePartnerMode(newMode: number) {
    if (!clientId) return;
    const previous = partnerMode;
    setPartnerMode(newMode);
    void patchRules(
      { partnerMode: newMode as Rules["partnerMode"] },
      "partner mode",
    ).then((ok) => ok || setPartnerMode(previous));
  }

  function updateAllPassMode(newMode: AllPassMode) {
    if (!clientId) return;
    const previous = allPassMode;
    setAllPassMode(newMode);
    void patchRules({ allPassMode: newMode }, "all-pass mode").then(
      (ok) => ok || setAllPassMode(previous),
    );
  }

  function updateScoringMode(newMode: number) {
    if (!clientId) return;
    const previous = scoringMode;
    setScoringMode(newMode);
    void patchRules({ doubleOnTheBump: newMode === 1 }, "scoring mode").then(
      (ok) => ok || setScoringMode(previous),
    );
  }

  // Held busy on success: the table page replaces this one.
  const starting = useBusy();
  function startGame() {
    void starting.run(
      async () => {
        const res = await apiFetch(`/api/tables/${params?.id}/start`, {
          method: "POST",
          body: JSON.stringify({ client_id: clientId }),
        });
        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          setError(j?.detail || "Start failed");
          return false;
        }
        router.push(`/table/${params?.id}`);
        return true;
      },
      { hold: true },
    );
  }

  // Held busy on success: the table's close sends everyone to the lobby.
  const closing = useBusy();
  function closeTable() {
    if (!params?.id || !clientId) return;
    void closing.run(
      async () => {
        const res = await apiFetch(`/api/tables/${params?.id}/close`, {
          method: "POST",
          body: JSON.stringify({ client_id: clientId }),
        });
        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          setError(j?.detail || "Close failed");
          return false;
        }
        return true;
      },
      { hold: true },
    );
  }

  // Host-only Remove for another human: on their seat, and on their chat
  // messages (the only place an unseated spectator shows up).
  const removableId = useCallback(
    (target: string | null | undefined) =>
      isHost && target && target !== clientId ? target : null,
    [isHost, clientId],
  );

  const seatItems = useMemo(() => {
    if (!table) return [] as SeatInfo[];
    const out: SeatInfo[] = [];
    for (let i = 1; i <= 5; i++) {
      const isAI = Boolean(table.seatIsAI?.[String(i)]);
      const occupant = table.seatOccupants[String(i)];
      out.push({
        seat: i,
        name: table.seats[String(i)] ?? "",
        isAI,
        isYou: !!clientId && occupant === clientId,
        account: table.seatAccount?.[String(i)] ?? null,
        removableId: isAI ? null : removableId(occupant),
        takeable: Boolean(table.seatTakeable?.[String(i)]),
      });
    }
    return out;
  }, [table, removableId, clientId]);

  const isMobile = useIsMobile();
  const mutes = useChatMutes(params?.id);
  const shortId = `#${String(params?.id || "")
    .slice(0, 4)
    .toUpperCase()}`;
  const roomName = table?.name || params?.id || "Table";
  const hostName = table?.host || null;
  const humanCount = seatItems.filter((s) => !s.isAI).length;
  const visibleChat = mutes.visible(chatMessages);
  const chatUnread = useChatUnread(visibleChat, clientId, chatOpen);

  // Anyone can mute another player's chat for themselves; the host can
  // also remove them.
  const chat = (
    <ChatPanel
      messages={visibleChat}
      onSendMessage={sendChatMessage}
      muted={mutes.muted}
      onUnmute={mutes.unmute}
      authorActions={(msg) => {
        const author = msg.author_id;
        if (!author || author === clientId) return null;
        const name = msg.author ?? "this player";
        const target = removableId(author);
        return (
          <>
            <button
              type="button"
              className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
              onClick={() => mutes.mute(author, name)}
            >
              Mute
            </button>
            {target && (
              <RemovePlayerButton
                name={name}
                onRemove={() => kickPlayer(target)}
              />
            )}
          </>
        );
      }}
    />
  );
  const rules = (
    <RulesPanel
      partnerMode={partnerMode}
      allPassMode={allPassMode}
      scoringMode={scoringMode}
      onPartnerMode={updatePartnerMode}
      onAllPassMode={updateAllPassMode}
      onScoringMode={updateScoringMode}
      variant={isMobile ? "inline" : "panel"}
    />
  );

  const layoutProps = {
    roomName,
    shortId,
    hostName,
    seatItems,
    humanCount,
    error,
    isHost,
    chooseSeat,
    takingSeat: seating.busy ? seatPressed : null,
    removePlayer: kickPlayer,
    startGame,
    starting: starting.busy,
    closeTable,
    closing: closing.busy,
    onLeave: () => router.push("/"),
    chat,
    rules,
  };

  return (
    <div className={styles.root}>
      {callout && <div className={styles.callout}>{callout}</div>}
      {isMobile ? (
        <MobileWaitingLayout
          {...layoutProps}
          chatOpen={chatOpen}
          setChatOpen={setChatOpen}
          chatUnread={chatUnread}
        />
      ) : (
        <DesktopWaitingLayout {...layoutProps} />
      )}
    </div>
  );
}
