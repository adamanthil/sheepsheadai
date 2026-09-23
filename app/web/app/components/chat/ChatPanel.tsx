"use client";

import { useState, useRef, useEffect } from "react";
import { ds } from "../../../lib/ds";
import styles from "./ChatPanel.module.css";
import type { ChatMessage } from "../../../lib/types";

export type { ChatMessage };

export interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  /** Controls offered for a player message's author (e.g. remove, mute),
   * or null for none. Clicking the author's name shows them. */
  authorActions?: (msg: ChatMessage) => React.ReactNode;
  /** Authors this viewer muted (client id -> name), listed in the header
   * so they can be unmuted; the caller filters their messages out. */
  muted?: Record<string, string>;
  onUnmute?: (clientId: string) => void;
}

export function ChatPanel({
  messages,
  onSendMessage,
  authorActions,
  muted,
  onUnmute,
}: ChatPanelProps) {
  const [inputValue, setInputValue] = useState("");
  // The message whose author controls are open, if any.
  const [actionsOpenFor, setActionsOpenFor] = useState<string | null>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom within the messages container (not the page)
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSendMessage(inputValue.trim());
      setInputValue("");
    }
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className={styles.chatPanel}>
      <div className={styles.chatHeader}>
        <span className={styles.headerTitle}>Chat</span>
        {muted && Object.keys(muted).length > 0 && (
          <div className={styles.mutedList}>
            {Object.entries(muted).map(([id, name]) => (
              <button
                key={id}
                type="button"
                className={styles.mutedChip}
                onClick={() => onUnmute?.(id)}
                title={`Unmute ${name}`}
              >
                {name} muted ×
              </button>
            ))}
          </div>
        )}
      </div>
      <div ref={messagesContainerRef} className={styles.messagesContainer}>
        {messages.length === 0 ? (
          <div className={styles.emptyMessage}>
            No messages yet. Say hello! 👋
          </div>
        ) : (
          messages.map((msg) => {
            const actions =
              msg.type === "player" && authorActions
                ? authorActions(msg)
                : null;
            const open = actions !== null && actionsOpenFor === msg.id;
            return (
              <div
                key={msg.id}
                className={`${styles.message} ${msg.type === "system" ? styles.systemMessage : styles.playerMessage}`}
              >
                {msg.type === "system" ? (
                  <div className={styles.systemText}>
                    {msg.author && (
                      <>
                        <span className={styles.systemActor}>{msg.author}</span>
                        {msg.author_is_ai && (
                          <span
                            className={`${ds.badge} ${ds.badgeQuiet}`}
                            style={{ fontSize: 8, marginLeft: 4 }}
                          >
                            AI
                          </span>
                        )}{" "}
                      </>
                    )}
                    {msg.body}
                  </div>
                ) : (
                  <div className={styles.playerMessageContent}>
                    {actions !== null ? (
                      <button
                        type="button"
                        className={`${styles.author} ${styles.authorButton}`}
                        aria-expanded={open}
                        onClick={() => setActionsOpenFor(open ? null : msg.id)}
                      >
                        {msg.author}:
                      </button>
                    ) : (
                      <span className={styles.author}>{msg.author}:</span>
                    )}
                    <span className={styles.body}>{msg.body}</span>
                  </div>
                )}
                {open && <div className={styles.authorActions}>{actions}</div>}
                <div className={styles.timestamp}>
                  {formatTime(msg.timestamp)}
                </div>
              </div>
            );
          })
        )}
      </div>
      <form onSubmit={handleSubmit} className={styles.inputForm}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type a message..."
          className={styles.input}
          maxLength={500}
        />
        <button
          type="submit"
          className={styles.sendButton}
          disabled={!inputValue.trim()}
        >
          ↑
        </button>
      </form>
    </div>
  );
}
