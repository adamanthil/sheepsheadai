"use client";

import { useState, useRef, useEffect } from 'react';
import styles from './ChatPanel.module.css';

export interface ChatMessage {
  id: string;
  table_id: string;
  type: 'player' | 'system';
  author: string | null;
  body: string;
  timestamp: number;
}

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
}

export function ChatPanel({ messages, onSendMessage }: ChatPanelProps) {
  const [inputValue, setInputValue] = useState('');
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [collapseInitialized, setCollapseInitialized] = useState(false);
  const [hasUnread, setHasUnread] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const prevMessageCount = useRef(messages.length);

  // On first mount, auto-collapse on narrow/mobile screens
  useEffect(() => {
    if (collapseInitialized) return;
    if (typeof window === 'undefined') return;
    const prefersCollapsed = window.matchMedia('(max-width: 640px)').matches;
    if (prefersCollapsed) {
      setIsCollapsed(true);
    }
    setCollapseInitialized(true);
  }, [collapseInitialized]);

  // Auto-scroll to bottom when new messages arrive (if not collapsed)
  useEffect(() => {
    if (!isCollapsed) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      setHasUnread(false);
    } else if (messages.length > prevMessageCount.current) {
      setHasUnread(true);
    }
    prevMessageCount.current = messages.length;
  }, [messages, isCollapsed]);

  // Clear unread when expanding
  useEffect(() => {
    if (!isCollapsed) {
      setHasUnread(false);
      // Scroll to bottom when expanding
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'auto' });
      }, 50);
    }
  }, [isCollapsed]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (isCollapsed) {
    return (
      <button
        className={`${styles.collapsedToggle} ${hasUnread ? styles.hasUnread : ''}`}
        onClick={() => setIsCollapsed(false)}
        aria-label="Open chat"
      >
        <span className={styles.chatIcon}>💬</span>
        {hasUnread && <span className={styles.unreadDot} />}
      </button>
    );
  }

  return (
    <div className={styles.chatPanel}>
      <div className={styles.chatHeader}>
        <span className={styles.headerTitle}>Chat</span>
        <button
          className={styles.collapseButton}
          onClick={() => setIsCollapsed(true)}
          aria-label="Minimize chat"
        >
          −
        </button>
      </div>
      <div className={styles.messagesContainer}>
        {messages.length === 0 ? (
          <div className={styles.emptyMessage}>No messages yet</div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`${styles.message} ${msg.type === 'system' ? styles.systemMessage : styles.playerMessage}`}
            >
              {msg.type === 'system' ? (
                <div className={styles.systemText}>{msg.body}</div>
              ) : (
                <div className={styles.playerMessageContent}>
                  <span className={styles.author}>{msg.author}:</span>
                  <span className={styles.body}>{msg.body}</span>
                </div>
              )}
              <div className={styles.timestamp}>{formatTime(msg.timestamp)}</div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
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
        <button type="submit" className={styles.sendButton} disabled={!inputValue.trim()}>
          ↑
        </button>
      </form>
    </div>
  );
}
