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
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom within the messages container (not the page)
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

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

  return (
    <div className={styles.chatPanel}>
      <div className={styles.chatHeader}>
        <span className={styles.headerTitle}>Chat</span>
      </div>
      <div ref={messagesContainerRef} className={styles.messagesContainer}>
        {messages.length === 0 ? (
          <div className={styles.emptyMessage}>No messages yet. Say hello! 👋</div>
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
