"use client";

import type React from "react";
import { useState, useRef, useEffect } from "react";
import { Send, Menu, Plus, Loader2 } from "lucide-react";
import ChatMessage, { Message } from "../components/chat-message";
import Sidebar from "../components/sidebar";
import { useMobile } from "../hooks/use-mobile";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const isMobile = useMobile();

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: userMessage }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response },
      ]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
      if (isMobile) {
        setSidebarOpen(false);
      }
    }
  };

  return (
    <div className="chat-container">
      {/* Sidebar */}
      <Sidebar isOpen={sidebarOpen} setIsOpen={setSidebarOpen} />

      {/* Main Content */}
      <div className="main-chat-content">
        {/* Header */}
        <header className="chat-header">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="menu-button-mobile"
          >
            <Menu />
          </button>
          <h1>RAG Chat</h1>
          <button className="new-chat-header-button">
            <Plus className="h-4 w-4" />
          </button>
        </header>

        {/* Chat Messages */}
        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="welcome-container">
              <h2 className="welcome-title">How can I help you today?</h2>
              <p className="welcome-subtitle">
                Ask me anything about the company policies, vision, mission, or
                any other information available in our knowledge base.
              </p>
            </div>
          ) : (
            messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))
          )}
          {isLoading && (
            <div className="loading-dots">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>AI is thinking...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="chat-input-container">
          <form onSubmit={handleSubmit} className="chat-input-wrapper">
            <textarea
              value={input}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
                setInput(e.target.value)
              }
              placeholder="Message RAG Chat..."
              className="chat-input"
              rows={1}
              onKeyDown={(e: React.KeyboardEvent<HTMLTextAreaElement>) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (input.trim() !== "") {
                    handleSubmit(e as any);
                  }
                }
              }}
            />
            <button
              type="submit"
              disabled={isLoading || input.trim() === ""}
              className="chat-input-button"
            >
              <Send className="h-4 w-4" />
            </button>
          </form>
          <p className="text-xs text-center mt-2 text-zinc-500">
            RAG Chat can make mistakes. Consider checking important information.
          </p>
        </div>
      </div>
    </div>
  );
}
