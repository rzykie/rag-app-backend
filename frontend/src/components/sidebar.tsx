"use client";

import { Plus, MessageSquare, X } from "lucide-react";

interface SidebarProps {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
}

const Sidebar = ({ isOpen, setIsOpen }: SidebarProps) => {
  const chatHistory = [
    { id: 1, title: "Website Design Ideas" },
    { id: 2, title: "JavaScript Help" },
    { id: 3, title: "Travel Recommendations" },
    { id: 4, title: "Recipe Suggestions" },
  ];

  return (
    <div className={`sidebar ${isOpen ? "sidebar-open" : "sidebar-closed"}`}>
      <div className="sidebar-content">
        <div className="sidebar-header">
          <button
            className="new-chat-button"
            onClick={() => {
              // Start a new chat
            }}
          >
            <Plus className="h-4 w-4" /> New chat
          </button>
          <button
            className="close-sidebar-button"
            onClick={() => setIsOpen(false)}
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="chat-history-container">
          <h2 className="chat-history-title">Recent chats</h2>
          {chatHistory.map((chat) => (
            <button key={chat.id} className="chat-history-item">
              <MessageSquare className="h-4 w-4" />
              <span className="truncate">{chat.title}</span>
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          <button className="upgrade-button">Upgrade to Plus</button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
