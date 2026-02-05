"use client";

import { useState, FormEvent, KeyboardEvent } from "react";

interface SearchBarProps {
  onSubmit: (query: string) => void;
  isLoading?: boolean;
  placeholder?: string;
}

export function SearchBar({
  onSubmit,
  isLoading = false,
  placeholder = "Ask about drug safety, interactions, side effects...",
}: SearchBarProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query.trim());
      setQuery("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="flex items-end gap-2 bg-white border border-gray-300 rounded-lg p-2 shadow-sm focus-within:ring-2 focus-within:ring-primary-500 focus-within:border-primary-500">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          className="flex-1 resize-none border-0 focus:ring-0 focus:outline-none text-gray-900 placeholder-gray-400 min-h-[40px] max-h-[120px]"
          style={{
            height: "auto",
            overflowY: query.split("\n").length > 3 ? "auto" : "hidden",
          }}
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={!query.trim() || isLoading}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            query.trim() && !isLoading
              ? "bg-primary-600 text-white hover:bg-primary-700"
              : "bg-gray-200 text-gray-400 cursor-not-allowed"
          }`}
        >
          {isLoading ? (
            <span className="flex items-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Searching...
            </span>
          ) : (
            "Send"
          )}
        </button>
      </div>
      <p className="text-xs text-gray-400 mt-1 ml-2">
        Press Enter to send, Shift+Enter for new line
      </p>
    </form>
  );
}
