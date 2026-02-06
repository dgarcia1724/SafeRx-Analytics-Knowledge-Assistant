"use client";

import { useState, useRef, useEffect } from "react";
import { Message } from "./Message";
import { SourceCard } from "./SourceCard";
import { SearchBar } from "./SearchBar";
import { sendChatMessageStream, Source } from "@/lib/api";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timestamp: Date;
}

export function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [selectedSources, setSelectedSources] = useState<Source[]>([]);
  const [showSources, setShowSources] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (query: string) => {
    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: query,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setError(null);
    setIsLoading(true);

    // Create placeholder for streaming response
    const assistantMessageId = (Date.now() + 1).toString();
    let streamedContent = "";

    try {
      await sendChatMessageStream(
        {
          query,
          conversation_id: conversationId || undefined,
        },
        {
          onSources: (sources, convId) => {
            // Update conversation ID and sources
            setConversationId(convId);
            setSelectedSources(sources);

            // Add initial assistant message
            const assistantMessage: ChatMessage = {
              id: assistantMessageId,
              role: "assistant",
              content: "",
              sources: sources,
              timestamp: new Date(),
            };
            setMessages((prev) => [...prev, assistantMessage]);
          },
          onChunk: (content) => {
            streamedContent += content;
            // Update the assistant message with new content
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? { ...msg, content: streamedContent }
                  : msg
              )
            );
          },
          onDone: () => {
            setIsLoading(false);
          },
          onError: (errorMessage) => {
            setError(errorMessage);
            setIsLoading(false);

            // Update message with error if it exists, or add new error message
            setMessages((prev) => {
              const hasAssistantMsg = prev.some(
                (msg) => msg.id === assistantMessageId
              );
              if (hasAssistantMsg) {
                return prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? {
                        ...msg,
                        content: `Sorry, I encountered an error: ${errorMessage}`,
                      }
                    : msg
                );
              } else {
                return [
                  ...prev,
                  {
                    id: assistantMessageId,
                    role: "assistant" as const,
                    content: `Sorry, I encountered an error: ${errorMessage}`,
                    timestamp: new Date(),
                  },
                ];
              }
            });
          },
        }
      );
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "An error occurred";
      setError(errorMessage);
      setIsLoading(false);

      // Add error message if streaming failed before onSources
      setMessages((prev) => {
        const hasAssistantMsg = prev.some(
          (msg) => msg.id === assistantMessageId
        );
        if (!hasAssistantMsg) {
          return [
            ...prev,
            {
              id: assistantMessageId,
              role: "assistant" as const,
              content: `Sorry, I encountered an error: ${errorMessage}. Please make sure the backend is running and try again.`,
              timestamp: new Date(),
            },
          ];
        }
        return prev;
      });
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setConversationId(null);
    setSelectedSources([]);
    setError(null);
  };

  return (
    <div className="flex h-full">
      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        {/* Messages area */}
        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-500">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mb-4">
                <span className="text-primary-700 font-bold text-2xl">Rx</span>
              </div>
              <h2 className="text-xl font-semibold text-gray-700 mb-2">
                Welcome to SafeRx Analytics
              </h2>
              <p className="text-center max-w-md mb-6">
                Ask me about drug safety, side effects, interactions, and more.
                I&apos;ll search through FDA drug labels to find relevant information.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl">
                {[
                  "What are the side effects of metformin?",
                  "Does warfarin interact with aspirin?",
                  "What are the contraindications for lisinopril?",
                  "What is the recommended dosage for atorvastatin?",
                ].map((example) => (
                  <button
                    key={example}
                    onClick={() => handleSubmit(example)}
                    className="text-left p-3 border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-primary-300 transition text-sm text-gray-600"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto">
              {messages.map((message) => (
                <Message
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  sources={message.sources}
                  timestamp={message.timestamp}
                />
              ))}
              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center text-white text-sm font-medium">
                      Rx
                    </div>
                    <div className="bg-gray-100 rounded-lg px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.1s" }}
                        />
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.2s" }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="border-t bg-white p-4">
          <div className="max-w-4xl mx-auto">
            {error && (
              <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                {error}
              </div>
            )}
            <SearchBar onSubmit={handleSubmit} isLoading={isLoading} />
            {messages.length > 0 && (
              <div className="mt-2 flex justify-end">
                <button
                  onClick={handleClearChat}
                  className="text-xs text-gray-500 hover:text-gray-700"
                >
                  Clear conversation
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Sources sidebar */}
      {showSources && selectedSources.length > 0 && (
        <div className="w-96 border-l bg-gray-50 flex flex-col">
          <div className="p-4 border-b bg-white flex items-center justify-between">
            <h3 className="font-semibold text-gray-900">Sources</h3>
            <button
              onClick={() => setShowSources(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {selectedSources.map((source) => (
              <SourceCard key={source.source_number} source={source} />
            ))}
          </div>
        </div>
      )}

      {/* Show sources button when hidden */}
      {!showSources && selectedSources.length > 0 && (
        <button
          onClick={() => setShowSources(true)}
          className="fixed right-4 bottom-24 bg-primary-600 text-white px-4 py-2 rounded-lg shadow-lg hover:bg-primary-700 transition flex items-center gap-2"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          Sources ({selectedSources.length})
        </button>
      )}
    </div>
  );
}
