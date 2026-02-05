"use client";

import ReactMarkdown from "react-markdown";
import { Source } from "@/lib/api";

interface MessageProps {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timestamp?: Date;
}

export function Message({ role, content, sources, timestamp }: MessageProps) {
  const isUser = role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[80%] ${
          isUser ? "order-2" : "order-1"
        }`}
      >
        {/* Avatar */}
        <div
          className={`flex items-start gap-3 ${
            isUser ? "flex-row-reverse" : "flex-row"
          }`}
        >
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium ${
              isUser ? "bg-primary-600" : "bg-gray-600"
            }`}
          >
            {isUser ? "U" : "Rx"}
          </div>

          {/* Message content */}
          <div
            className={`rounded-lg px-4 py-3 ${
              isUser
                ? "bg-primary-600 text-white"
                : "bg-gray-100 text-gray-900"
            }`}
          >
            {isUser ? (
              <p className="whitespace-pre-wrap">{content}</p>
            ) : (
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown
                  components={{
                    // Style links
                    a: ({ ...props }) => (
                      <a
                        className="text-primary-600 hover:underline"
                        target="_blank"
                        rel="noopener noreferrer"
                        {...props}
                      />
                    ),
                    // Style code blocks
                    code: ({ ...props }) => (
                      <code
                        className="bg-gray-200 px-1 py-0.5 rounded text-sm"
                        {...props}
                      />
                    ),
                    // Style lists
                    ul: ({ ...props }) => (
                      <ul className="list-disc list-inside my-2" {...props} />
                    ),
                    ol: ({ ...props }) => (
                      <ol className="list-decimal list-inside my-2" {...props} />
                    ),
                  }}
                >
                  {content}
                </ReactMarkdown>
              </div>
            )}

            {/* Timestamp */}
            {timestamp && (
              <p
                className={`text-xs mt-2 ${
                  isUser ? "text-primary-200" : "text-gray-400"
                }`}
              >
                {timestamp.toLocaleTimeString()}
              </p>
            )}
          </div>
        </div>

        {/* Sources (only for assistant messages) */}
        {!isUser && sources && sources.length > 0 && (
          <div className="mt-2 ml-11">
            <p className="text-xs text-gray-500 mb-1">Sources:</p>
            <div className="flex flex-wrap gap-1">
              {sources.map((source) => (
                <span
                  key={source.source_number}
                  className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-gray-200 text-gray-700"
                  title={`${source.doc_title} - ${source.section}`}
                >
                  [{source.source_number}] {source.doc_title.slice(0, 20)}
                  {source.doc_title.length > 20 ? "..." : ""}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
