"use client";

import { useState } from "react";
import { Source } from "@/lib/api";

interface SourceCardProps {
  source: Source;
  isHighlighted?: boolean;
}

export function SourceCard({ source, isHighlighted = false }: SourceCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Calculate relevance level for visual indicator
  const getRelevanceColor = (score: number) => {
    if (score >= 0.8) return "bg-green-500";
    if (score >= 0.6) return "bg-yellow-500";
    return "bg-orange-500";
  };

  const getRelevanceLabel = (score: number) => {
    if (score >= 0.8) return "High";
    if (score >= 0.6) return "Medium";
    return "Low";
  };

  return (
    <div
      className={`border rounded-lg p-3 transition-all ${
        isHighlighted
          ? "border-primary-500 bg-primary-50"
          : "border-gray-200 hover:border-gray-300"
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-primary-100 text-primary-700 text-xs font-medium">
              {source.source_number}
            </span>
            <h4 className="font-medium text-gray-900 text-sm">
              {source.doc_title}
            </h4>
          </div>
          <p className="text-xs text-gray-500 mt-1 ml-8">{source.section}</p>
        </div>

        {/* Relevance indicator */}
        <div className="flex items-center gap-1">
          <div
            className={`w-2 h-2 rounded-full ${getRelevanceColor(
              source.relevance_score
            )}`}
          />
          <span className="text-xs text-gray-500">
            {getRelevanceLabel(source.relevance_score)} (
            {(source.relevance_score * 100).toFixed(0)}%)
          </span>
        </div>
      </div>

      {/* Content preview */}
      <div className="ml-8">
        <p className="text-sm text-gray-700 leading-relaxed">
          {isExpanded
            ? source.text
            : source.text.length > 200
            ? source.text.slice(0, 200) + "..."
            : source.text}
        </p>

        {/* Expand/collapse button */}
        {source.text.length > 200 && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs text-primary-600 hover:text-primary-700 mt-1"
          >
            {isExpanded ? "Show less" : "Show more"}
          </button>
        )}

        {/* Source URL */}
        {source.source_url && (
          <a
            href={source.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="block text-xs text-primary-600 hover:underline mt-2"
          >
            View full document
          </a>
        )}
      </div>
    </div>
  );
}
