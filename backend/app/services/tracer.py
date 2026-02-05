"""
Query Tracing Service for observability.

Tracks every query through the RAG pipeline:
- Retrieval scores (vector, BM25, RRF)
- Reranking scores
- Step-by-step latencies
- Token usage
- Final response metadata
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings


@dataclass
class StepTiming:
    """Timing information for a single pipeline step."""
    step_name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0

    def complete(self):
        """Mark the step as complete and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class RetrievalScores:
    """Scores from the retrieval phase."""
    doc_id: str
    text_preview: str  # First 100 chars
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_rank: int = 0


@dataclass
class TokenUsage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0


@dataclass
class QueryTrace:
    """Complete trace of a single query through the RAG pipeline."""

    # Identification
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Query information
    query: str = ""
    conversation_id: Optional[str] = None

    # Configuration used
    config: dict = field(default_factory=dict)

    # Retrieval information
    retrieval_method: str = "hybrid"  # "hybrid", "vector_only", "bm25_only"
    candidates_retrieved: int = 0
    final_results_count: int = 0
    retrieval_scores: list[RetrievalScores] = field(default_factory=list)

    # Generation information
    response_text: str = ""
    response_length: int = 0
    sources_cited: int = 0

    # Token usage
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    # Timing
    timings: list[StepTiming] = field(default_factory=list)
    total_duration_ms: float = 0.0

    # Status
    success: bool = True
    error: Optional[str] = None

    # Current step tracking (not serialized)
    _current_step: Optional[StepTiming] = field(default=None, repr=False)
    _start_time: float = field(default_factory=time.time, repr=False)

    def start_step(self, step_name: str) -> None:
        """Start timing a new step."""
        self._current_step = StepTiming(
            step_name=step_name,
            start_time=time.time(),
        )

    def end_step(self) -> None:
        """End the current step and record timing."""
        if self._current_step:
            self._current_step.complete()
            self.timings.append(self._current_step)
            self._current_step = None

    def add_retrieval_score(
        self,
        doc_id: str,
        text: str,
        vector_score: Optional[float] = None,
        vector_rank: Optional[int] = None,
        bm25_score: Optional[float] = None,
        bm25_rank: Optional[int] = None,
        rrf_score: Optional[float] = None,
        rerank_score: Optional[float] = None,
        final_rank: int = 0,
    ) -> None:
        """Add retrieval scores for a document."""
        self.retrieval_scores.append(RetrievalScores(
            doc_id=doc_id,
            text_preview=text[:100] + "..." if len(text) > 100 else text,
            vector_score=vector_score,
            vector_rank=vector_rank,
            bm25_score=bm25_score,
            bm25_rank=bm25_rank,
            rrf_score=rrf_score,
            rerank_score=rerank_score,
            final_rank=final_rank,
        ))

    def set_token_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        embedding_tokens: int = 0,
    ) -> None:
        """Set token usage information."""
        self.token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            embedding_tokens=embedding_tokens,
        )

    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark the trace as complete."""
        self.success = success
        self.error = error
        self.total_duration_ms = (time.time() - self._start_time) * 1000

    def to_dict(self) -> dict:
        """Convert trace to dictionary for serialization."""
        data = {
            'trace_id': self.trace_id,
            'timestamp': self.timestamp,
            'query': self.query,
            'conversation_id': self.conversation_id,
            'config': self.config,
            'retrieval_method': self.retrieval_method,
            'candidates_retrieved': self.candidates_retrieved,
            'final_results_count': self.final_results_count,
            'retrieval_scores': [asdict(s) for s in self.retrieval_scores],
            'response_text': self.response_text[:500] + "..." if len(self.response_text) > 500 else self.response_text,
            'response_length': self.response_length,
            'sources_cited': self.sources_cited,
            'token_usage': asdict(self.token_usage),
            'timings': [asdict(t) for t in self.timings],
            'total_duration_ms': self.total_duration_ms,
            'success': self.success,
            'error': self.error,
        }
        return data


class TracerService:
    """
    Service for recording and retrieving query traces.

    Provides observability into the RAG pipeline by:
    - Recording detailed traces for each query
    - Persisting traces to disk (JSON files)
    - Providing APIs to retrieve and analyze traces
    """

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.enable_tracing
        self.trace_path = Path(self.settings.trace_log_path)
        self.max_traces = self.settings.max_traces_stored

        # Ensure trace directory exists
        if self.enabled:
            self.trace_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache of recent traces
        self._recent_traces: list[QueryTrace] = []

    def create_trace(
        self,
        query: str,
        conversation_id: Optional[str] = None,
    ) -> QueryTrace:
        """Create a new trace for a query."""
        trace = QueryTrace(
            query=query,
            conversation_id=conversation_id,
            config={
                'hybrid_enabled': self.settings.enable_hybrid_search,
                'reranking_enabled': self.settings.enable_reranking,
                'reranker_model': self.settings.reranker_model,
                'vector_weight': self.settings.vector_weight,
                'bm25_weight': self.settings.bm25_weight,
                'top_k': self.settings.top_k_results,
            }
        )
        return trace

    def save_trace(self, trace: QueryTrace) -> None:
        """Save a completed trace."""
        if not self.enabled:
            return

        # Add to recent traces
        self._recent_traces.append(trace)

        # Trim if over limit
        if len(self._recent_traces) > self.max_traces:
            self._recent_traces = self._recent_traces[-self.max_traces:]

        # Save to disk
        try:
            # Save to daily file
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            trace_file = self.trace_path / f"traces_{date_str}.jsonl"

            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trace.to_dict()) + '\n')

        except Exception as e:
            print(f"Warning: Failed to save trace: {e}")

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """Get a specific trace by ID."""
        # Check in-memory cache first
        for trace in self._recent_traces:
            if trace.trace_id == trace_id:
                return trace.to_dict()

        # Search on disk
        if self.enabled:
            for trace_file in sorted(self.trace_path.glob("traces_*.jsonl"), reverse=True):
                try:
                    with open(trace_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            if data.get('trace_id') == trace_id:
                                return data
                except Exception:
                    continue

        return None

    def get_recent_traces(
        self,
        limit: int = 50,
        query_filter: Optional[str] = None,
    ) -> list[dict]:
        """Get recent traces, optionally filtered by query text."""
        traces = []

        # Start with in-memory traces
        for trace in reversed(self._recent_traces):
            trace_dict = trace.to_dict()
            if query_filter is None or query_filter.lower() in trace_dict['query'].lower():
                traces.append(trace_dict)
                if len(traces) >= limit:
                    break

        return traces

    def get_trace_summary(self, days: int = 7) -> dict:
        """Get summary statistics for recent traces."""
        traces = self.get_recent_traces(limit=1000)

        if not traces:
            return {
                'total_queries': 0,
                'avg_duration_ms': 0,
                'success_rate': 0,
                'avg_sources_cited': 0,
            }

        total = len(traces)
        successful = sum(1 for t in traces if t['success'])
        total_duration = sum(t['total_duration_ms'] for t in traces)
        total_sources = sum(t['sources_cited'] for t in traces)

        return {
            'total_queries': total,
            'avg_duration_ms': total_duration / total if total else 0,
            'success_rate': successful / total if total else 0,
            'avg_sources_cited': total_sources / total if total else 0,
            'traces_by_method': self._count_by_field(traces, 'retrieval_method'),
        }

    def _count_by_field(self, traces: list[dict], field: str) -> dict:
        """Count traces by a specific field value."""
        counts: dict[str, int] = {}
        for trace in traces:
            value = trace.get(field, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts


# Global tracer instance
_tracer: Optional[TracerService] = None


def get_tracer() -> TracerService:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = TracerService()
    return _tracer
