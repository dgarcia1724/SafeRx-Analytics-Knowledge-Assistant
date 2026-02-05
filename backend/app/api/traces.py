"""
API endpoints for viewing query traces.

Provides observability into the RAG pipeline by exposing
trace data for debugging and analysis.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.tracer import get_tracer

router = APIRouter(prefix="/api/traces", tags=["traces"])


@router.get("")
async def list_traces(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of traces to return"),
    query: Optional[str] = Query(default=None, description="Filter by query text (case-insensitive)"),
):
    """
    List recent query traces.

    Returns traces in reverse chronological order (most recent first).
    Optionally filter by query text.
    """
    tracer = get_tracer()
    traces = tracer.get_recent_traces(limit=limit, query_filter=query)

    return {
        "count": len(traces),
        "traces": traces,
    }


@router.get("/summary")
async def get_trace_summary(
    days: int = Query(default=7, ge=1, le=30, description="Number of days to summarize"),
):
    """
    Get summary statistics for recent traces.

    Returns aggregated metrics like average latency, success rate, etc.
    """
    tracer = get_tracer()
    summary = tracer.get_trace_summary(days=days)

    return summary


@router.get("/{trace_id}")
async def get_trace(trace_id: str):
    """
    Get a specific trace by ID.

    Returns the full trace details including:
    - Query and response
    - Retrieval scores (vector, BM25, RRF, rerank)
    - Timing breakdown
    - Token usage
    """
    tracer = get_tracer()
    trace = tracer.get_trace(trace_id)

    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    return trace


@router.get("/{trace_id}/scores")
async def get_trace_scores(trace_id: str):
    """
    Get retrieval scores for a specific trace.

    Returns detailed scoring information for each retrieved document,
    useful for debugging retrieval quality.
    """
    tracer = get_tracer()
    trace = tracer.get_trace(trace_id)

    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    return {
        "trace_id": trace_id,
        "query": trace.get("query"),
        "retrieval_method": trace.get("retrieval_method"),
        "scores": trace.get("retrieval_scores", []),
    }


@router.get("/{trace_id}/timings")
async def get_trace_timings(trace_id: str):
    """
    Get timing breakdown for a specific trace.

    Returns step-by-step latency information for identifying
    performance bottlenecks.
    """
    tracer = get_tracer()
    trace = tracer.get_trace(trace_id)

    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    return {
        "trace_id": trace_id,
        "total_duration_ms": trace.get("total_duration_ms"),
        "timings": trace.get("timings", []),
    }
