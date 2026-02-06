"""
Chat API endpoint with hybrid retrieval and tracing.

Pipeline:
1. Create query trace
2. Hybrid retrieval (vector + BM25)
3. RRF fusion
4. Optional reranking
5. LLM generation with context
6. Save trace and return response
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json

from app.config import get_settings
from app.services.retriever import RetrieverService, RetrievalResult
from app.services.reranker import RerankerPipeline
from app.services.generator import GeneratorService
from app.services.tracer import get_tracer, QueryTrace
from app.prompts.system import format_sources


router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    query: str
    conversation_id: str | None = None
    top_k: int | None = None

    # Optional overrides for testing/comparison
    enable_hybrid: bool | None = None
    enable_reranking: bool | None = None


class Source(BaseModel):
    """Source document model."""

    source_number: int
    doc_title: str
    section: str
    text: str
    relevance_score: float
    doc_id: str
    source_url: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    sources: list[Source]
    conversation_id: str
    query: str
    trace_id: Optional[str] = None


class DebugChatResponse(ChatResponse):
    """Extended response with debug information."""

    debug: dict = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat query through the RAG pipeline.

    Pipeline:
    1. Hybrid retrieval (vector + BM25 with RRF fusion)
    2. Optional cross-encoder reranking
    3. LLM generation with retrieved context
    4. Return response with citations
    """
    settings = get_settings()
    tracer = get_tracer()

    # Create trace
    trace = tracer.create_trace(
        query=request.query,
        conversation_id=request.conversation_id,
    )

    try:
        # Initialize services
        trace.start_step("init_services")
        pipeline = RerankerPipeline()
        generator = GeneratorService()
        trace.end_step()

        # Check if collection has documents
        trace.start_step("check_collection")
        stats = pipeline.get_stats()
        vector_count = stats['hybrid']['vector'].get('count', 0)
        if vector_count == 0:
            raise HTTPException(
                status_code=503,
                detail="No documents in knowledge base. Please run ingestion first.",
            )
        trace.end_step()

        # Determine retrieval parameters
        top_k = request.top_k if request.top_k else settings.top_k_results
        enable_reranking = (
            request.enable_reranking
            if request.enable_reranking is not None
            else settings.enable_reranking
        )

        # Retrieve relevant documents using hybrid search + reranking
        trace.start_step("retrieval")
        hybrid_results, retrieval_info = pipeline.search(
            query=request.query,
            top_k=top_k,
            enable_reranking=enable_reranking,
        )
        trace.end_step()

        # Record retrieval info in trace
        trace.candidates_retrieved = retrieval_info.get('hybrid_candidates', 0)
        trace.final_results_count = len(hybrid_results)
        trace.retrieval_method = (
            "hybrid_reranked" if retrieval_info.get('reranked')
            else "hybrid"
        )

        # Add retrieval scores to trace
        for result in hybrid_results:
            trace.add_retrieval_score(
                doc_id=result.doc_id,
                text=result.text,
                vector_score=result.vector_score,
                vector_rank=result.vector_rank,
                bm25_score=result.bm25_score,
                bm25_rank=result.bm25_rank,
                rrf_score=result.rrf_score,
                rerank_score=result.rerank_score,
                final_rank=result.final_rank,
            )

        # Convert to RetrievalResult for generator compatibility
        retrieval_results = [r.to_retrieval_result() for r in hybrid_results]

        # Generate response
        trace.start_step("generation")
        if retrieval_results:
            generation_result = generator.generate(
                query=request.query,
                context_chunks=retrieval_results,
            )
        else:
            generation_result = generator.generate_refusal(request.query)
        trace.end_step()

        # Record generation info
        trace.response_text = generation_result.text
        trace.response_length = len(generation_result.text)
        if generation_result.usage:
            trace.set_token_usage(
                prompt_tokens=generation_result.usage.get('prompt_tokens', 0),
                completion_tokens=generation_result.usage.get('completion_tokens', 0),
                total_tokens=generation_result.usage.get('total_tokens', 0),
            )

        # Format sources for response
        sources = format_sources(retrieval_results)
        trace.sources_cited = len(sources)

        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Complete trace
        trace.complete(success=True)
        tracer.save_trace(trace)

        return ChatResponse(
            response=generation_result.text,
            sources=[Source(**s) for s in sources],
            conversation_id=conversation_id,
            query=request.query,
            trace_id=trace.trace_id,
        )

    except HTTPException:
        trace.complete(success=False, error="HTTP exception")
        tracer.save_trace(trace)
        raise

    except Exception as e:
        trace.complete(success=False, error=str(e))
        tracer.save_trace(trace)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}",
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Process a chat query with streaming response.

    Returns Server-Sent Events (SSE) with:
    - 'sources' event: Retrieved sources (sent first)
    - 'chunk' events: Streamed response tokens
    - 'done' event: Final message with metadata
    """
    settings = get_settings()
    tracer = get_tracer()

    # Create trace
    trace = tracer.create_trace(
        query=request.query,
        conversation_id=request.conversation_id,
    )

    async def generate_stream():
        try:
            # Initialize services
            trace.start_step("init_services")
            pipeline = RerankerPipeline()
            generator = GeneratorService()
            trace.end_step()

            # Check if collection has documents
            stats = pipeline.get_stats()
            vector_count = stats['hybrid']['vector'].get('count', 0)
            if vector_count == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No documents in knowledge base'})}\n\n"
                return

            # Determine retrieval parameters
            top_k = request.top_k if request.top_k else settings.top_k_results
            enable_reranking = (
                request.enable_reranking
                if request.enable_reranking is not None
                else settings.enable_reranking
            )

            # Retrieve relevant documents
            trace.start_step("retrieval")
            hybrid_results, retrieval_info = pipeline.search(
                query=request.query,
                top_k=top_k,
                enable_reranking=enable_reranking,
            )
            trace.end_step()

            # Record retrieval info
            trace.candidates_retrieved = retrieval_info.get('hybrid_candidates', 0)
            trace.final_results_count = len(hybrid_results)

            # Convert to RetrievalResult for generator compatibility
            retrieval_results = [r.to_retrieval_result() for r in hybrid_results]

            # Format and send sources first
            sources = format_sources(retrieval_results)
            conversation_id = request.conversation_id or str(uuid.uuid4())

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'conversation_id': conversation_id})}\n\n"

            # Stream generation
            trace.start_step("generation")
            full_response = ""

            if retrieval_results:
                for chunk in generator.generate_stream(
                    query=request.query,
                    context_chunks=retrieval_results,
                ):
                    if chunk.content:
                        full_response += chunk.content
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk.content})}\n\n"

                    if chunk.is_final and chunk.usage:
                        trace.set_token_usage(
                            prompt_tokens=chunk.usage.get('prompt_tokens', 0),
                            completion_tokens=chunk.usage.get('completion_tokens', 0),
                            total_tokens=chunk.usage.get('total_tokens', 0),
                        )
            else:
                # No results - generate refusal
                refusal = generator.generate_refusal(request.query)
                full_response = refusal.text
                yield f"data: {json.dumps({'type': 'chunk', 'content': refusal.text})}\n\n"

            trace.end_step()

            # Record response
            trace.response_text = full_response
            trace.response_length = len(full_response)
            trace.sources_cited = len(sources)

            # Complete trace
            trace.complete(success=True)
            tracer.save_trace(trace)

            # Send done event
            yield f"data: {json.dumps({'type': 'done', 'trace_id': trace.trace_id})}\n\n"

        except Exception as e:
            trace.complete(success=False, error=str(e))
            tracer.save_trace(trace)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/debug", response_model=DebugChatResponse)
async def chat_debug(request: ChatRequest):
    """
    Process a chat query with detailed debug information.

    Returns the same response as /chat but with additional
    debug information including retrieval scores, timings, etc.
    """
    settings = get_settings()
    tracer = get_tracer()

    trace = tracer.create_trace(
        query=request.query,
        conversation_id=request.conversation_id,
    )

    try:
        # Initialize services
        trace.start_step("init_services")
        pipeline = RerankerPipeline()
        generator = GeneratorService()
        trace.end_step()

        # Check collection
        stats = pipeline.get_stats()
        vector_count = stats['hybrid']['vector'].get('count', 0)
        if vector_count == 0:
            raise HTTPException(
                status_code=503,
                detail="No documents in knowledge base.",
            )

        # Retrieve with timing
        top_k = request.top_k if request.top_k else settings.top_k_results
        enable_reranking = (
            request.enable_reranking
            if request.enable_reranking is not None
            else settings.enable_reranking
        )

        trace.start_step("retrieval")
        hybrid_results, retrieval_info = pipeline.search(
            query=request.query,
            top_k=top_k,
            enable_reranking=enable_reranking,
        )
        trace.end_step()

        trace.candidates_retrieved = retrieval_info.get('hybrid_candidates', 0)
        trace.final_results_count = len(hybrid_results)

        # Build debug scores
        debug_scores = []
        for result in hybrid_results:
            trace.add_retrieval_score(
                doc_id=result.doc_id,
                text=result.text,
                vector_score=result.vector_score,
                vector_rank=result.vector_rank,
                bm25_score=result.bm25_score,
                bm25_rank=result.bm25_rank,
                rrf_score=result.rrf_score,
                rerank_score=result.rerank_score,
                final_rank=result.final_rank,
            )
            debug_scores.append({
                'doc_id': result.doc_id,
                'final_rank': result.final_rank,
                'vector_score': result.vector_score,
                'vector_rank': result.vector_rank,
                'bm25_score': result.bm25_score,
                'bm25_rank': result.bm25_rank,
                'rrf_score': result.rrf_score,
                'rerank_score': result.rerank_score,
            })

        retrieval_results = [r.to_retrieval_result() for r in hybrid_results]

        # Generate
        trace.start_step("generation")
        if retrieval_results:
            generation_result = generator.generate(
                query=request.query,
                context_chunks=retrieval_results,
            )
        else:
            generation_result = generator.generate_refusal(request.query)
        trace.end_step()

        trace.response_text = generation_result.text
        trace.response_length = len(generation_result.text)

        sources = format_sources(retrieval_results)
        trace.sources_cited = len(sources)

        conversation_id = request.conversation_id or str(uuid.uuid4())

        trace.complete(success=True)
        tracer.save_trace(trace)

        # Build debug info
        debug_info = {
            'retrieval': {
                'method': 'hybrid_reranked' if retrieval_info.get('reranked') else 'hybrid',
                'candidates': retrieval_info.get('hybrid_candidates', 0),
                'vector_results': retrieval_info.get('vector_results', 0),
                'bm25_results': retrieval_info.get('bm25_results', 0),
                'reranked': retrieval_info.get('reranked', False),
                'final_count': len(hybrid_results),
            },
            'scores': debug_scores,
            'timings': [
                {'step': t.step_name, 'duration_ms': t.duration_ms}
                for t in trace.timings
            ],
            'total_duration_ms': trace.total_duration_ms,
            'config': trace.config,
        }

        return DebugChatResponse(
            response=generation_result.text,
            sources=[Source(**s) for s in sources],
            conversation_id=conversation_id,
            query=request.query,
            trace_id=trace.trace_id,
            debug=debug_info,
        )

    except HTTPException:
        trace.complete(success=False, error="HTTP exception")
        tracer.save_trace(trace)
        raise

    except Exception as e:
        trace.complete(success=False, error=str(e))
        tracer.save_trace(trace)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}",
        )


@router.get("/chat/stats")
async def chat_stats():
    """Get statistics about the knowledge base and pipeline."""
    try:
        pipeline = RerankerPipeline()
        stats = pipeline.get_stats()

        return {
            "status": "ok",
            "pipeline": stats,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}",
        )


@router.get("/chat/compare")
async def compare_retrieval(
    query: str = Query(..., description="Query to compare"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
):
    """
    Compare retrieval methods side-by-side.

    Returns results from:
    - Vector-only search
    - BM25-only search
    - Hybrid (RRF fusion)
    - Hybrid + reranking

    Useful for understanding the impact of each retrieval component.
    """
    try:
        from app.services.hybrid_retriever import HybridRetrieverService
        from app.services.reranker import RerankerService

        hybrid_retriever = HybridRetrieverService()
        reranker = RerankerService()

        results = {}

        # Vector-only
        vector_results = hybrid_retriever.vector_only_search(query, top_k=top_k)
        results['vector_only'] = [
            {
                'rank': i + 1,
                'doc_id': r.metadata.get('doc_id', 'unknown'),
                'score': r.score,
                'text_preview': r.text[:100] + '...' if len(r.text) > 100 else r.text,
            }
            for i, r in enumerate(vector_results)
        ]

        # BM25-only
        bm25_results = hybrid_retriever.bm25_only_search(query, top_k=top_k)
        results['bm25_only'] = [
            {
                'rank': r.rank,
                'doc_id': r.doc_id,
                'score': r.score,
                'text_preview': r.text[:100] + '...' if len(r.text) > 100 else r.text,
            }
            for r in bm25_results
        ]

        # Hybrid (no reranking)
        hybrid_results, _ = hybrid_retriever.search(query, top_k=top_k)
        results['hybrid'] = [
            {
                'rank': r.final_rank,
                'doc_id': r.doc_id,
                'rrf_score': r.rrf_score,
                'vector_rank': r.vector_rank,
                'bm25_rank': r.bm25_rank,
                'text_preview': r.text[:100] + '...' if len(r.text) > 100 else r.text,
            }
            for r in hybrid_results[:top_k]
        ]

        # Hybrid + reranking
        try:
            reranked = reranker.rerank(query, hybrid_results[:20], top_k=top_k)
            results['hybrid_reranked'] = [
                {
                    'rank': i + 1,
                    'doc_id': r.doc_id,
                    'rerank_score': r.rerank_score,
                    'original_rrf_score': r.rrf_score,
                    'text_preview': r.text[:100] + '...' if len(r.text) > 100 else r.text,
                }
                for i, r in enumerate(reranked)
            ]
        except Exception as e:
            results['hybrid_reranked'] = {'error': str(e)}

        return {
            'query': query,
            'top_k': top_k,
            'results': results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error comparing retrieval: {str(e)}",
        )
