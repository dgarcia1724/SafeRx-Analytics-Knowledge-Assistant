from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.retriever import RetrieverService


router = APIRouter()


class SearchRequest(BaseModel):
    """Request model for search endpoint."""

    query: str
    top_k: int = 5
    filter_doc_id: str | None = None
    filter_section: str | None = None


class SearchResult(BaseModel):
    """Individual search result."""

    text: str
    doc_title: str
    section: str
    doc_id: str
    source_url: str
    relevance_score: float
    chunk_index: int | None = None


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    query: str
    results: list[SearchResult]
    total_results: int


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for relevant documents without generating a response.

    This endpoint is useful for:
    - Debugging retrieval quality
    - Building evaluation datasets
    - Direct document search
    """
    try:
        retriever = RetrieverService()

        # Build metadata filter if provided
        filter_metadata = None
        if request.filter_doc_id or request.filter_section:
            filter_metadata = {}
            if request.filter_doc_id:
                filter_metadata["doc_id"] = request.filter_doc_id
            if request.filter_section:
                filter_metadata["section"] = request.filter_section

        # Search
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=filter_metadata,
        )

        # Format results
        search_results = [
            SearchResult(
                text=r.text,
                doc_title=r.metadata.get("doc_title", "Unknown"),
                section=r.metadata.get("section", "General"),
                doc_id=r.metadata.get("doc_id", ""),
                source_url=r.metadata.get("source_url", ""),
                relevance_score=round(r.score, 4),
                chunk_index=r.metadata.get("chunk_index"),
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}",
        )


@router.get("/search/documents")
async def list_documents():
    """List all unique documents in the knowledge base."""
    try:
        retriever = RetrieverService()

        # Get all documents (limited query)
        # Note: This is a simple implementation - for production,
        # you'd want to track documents separately
        stats = retriever.get_collection_stats()

        return {
            "status": "ok",
            "total_chunks": stats["count"],
            "collection_name": stats["name"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}",
        )
