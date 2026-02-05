"""
Hybrid Retriever Service combining vector search and BM25.

Uses Reciprocal Rank Fusion (RRF) to combine results from both retrievers,
leveraging the strengths of each:
- Vector search: semantic similarity, paraphrase matching
- BM25: exact keyword matches, rare terms, drug names
"""

from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings
from app.services.retriever import RetrieverService, RetrievalResult
from app.services.bm25_retriever import BM25RetrieverService, BM25Result


@dataclass
class HybridResult:
    """Represents a result from hybrid retrieval with fusion scores."""

    doc_id: str
    text: str
    metadata: dict
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf_score: float = 0.0
    final_rank: int = 0
    rerank_score: Optional[float] = None

    def to_retrieval_result(self) -> RetrievalResult:
        """Convert to standard RetrievalResult for downstream compatibility."""
        return RetrievalResult(
            text=self.text,
            metadata=self.metadata,
            score=self.rerank_score if self.rerank_score is not None else self.rrf_score,
        )


@dataclass
class RetrievalTrace:
    """Traces the retrieval process for observability."""

    query: str
    vector_results_count: int = 0
    bm25_results_count: int = 0
    fused_results_count: int = 0
    reranked_results_count: int = 0
    final_results_count: int = 0
    vector_top_docs: list[str] = field(default_factory=list)
    bm25_top_docs: list[str] = field(default_factory=list)
    fusion_weights: dict = field(default_factory=dict)


class HybridRetrieverService:
    """
    Hybrid retriever combining vector search and BM25 with RRF fusion.

    The retrieval pipeline:
    1. Query both vector store (ChromaDB) and BM25 index
    2. Apply Reciprocal Rank Fusion to combine rankings
    3. Return top candidates for optional reranking
    """

    def __init__(self):
        self.settings = get_settings()
        self.vector_retriever = RetrieverService()
        self.bm25_retriever = BM25RetrieverService()

        # Fusion parameters
        self.rrf_k = self.settings.rrf_k
        self.vector_weight = self.settings.vector_weight
        self.bm25_weight = self.settings.bm25_weight

    def search(
        self,
        query: str,
        top_k: int = None,
        enable_hybrid: bool = None,
    ) -> tuple[list[HybridResult], RetrievalTrace]:
        """
        Perform hybrid search combining vector and BM25 retrieval.

        Args:
            query: Search query
            top_k: Number of results to return (defaults to retrieval_top_k)
            enable_hybrid: Override hybrid search setting

        Returns:
            Tuple of (results, trace) for downstream processing
        """
        if top_k is None:
            top_k = self.settings.retrieval_top_k

        if enable_hybrid is None:
            enable_hybrid = self.settings.enable_hybrid_search

        trace = RetrievalTrace(
            query=query,
            fusion_weights={
                'vector': self.vector_weight,
                'bm25': self.bm25_weight,
                'rrf_k': self.rrf_k,
            }
        )

        # Get vector search results
        vector_results = self.vector_retriever.search(query, top_k=top_k)
        trace.vector_results_count = len(vector_results)
        trace.vector_top_docs = [
            r.metadata.get('doc_id', 'unknown')[:30]
            for r in vector_results[:5]
        ]

        if not enable_hybrid:
            # Return vector-only results
            results = [
                HybridResult(
                    doc_id=r.metadata.get('doc_id', f'vec_{i}'),
                    text=r.text,
                    metadata=r.metadata,
                    vector_score=r.score,
                    vector_rank=i + 1,
                    rrf_score=r.score,
                    final_rank=i + 1,
                )
                for i, r in enumerate(vector_results)
            ]
            trace.final_results_count = len(results)
            return results, trace

        # Get BM25 results
        bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        trace.bm25_results_count = len(bm25_results)
        trace.bm25_top_docs = [
            r.metadata.get('doc_id', 'unknown')[:30]
            for r in bm25_results[:5]
        ]

        # Fuse results using RRF
        fused_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
        )
        trace.fused_results_count = len(fused_results)

        # Sort by RRF score and assign final ranks
        fused_results.sort(key=lambda x: x.rrf_score, reverse=True)
        for i, result in enumerate(fused_results):
            result.final_rank = i + 1

        # Limit to top_k
        results = fused_results[:top_k]
        trace.final_results_count = len(results)

        return results, trace

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[RetrievalResult],
        bm25_results: list[BM25Result],
    ) -> list[HybridResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF Formula: score = sum(1 / (k + rank)) for each retriever
        where k is a constant (typically 60) that dampens the impact of
        high rankings.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search

        Returns:
            List of HybridResult with fused scores
        """
        # Build a map of doc_id -> HybridResult
        results_map: dict[str, HybridResult] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            doc_id = self._get_doc_id(result)

            if doc_id not in results_map:
                results_map[doc_id] = HybridResult(
                    doc_id=doc_id,
                    text=result.text,
                    metadata=result.metadata,
                )

            results_map[doc_id].vector_score = result.score
            results_map[doc_id].vector_rank = rank

            # Add weighted RRF contribution
            rrf_contrib = self.vector_weight * (1 / (self.rrf_k + rank))
            results_map[doc_id].rrf_score += rrf_contrib

        # Process BM25 results
        for result in bm25_results:
            doc_id = result.doc_id

            if doc_id not in results_map:
                results_map[doc_id] = HybridResult(
                    doc_id=doc_id,
                    text=result.text,
                    metadata=result.metadata,
                )

            results_map[doc_id].bm25_score = result.score
            results_map[doc_id].bm25_rank = result.rank

            # Add weighted RRF contribution
            rrf_contrib = self.bm25_weight * (1 / (self.rrf_k + result.rank))
            results_map[doc_id].rrf_score += rrf_contrib

        return list(results_map.values())

    def _get_doc_id(self, result: RetrievalResult) -> str:
        """Extract a unique document ID from a retrieval result."""
        # Use the combination of doc_id and chunk_index if available
        doc_id = result.metadata.get('doc_id', '')
        chunk_index = result.metadata.get('chunk_index', '')

        if doc_id and chunk_index is not None:
            return f"{doc_id}_chunk_{chunk_index}"
        elif doc_id:
            return doc_id

        # Fallback: hash the text
        return f"doc_{hash(result.text)}"

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """
        Add documents to both vector and BM25 indexes.

        Args:
            texts: List of document texts
            metadatas: List of metadata dicts
            ids: List of unique document IDs
        """
        # Add to vector store
        self.vector_retriever.add_documents(texts, metadatas, ids)

        # Add to BM25 index
        self.bm25_retriever.add_documents(texts, metadatas, ids)

    def clear(self) -> None:
        """Clear both indexes."""
        self.vector_retriever.clear_collection()
        self.bm25_retriever.clear_index()

    def get_stats(self) -> dict:
        """Get statistics from both retrievers."""
        return {
            'vector': self.vector_retriever.get_collection_stats(),
            'bm25': self.bm25_retriever.get_stats(),
            'hybrid_config': {
                'enabled': self.settings.enable_hybrid_search,
                'vector_weight': self.vector_weight,
                'bm25_weight': self.bm25_weight,
                'rrf_k': self.rrf_k,
            }
        }

    def vector_only_search(
        self,
        query: str,
        top_k: int = None,
    ) -> list[RetrievalResult]:
        """Perform vector-only search (for comparison/fallback)."""
        if top_k is None:
            top_k = self.settings.retrieval_top_k
        return self.vector_retriever.search(query, top_k=top_k)

    def bm25_only_search(
        self,
        query: str,
        top_k: int = None,
    ) -> list[BM25Result]:
        """Perform BM25-only search (for comparison/fallback)."""
        if top_k is None:
            top_k = self.settings.retrieval_top_k
        return self.bm25_retriever.search(query, top_k=top_k)
