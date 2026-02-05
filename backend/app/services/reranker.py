"""
Reranker Service for improving retrieval precision.

Cross-encoder reranking provides more accurate relevance scoring than
bi-encoder (embedding) approaches by considering query-document pairs
jointly. This is slower but significantly improves precision.

Supports:
- Cohere Rerank API (cloud, free tier available)
- Local cross-encoder models (sentence-transformers)
"""

from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.services.hybrid_retriever import HybridResult


@dataclass
class RerankResult:
    """Result from reranking with updated scores."""
    original_result: HybridResult
    rerank_score: float
    rerank_rank: int


class RerankerService:
    """
    Service for reranking retrieval results using cross-encoders.

    The reranking pipeline:
    1. Take top N candidates from hybrid retrieval
    2. Score each (query, document) pair with cross-encoder
    3. Re-sort by cross-encoder scores
    4. Return top K results
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_type = self.settings.reranker_model

        # Lazy initialization of models
        self._cohere_client = None
        self._cross_encoder = None

    @property
    def cohere_client(self):
        """Lazy initialization of Cohere client."""
        if self._cohere_client is None:
            if not self.settings.cohere_api_key:
                raise ValueError(
                    "Cohere API key not configured. Set COHERE_API_KEY in .env "
                    "or use cross-encoder reranker instead."
                )
            import cohere
            self._cohere_client = cohere.Client(self.settings.cohere_api_key)
        return self._cohere_client

    @property
    def cross_encoder(self):
        """Lazy initialization of cross-encoder model."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(
                self.settings.cross_encoder_model,
                max_length=512,
            )
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        top_k: int = None,
        model: str = None,
    ) -> list[HybridResult]:
        """
        Rerank results using cross-encoder scoring.

        Args:
            query: Original search query
            results: List of HybridResult from retrieval
            top_k: Number of results to return after reranking
            model: Override reranker model ('cohere' or 'cross-encoder')

        Returns:
            List of HybridResult with updated rerank_score, sorted by rerank score
        """
        if not results:
            return []

        if top_k is None:
            top_k = self.settings.top_k_results

        if model is None:
            model = self.model_type

        # Get rerank scores
        if model == "cohere":
            scored_results = self._rerank_cohere(query, results)
        else:
            scored_results = self._rerank_cross_encoder(query, results)

        # Sort by rerank score (descending)
        scored_results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Update ranks and return top_k
        for i, result in enumerate(scored_results):
            result.rerank_rank = i + 1

        return scored_results[:top_k]

    def _rerank_cohere(
        self,
        query: str,
        results: list[HybridResult],
    ) -> list[RerankResult]:
        """Rerank using Cohere Rerank API."""
        documents = [r.text for r in results]

        try:
            response = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=len(documents),  # Get scores for all documents
                return_documents=False,
            )

            # Create scored results
            scored_results = []
            for rerank_result in response.results:
                idx = rerank_result.index
                original = results[idx]

                # Update the original result with rerank score
                original.rerank_score = rerank_result.relevance_score

                scored_results.append(RerankResult(
                    original_result=original,
                    rerank_score=rerank_result.relevance_score,
                    rerank_rank=0,  # Will be set after sorting
                ))

            return [r.original_result for r in scored_results]

        except Exception as e:
            print(f"Cohere rerank failed: {e}. Falling back to original scores.")
            return results

    def _rerank_cross_encoder(
        self,
        query: str,
        results: list[HybridResult],
    ) -> list[HybridResult]:
        """Rerank using local cross-encoder model."""
        # Create query-document pairs
        pairs = [(query, r.text) for r in results]

        try:
            # Get scores from cross-encoder
            scores = self.cross_encoder.predict(pairs)

            # Update results with rerank scores
            for result, score in zip(results, scores):
                result.rerank_score = float(score)

            return results

        except Exception as e:
            print(f"Cross-encoder rerank failed: {e}. Falling back to original scores.")
            return results

    def is_available(self) -> bool:
        """Check if reranking is available with current configuration."""
        if self.model_type == "cohere":
            return bool(self.settings.cohere_api_key)
        else:
            # Cross-encoder is always available (will download model if needed)
            return True

    def get_model_info(self) -> dict:
        """Get information about the configured reranker."""
        return {
            'model_type': self.model_type,
            'cohere_available': bool(self.settings.cohere_api_key),
            'cross_encoder_model': self.settings.cross_encoder_model,
            'enabled': self.settings.enable_reranking,
        }


class RerankerPipeline:
    """
    High-level pipeline combining hybrid retrieval with reranking.

    This is the main entry point for the full retrieval pipeline:
    1. Hybrid search (vector + BM25)
    2. RRF fusion
    3. Cross-encoder reranking
    4. Return final results
    """

    def __init__(self):
        from app.services.hybrid_retriever import HybridRetrieverService

        self.settings = get_settings()
        self.hybrid_retriever = HybridRetrieverService()
        self.reranker = RerankerService()

    def search(
        self,
        query: str,
        top_k: int = None,
        enable_reranking: bool = None,
    ) -> tuple[list[HybridResult], dict]:
        """
        Perform full retrieval pipeline with optional reranking.

        Args:
            query: Search query
            top_k: Final number of results to return
            enable_reranking: Override reranking setting

        Returns:
            Tuple of (results, trace_info)
        """
        if top_k is None:
            top_k = self.settings.top_k_results

        if enable_reranking is None:
            enable_reranking = self.settings.enable_reranking

        # Step 1: Hybrid retrieval (gets more candidates than final top_k)
        candidates, retrieval_trace = self.hybrid_retriever.search(
            query=query,
            top_k=self.settings.retrieval_top_k,
        )

        trace_info = {
            'query': query,
            'hybrid_candidates': len(candidates),
            'vector_results': retrieval_trace.vector_results_count,
            'bm25_results': retrieval_trace.bm25_results_count,
            'fusion_weights': retrieval_trace.fusion_weights,
            'reranking_enabled': enable_reranking,
            'reranker_model': self.reranker.model_type if enable_reranking else None,
        }

        # Step 2: Reranking (if enabled)
        if enable_reranking and candidates:
            try:
                results = self.reranker.rerank(
                    query=query,
                    results=candidates,
                    top_k=top_k,
                )
                trace_info['reranked'] = True
                trace_info['reranked_count'] = len(results)
            except Exception as e:
                print(f"Reranking failed: {e}. Using fusion scores.")
                results = candidates[:top_k]
                trace_info['reranked'] = False
                trace_info['rerank_error'] = str(e)
        else:
            results = candidates[:top_k]
            trace_info['reranked'] = False

        trace_info['final_results'] = len(results)

        return results, trace_info

    def get_stats(self) -> dict:
        """Get statistics about the pipeline configuration."""
        return {
            'hybrid': self.hybrid_retriever.get_stats(),
            'reranker': self.reranker.get_model_info(),
            'pipeline_config': {
                'retrieval_top_k': self.settings.retrieval_top_k,
                'final_top_k': self.settings.top_k_results,
                'reranking_enabled': self.settings.enable_reranking,
            }
        }
