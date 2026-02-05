import chromadb
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.services.embedder import EmbeddingService


@dataclass
class RetrievalResult:
    """Represents a retrieval result with text, metadata, and score."""

    text: str
    metadata: dict
    score: float


class RetrieverService:
    """Service for retrieving relevant documents from ChromaDB."""

    def __init__(self):
        settings = get_settings()
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection_name = settings.collection_name
        self.top_k = settings.top_k_results
        self.embedder = EmbeddingService()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """Add documents to the collection."""
        if not texts:
            return

        # Generate embeddings
        embeddings = self.embedder.embed_texts(texts)

        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search for relevant documents."""
        if top_k is None:
            top_k = self.top_k

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }

        if filter_metadata:
            query_params["where"] = filter_metadata

        # Search
        results = self.collection.query(**query_params)

        # Convert to RetrievalResult objects
        retrieval_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                # ChromaDB returns distances, convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Convert cosine distance to similarity

                retrieval_results.append(
                    RetrievalResult(
                        text=doc,
                        metadata=metadata,
                        score=score,
                    )
                )

        return retrieval_results

    def search_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search using a pre-computed embedding."""
        if top_k is None:
            top_k = self.top_k

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }

        if filter_metadata:
            query_params["where"] = filter_metadata

        results = self.collection.query(**query_params)

        retrieval_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance

                retrieval_results.append(
                    RetrievalResult(
                        text=doc,
                        metadata=metadata,
                        score=score,
                    )
                )

        return retrieval_results

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
