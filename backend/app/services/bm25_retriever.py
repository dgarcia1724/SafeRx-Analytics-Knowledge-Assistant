"""
BM25 Retriever Service for keyword-based search.

BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks
documents based on query terms appearing in each document.
"""

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from app.config import get_settings


@dataclass
class BM25Document:
    """Represents a document stored in the BM25 index."""
    doc_id: str
    text: str
    metadata: dict


@dataclass
class BM25Result:
    """Represents a BM25 search result."""
    doc_id: str
    text: str
    metadata: dict
    score: float
    rank: int


class BM25RetrieverService:
    """
    Service for BM25-based keyword search.

    BM25 is particularly good at:
    - Exact keyword matches (drug names, dosages)
    - Handling rare/specific terms
    - Cases where semantic similarity might miss exact matches
    """

    def __init__(self):
        self.settings = get_settings()
        self.index_path = Path(self.settings.bm25_index_path)

        self.documents: list[BM25Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: list[list[str]] = []

        # Try to load existing index
        self._load_index()

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing.

        Uses simple whitespace + punctuation splitting, lowercasing,
        and basic normalization. For drug safety, we preserve numbers
        (important for dosages) and keep hyphenated words together.
        """
        # Lowercase
        text = text.lower()

        # Replace common separators with spaces
        text = re.sub(r'[/\\|]', ' ', text)

        # Keep alphanumeric, hyphens, and periods (for dosages like "500mg" or "0.5")
        text = re.sub(r'[^\w\s\-\.]', ' ', text)

        # Split on whitespace
        tokens = text.split()

        # Filter empty tokens and very short tokens (except numbers)
        tokens = [t.strip('.-') for t in tokens]
        tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]

        return tokens

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """
        Add documents to the BM25 index.

        Args:
            texts: List of document texts
            metadatas: List of metadata dicts for each document
            ids: List of unique document IDs
        """
        if not texts:
            return

        # Add to documents list
        for text, metadata, doc_id in zip(texts, metadatas, ids):
            self.documents.append(BM25Document(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
            ))
            self.tokenized_corpus.append(self._tokenize(text))

        # Rebuild BM25 index
        self._rebuild_index()

        # Save to disk
        self._save_index()

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from the tokenized corpus."""
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[BM25Result]:
        """
        Search for documents using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of BM25Result objects sorted by score (descending)
        """
        if not self.bm25 or not self.documents:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Create (index, score) pairs and sort by score
        scored_indices = [(i, score) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        # Take top_k results
        results = []
        for rank, (idx, score) in enumerate(scored_indices[:top_k]):
            if score > 0:  # Only include documents with positive scores
                doc = self.documents[idx]
                results.append(BM25Result(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=score,
                    rank=rank + 1,
                ))

        return results

    def _save_index(self) -> None:
        """Save the BM25 index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'documents': self.documents,
            'tokenized_corpus': self.tokenized_corpus,
        }

        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_index(self) -> None:
        """Load the BM25 index from disk if it exists."""
        if not self.index_path.exists():
            return

        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)

            self.documents = data.get('documents', [])
            self.tokenized_corpus = data.get('tokenized_corpus', [])

            # Rebuild BM25 from loaded corpus
            self._rebuild_index()

        except Exception as e:
            print(f"Warning: Could not load BM25 index: {e}")
            self.documents = []
            self.tokenized_corpus = []
            self.bm25 = None

    def clear_index(self) -> None:
        """Clear the BM25 index."""
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

        # Remove persisted index
        if self.index_path.exists():
            self.index_path.unlink()

    def get_stats(self) -> dict:
        """Get statistics about the BM25 index."""
        return {
            'document_count': len(self.documents),
            'index_loaded': self.bm25 is not None,
            'index_path': str(self.index_path),
            'avg_doc_length': (
                sum(len(tokens) for tokens in self.tokenized_corpus) / len(self.tokenized_corpus)
                if self.tokenized_corpus else 0
            ),
        }

    def get_document_by_id(self, doc_id: str) -> Optional[BM25Document]:
        """Get a document by its ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None
