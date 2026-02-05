"""
Ingest documents into both ChromaDB (vector) and BM25 index for hybrid retrieval.

Usage:
    python scripts/ingest.py              # Ingest all documents
    python scripts/ingest.py --vector-only  # Only update vector store
    python scripts/ingest.py --bm25-only    # Only update BM25 index
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.chunker import DocumentChunker
from app.services.hybrid_retriever import HybridRetrieverService
from app.services.retriever import RetrieverService
from app.services.bm25_retriever import BM25RetrieverService


# Document directories
DATA_DIR = Path(__file__).parent.parent / "data" / "documents"
DRUG_LABELS_DIR = DATA_DIR / "drug_labels"
SAFETY_ALERTS_DIR = DATA_DIR / "safety_alerts"
INTERNAL_DIR = DATA_DIR / "internal"

# Also check root data directory for legacy files
ROOT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_text_files(directory: Path) -> list[tuple[Path, str]]:
    """Load all .txt files from a directory."""
    files = []
    if directory.exists():
        for filepath in directory.glob("*.txt"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                files.append((filepath, content))
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    return files


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into SafeRx knowledge base"
    )
    parser.add_argument(
        "--vector-only",
        action="store_true",
        help="Only update vector store (ChromaDB)",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Only update BM25 index",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing data before ingestion",
    )
    return parser.parse_args()


def main():
    """Ingest all documents into hybrid retrieval system."""
    args = parse_args()

    print("SafeRx Analytics - Document Ingestion (Hybrid)")
    print("=" * 60)

    # Initialize services based on mode
    chunker = DocumentChunker()

    if args.vector_only:
        print("Mode: Vector-only (ChromaDB)")
        retriever = RetrieverService()
        bm25_retriever = None
    elif args.bm25_only:
        print("Mode: BM25-only")
        retriever = None
        bm25_retriever = BM25RetrieverService()
    else:
        print("Mode: Hybrid (ChromaDB + BM25)")
        # Use hybrid retriever for both indexes
        hybrid = HybridRetrieverService()
        retriever = hybrid.vector_retriever
        bm25_retriever = hybrid.bm25_retriever

    # Clear existing data unless --no-clear specified
    if not args.no_clear:
        print("\nClearing existing indexes...")
        if retriever:
            retriever.clear_collection()
            print("  - Vector store cleared")
        if bm25_retriever:
            bm25_retriever.clear_index()
            print("  - BM25 index cleared")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    # Process drug labels
    print(f"\nProcessing drug labels from: {DRUG_LABELS_DIR}")
    drug_files = load_text_files(DRUG_LABELS_DIR)
    print(f"Found {len(drug_files)} drug label files")

    for filepath, content in drug_files:
        doc_id = filepath.stem  # filename without extension
        doc_title = doc_id.replace("_label", "").replace("_", " ").title()

        chunks = chunker.chunk_document(
            text=content,
            doc_id=doc_id,
            doc_title=f"{doc_title} - FDA Drug Label",
            source_url=f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={doc_title}",
        )

        for chunk in chunks:
            chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"
            all_chunks.append(chunk.text)
            all_metadatas.append(chunk.metadata)
            all_ids.append(chunk_id)

        print(f"  [+] {doc_title}: {len(chunks)} chunks")

    # Process safety alerts
    print(f"\nProcessing safety alerts from: {SAFETY_ALERTS_DIR}")
    alert_files = load_text_files(SAFETY_ALERTS_DIR)
    print(f"Found {len(alert_files)} safety alert files")

    for filepath, content in alert_files:
        doc_id = filepath.stem
        doc_title = doc_id.replace("_", " ").title()

        chunks = chunker.chunk_document(
            text=content,
            doc_id=doc_id,
            doc_title=f"{doc_title} - FDA Safety Alert",
            source_url="https://www.fda.gov/safety/medwatch-fda-safety-information-and-adverse-event-reporting-program",
        )

        for chunk in chunks:
            chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"
            all_chunks.append(chunk.text)
            all_metadatas.append(chunk.metadata)
            all_ids.append(chunk_id)

        print(f"  [+] {doc_title}: {len(chunks)} chunks")

    # Process internal documents
    print(f"\nProcessing internal documents from: {INTERNAL_DIR}")
    internal_files = load_text_files(INTERNAL_DIR)
    print(f"Found {len(internal_files)} internal document files")

    for filepath, content in internal_files:
        doc_id = filepath.stem
        doc_title = doc_id.replace("_", " ").title()

        chunks = chunker.chunk_document(
            text=content,
            doc_id=doc_id,
            doc_title=f"{doc_title} - Internal Document",
            source_url="",
        )

        for chunk in chunks:
            chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"
            all_chunks.append(chunk.text)
            all_metadatas.append(chunk.metadata)
            all_ids.append(chunk_id)

        print(f"  [+] {doc_title}: {len(chunks)} chunks")

    # Also check root data directory for legacy simple format files
    if ROOT_DATA_DIR.exists():
        print(f"\nChecking root data directory: {ROOT_DATA_DIR}")
        root_files = load_text_files(ROOT_DATA_DIR)
        if root_files:
            print(f"Found {len(root_files)} legacy files")
            for filepath, content in root_files:
                doc_id = filepath.stem
                doc_title = doc_id.replace("_", " ").title()

                chunks = chunker.chunk_document(
                    text=content,
                    doc_id=doc_id,
                    doc_title=f"{doc_title}",
                    source_url="",
                )

                for chunk in chunks:
                    chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"
                    all_chunks.append(chunk.text)
                    all_metadatas.append(chunk.metadata)
                    all_ids.append(chunk_id)

                print(f"  [+] {doc_title}: {len(chunks)} chunks")

    # Ingest all chunks
    print(f"\n" + "=" * 60)
    print(f"Total chunks to ingest: {len(all_chunks)}")

    if all_chunks:
        # Process in batches to avoid memory issues
        batch_size = 100

        if retriever:
            print(f"\nIngesting into vector store (ChromaDB)...")
            for i in range(0, len(all_chunks), batch_size):
                batch_end = min(i + batch_size, len(all_chunks))
                retriever.add_documents(
                    texts=all_chunks[i:batch_end],
                    metadatas=all_metadatas[i:batch_end],
                    ids=all_ids[i:batch_end],
                )
                print(f"  Batch {i // batch_size + 1}/{(len(all_chunks) - 1) // batch_size + 1} complete")

        if bm25_retriever:
            print(f"\nIngesting into BM25 index...")
            for i in range(0, len(all_chunks), batch_size):
                batch_end = min(i + batch_size, len(all_chunks))
                bm25_retriever.add_documents(
                    texts=all_chunks[i:batch_end],
                    metadatas=all_metadatas[i:batch_end],
                    ids=all_ids[i:batch_end],
                )
                print(f"  Batch {i // batch_size + 1}/{(len(all_chunks) - 1) // batch_size + 1} complete")

    # Print final stats
    print("\n" + "=" * 60)
    print("Ingestion Complete!")

    if retriever:
        vector_stats = retriever.get_collection_stats()
        print(f"\nVector Store (ChromaDB):")
        print(f"  Collection: {vector_stats['name']}")
        print(f"  Total chunks: {vector_stats['count']}")

    if bm25_retriever:
        bm25_stats = bm25_retriever.get_stats()
        print(f"\nBM25 Index:")
        print(f"  Document count: {bm25_stats['document_count']}")
        print(f"  Index path: {bm25_stats['index_path']}")
        print(f"  Avg doc length: {bm25_stats['avg_doc_length']:.1f} tokens")

    print("\nReady for hybrid retrieval!")


if __name__ == "__main__":
    main()
