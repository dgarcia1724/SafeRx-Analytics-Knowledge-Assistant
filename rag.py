"""
Simple RAG (Retrieval-Augmented Generation) Demo
A minimal implementation to learn RAG fundamentals.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# =====================
# SECTION 1: SETUP
# =====================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATA_DIR = Path(__file__).parent / "data"

# =====================
# SECTION 2: CHUNKING
# =====================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks

def load_documents() -> list[dict]:
    """Load all .txt files from data directory and chunk them."""
    documents = []
    for filepath in DATA_DIR.glob("*.txt"):
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{filepath.stem}_chunk_{i}",
                "text": chunk,
                "source": filepath.name,
                "chunk_index": i
            })
    return documents

# =====================
# SECTION 3: EMBEDDINGS
# =====================

def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a piece of text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts in one API call."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

# =====================
# SECTION 4: VECTOR STORE
# =====================

def create_vector_store(documents: list[dict]) -> chromadb.Collection:
    """Create ChromaDB collection and add documents."""
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="drug_info")

    # Get embeddings for all documents
    texts = [doc["text"] for doc in documents]
    embeddings = get_embeddings_batch(texts)

    # Add to ChromaDB
    collection.add(
        ids=[doc["id"] for doc in documents],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": doc["source"], "chunk_index": doc["chunk_index"]} for doc in documents]
    )
    return collection

def search_similar(collection: chromadb.Collection, query: str, n_results: int = 3) -> list[dict]:
    """Search for chunks similar to the query."""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "score": 1 - results["distances"][0][i]  # Convert distance to similarity
        })
    return matches

# =====================
# SECTION 5: GENERATION
# =====================

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate an answer using GPT-4o-mini with retrieved context."""
    context = "\n\n---\n\n".join([
        f"[Source: {chunk['source']}]\n{chunk['text']}"
        for chunk in context_chunks
    ])

    prompt = f"""You are a helpful medical information assistant. Answer the question based ONLY on the provided context. If the context doesn't contain enough information, say so.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

# =====================
# SECTION 6: RAG PIPELINE
# =====================

def rag_query(collection: chromadb.Collection, query: str) -> tuple[str, list[dict]]:
    """Complete RAG pipeline: retrieve relevant chunks and generate answer."""
    print("\n[Searching for relevant chunks...]")
    chunks = search_similar(collection, query, n_results=3)

    scores = [f"{c['score']:.2f}" for c in chunks]
    print(f"Found {len(chunks)} relevant chunks (scores: {', '.join(scores)})")

    print("\n[Generating answer...]")
    answer = generate_answer(query, chunks)

    return answer, chunks

# =====================
# SECTION 7: MAIN
# =====================

def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"✓ Loaded {len(set(d['source'] for d in documents))} documents")
    print(f"✓ Created {len(documents)} chunks")

    print("✓ Generating embeddings...")
    collection = create_vector_store(documents)
    print("✓ Stored in ChromaDB")

    print("\nReady! Ask a question (or 'quit' to exit):\n")

    while True:
        query = input("> ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer, chunks = rag_query(collection, query)

        print(f"\nAnswer: {answer}\n")
        print("Sources used:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. {chunk['source']} (chunk {chunk['chunk_index']}) - score: {chunk['score']:.2f}")
        print()

if __name__ == "__main__":
    main()
