# SafeRx Analytics - Drug Safety Knowledge Assistant

AI-powered drug safety assistant that helps pharmacovigilance analysts find drug interactions, side effects, and safety alerts instantly.

## Overview

SafeRx Analytics is a RAG (Retrieval-Augmented Generation) application built from scratch without LangChain or LlamaIndex. It demonstrates advanced RAG techniques including **hybrid retrieval**, **reranking**, and **comprehensive evaluation**.

### Key Features

- **Hybrid Retrieval**: Combines vector search (ChromaDB) + keyword search (BM25) with Reciprocal Rank Fusion
- **Streaming Responses**: Real-time token streaming via Server-Sent Events
- **Cross-Encoder Reranking**: Optional precision boost with sentence-transformers
- **Query Tracing**: Full observability with timing breakdowns and retrieval scores
- **Evaluation Framework**: 100 test cases with MRR, Recall@K, Precision@K metrics

### Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend | FastAPI | Async API server |
| Frontend | Next.js 15 | React-based chat interface |
| Vector DB | ChromaDB | Semantic similarity search |
| Keyword Search | rank-bm25 | Exact term matching |
| Embeddings | OpenAI text-embedding-3-small | Document/query embeddings |
| LLM | OpenAI GPT-4o-mini | Response generation |
| Reranker | sentence-transformers | Cross-encoder reranking |

## Evaluation Results

| Metric | Score | Target |
|--------|-------|--------|
| **MRR** | 0.942 | > 0.70 |
| **Recall@5** | 96.3% | > 85% |
| **Precision@5** | 73.6% | > 75% |
| **Keyword Match** | 54% | > 70% |

Retrieval is excellent (96% recall). Most failures are in generation (answer phrasing), not document retrieval.

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- OpenAI API key

### Backend Setup

```bash
cd backend

# Install dependencies
python -m pip install -r requirements.txt

# Configure environment
echo "OPENAI_API_KEY=your-key-here" > .env

# Download FDA drug labels (25 drugs)
python scripts/download_fda.py

# Ingest documents into ChromaDB + BM25
python scripts/ingest.py

# Start the server
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at http://localhost:3000

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│   FastAPI       │────▶│   OpenAI API    │
│   (Port 3000)   │     │   (Port 8000)   │     │   GPT-4o-mini   │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌──────────────┐          ┌──────────────┐
            │   ChromaDB   │          │  BM25 Index  │
            │   (Vectors)  │          │  (Keywords)  │
            └──────────────┘          └──────────────┘
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                    ┌──────────────────────┐
                    │   RRF Fusion         │
                    │   (70% vec, 30% bm25)│
                    └──────────────────────┘
```

## RAG Pipeline

### 1. Document Ingestion
- Load 25 FDA drug labels + 5 internal policy documents
- Chunk documents (500 tokens, 50 token overlap)
- Generate embeddings via OpenAI
- Store in ChromaDB (vectors) and BM25 index (keywords)

### 2. Hybrid Retrieval
- Embed user query → search ChromaDB
- Tokenize query → search BM25 index
- Combine rankings with Reciprocal Rank Fusion (RRF)
- Optional: Rerank top 20 candidates with cross-encoder

### 3. Response Generation
- Build context from top 5 retrieved chunks
- Stream response with GPT-4o-mini via SSE
- Include citations to source documents

## API Endpoints

### Chat (Standard)
```
POST /api/chat
{
  "query": "What are the side effects of metformin?",
  "conversation_id": "optional-uuid"
}
```

### Chat (Streaming)
```
POST /api/chat/stream
{
  "query": "What are the side effects of metformin?"
}

Response: Server-Sent Events
- event: sources (retrieved documents)
- event: chunk (streamed tokens)
- event: done (completion)
```

### Chat (Debug)
```
POST /api/chat/debug
Returns full retrieval scores, timings, and pipeline info
```

### Compare Retrieval Methods
```
GET /api/chat/compare?query=warfarin+interactions
Returns side-by-side: vector-only, BM25-only, hybrid, hybrid+reranked
```

### Query Traces
```
GET /api/traces              # List recent traces
GET /api/traces/{trace_id}   # Get specific trace
```

### Health Check
```
GET /health
```

## Project Structure

```
saferx/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI endpoints
│   │   │   ├── chat.py       # Chat + streaming endpoints
│   │   │   └── traces.py     # Trace viewer API
│   │   ├── services/
│   │   │   ├── retriever.py      # Vector retrieval
│   │   │   ├── bm25_retriever.py # Keyword retrieval
│   │   │   ├── hybrid_retriever.py # RRF fusion
│   │   │   ├── reranker.py       # Cross-encoder reranking
│   │   │   ├── generator.py      # LLM generation + streaming
│   │   │   └── tracer.py         # Query observability
│   │   └── prompts/          # System prompts
│   ├── scripts/
│   │   ├── download_fda.py   # Fetch FDA drug labels
│   │   └── ingest.py         # Build vector + BM25 indexes
│   ├── evaluation/
│   │   ├── test_cases.json   # 100 test cases
│   │   ├── runner.py         # Evaluation runner
│   │   ├── metrics.py        # MRR, Recall, Precision, NDCG
│   │   └── report.py         # Generate markdown reports
│   └── data/
│       ├── documents/        # FDA labels + internal docs
│       ├── chroma_db/        # Vector store
│       └── bm25_index.pkl    # Keyword index
├── frontend/
│   └── src/
│       ├── app/              # Next.js pages
│       ├── components/       # React components (streaming chat)
│       └── lib/              # API client with SSE support
└── docker-compose.yml
```

## Configuration

### Backend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENAI_API_KEY | OpenAI API key | Required |
| COHERE_API_KEY | Cohere API key (for reranking) | Optional |
| CHROMA_PERSIST_DIR | ChromaDB storage path | ./data/chroma_db |
| BM25_INDEX_PATH | BM25 index path | ./data/bm25_index.pkl |
| EMBEDDING_MODEL | Embedding model | text-embedding-3-small |
| LLM_MODEL | LLM model | gpt-4o-mini |
| ENABLE_HYBRID_SEARCH | Use hybrid retrieval | true |
| BM25_WEIGHT | BM25 weight in fusion | 0.3 |
| VECTOR_WEIGHT | Vector weight in fusion | 0.7 |
| ENABLE_RERANKING | Use cross-encoder reranking | false |
| ENABLE_TRACING | Log query traces | true |

## Drug Labels Included

25 FDA drug labels covering common therapeutic classes:

**Pain/Inflammation:** aspirin, ibuprofen, acetaminophen, naproxen

**Cardiovascular:** lisinopril, metoprolol, amlodipine, warfarin, atorvastatin, losartan, hydrochlorothiazide, furosemide

**Diabetes:** metformin, glipizide

**Mental Health:** sertraline, fluoxetine, alprazolam, gabapentin

**Antibiotics:** amoxicillin, azithromycin, ciprofloxacin

**Other:** omeprazole, levothyroxine, prednisone, albuterol

## Internal Policy Documents

5 synthetic documents for testing internal knowledge retrieval:

- SafeRx Formulary Guidelines 2024
- Opioid Prescribing Policy
- Pediatric Dosing Protocols
- Drug Shortage Alternatives List
- High-Alert Medications Checklist

## Evaluation

Run the full evaluation suite (100 test cases):

```bash
cd backend
python -m evaluation.runner
```

Quick evaluation (10 sampled cases):
```bash
python -m evaluation.runner --quick
```

Compare retrieval methods:
```bash
python -m evaluation.runner --compare
```

### Test Case Categories

| Category | Count | Description |
|----------|-------|-------------|
| Drug Interactions | 20 | "Can I take X with Y?" |
| Adverse Reactions | 20 | "What are side effects of X?" |
| Contraindications | 15 | "Who should not take X?" |
| Dosage | 15 | "What is the dose of X?" |
| Special Populations | 15 | "Is X safe during pregnancy?" |
| Out-of-Scope | 10 | Should refuse (not drug-related) |
| Edge Cases | 5 | Ambiguous or tricky queries |

## Example Queries

- "What are the side effects of metformin?"
- "Does warfarin interact with aspirin?"
- "What are the contraindications for lisinopril?"
- "Can pregnant women take sertraline?"
- "What is the recommended dosage for atorvastatin?"
- "What is our therapeutic substitution policy?" (internal doc)

## Interview Talking Points

1. **Hybrid Search**: "BM25 catches keyword matches that embeddings miss. I saw improved recall on drug name queries by combining both."

2. **Evaluation**: "I have 100 test cases across 7 categories. Retrieval achieves 96% recall. Most failures are generation issues, not retrieval."

3. **Streaming**: "Server-Sent Events provide real-time token streaming. Sources are sent first, then the response streams word by word."

4. **Tradeoffs**: "I chose RRF over linear fusion because it's more robust to score distribution differences between retrievers."

## License

This project is for educational and portfolio purposes.

## Disclaimer

This tool is for informational purposes only. Always consult a healthcare professional for medical advice.
