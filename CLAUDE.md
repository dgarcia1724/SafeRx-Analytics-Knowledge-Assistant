# SafeRx - Drug Safety RAG System

## Project Vision

A drug safety RAG (Retrieval-Augmented Generation) system demonstrating **measurable quality** through hybrid retrieval, comprehensive evaluation, and honest failure analysis.

**Goal**: Build a portfolio project that shows:
> "I improved retrieval recall from 72% to 89% with hybrid search. Here's my evaluation framework, here's where it fails, here's why I made these tradeoffs."

## Tech Stack

### Core
- **Backend**: FastAPI (Python 3.11+)
- **Frontend**: Next.js 15 with React 19
- **Vector DB**: ChromaDB (embedded, persisted)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **LLM**: OpenAI `gpt-4o-mini`

### Quality Enhancements
- **BM25 Search**: `rank-bm25` for keyword matching
- **Reranker**: Cohere Rerank API (free tier) or local cross-encoder
- **Evaluation**: Custom framework with RAGAS-inspired metrics
- **Observability**: Structured JSON traces

## Architecture

```
┌─────────────────┐     ┌─────────────────────────────────────────────────┐
│   Next.js UI    │────▶│                  FastAPI Backend                │
└─────────────────┘     │  ┌─────────────────────────────────────────┐   │
                        │  │           Hybrid Retriever               │   │
                        │  │  ┌──────────────┐  ┌──────────────┐     │   │
                        │  │  │ Vector Search │  │ BM25 Search  │     │   │
                        │  │  │  (ChromaDB)   │  │ (rank-bm25)  │     │   │
                        │  │  └──────────────┘  └──────────────┘     │   │
                        │  │           ↓ RRF Fusion ↓                 │   │
                        │  │         ┌──────────────┐                │   │
                        │  │         │   Reranker   │                │   │
                        │  │         └──────────────┘                │   │
                        │  └─────────────────────────────────────────┘   │
                        │                    ↓                           │
                        │  ┌─────────────────────────────────────────┐   │
                        │  │         Generator (GPT-4o-mini)         │   │
                        │  └─────────────────────────────────────────┘   │
                        └─────────────────────────────────────────────────┘
```

## Directory Structure

```
saferx/
├── CLAUDE.md                    # This file
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat.py          # Main chat endpoint
│   │   │   ├── search.py        # Debug search endpoint
│   │   │   └── traces.py        # Trace viewer endpoint
│   │   ├── services/
│   │   │   ├── retriever.py     # ChromaDB vector retriever
│   │   │   ├── bm25_retriever.py    # BM25 keyword retriever
│   │   │   ├── hybrid_retriever.py  # Fusion of vector + BM25
│   │   │   ├── reranker.py      # Cross-encoder reranking
│   │   │   ├── chunker.py       # Document chunking
│   │   │   ├── embedder.py      # OpenAI embeddings
│   │   │   ├── generator.py     # LLM response generation
│   │   │   └── tracer.py        # Query tracing/observability
│   │   ├── prompts/
│   │   │   └── system.py        # System prompts
│   │   ├── config.py            # Settings management
│   │   └── main.py              # FastAPI app
│   ├── scripts/
│   │   ├── download_fda.py      # FDA label downloader
│   │   └── ingest.py            # Document ingestion
│   ├── evaluation/
│   │   ├── test_cases.json      # Golden test set (100 cases)
│   │   ├── metrics.py           # RAG metrics calculations
│   │   ├── runner.py            # Evaluation test runner
│   │   └── report.py            # Report generation
│   ├── data/
│   │   ├── documents/
│   │   │   ├── drug_labels/     # FDA drug labels
│   │   │   ├── safety_alerts/   # Safety communications
│   │   │   └── internal/        # Synthetic internal docs
│   │   └── chroma_db/           # Vector DB persistence
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/                 # Next.js app router
│   │   ├── components/          # React components
│   │   └── lib/                 # API client
│   └── package.json
└── docker-compose.yml
```

## Key Concepts

### Hybrid Retrieval
- **Vector Search**: Semantic similarity via embeddings (catches paraphrases)
- **BM25 Search**: Keyword matching (catches exact drug names)
- **RRF Fusion**: Reciprocal Rank Fusion combines both result sets
- **Reranking**: Cross-encoder re-scores top candidates for precision

### Evaluation Metrics

**Retrieval:**
- `Recall@K`: Did we retrieve the relevant documents?
- `Precision@K`: Are retrieved documents relevant?
- `MRR`: Mean Reciprocal Rank - is best doc ranked first?

**Generation:**
- `Answer Relevance`: Does the answer address the query?
- `Faithfulness`: Is the answer grounded in retrieved context?
- `Citation Accuracy`: Are source references correct?

### Test Categories
1. **Drug Interactions** (20 cases): "Can I take warfarin with aspirin?"
2. **Side Effects** (20 cases): "What are common side effects of metformin?"
3. **Contraindications** (15 cases): "Who should not take lisinopril?"
4. **Dosing** (15 cases): "What's the max daily dose of ibuprofen?"
5. **Special Populations** (15 cases): "Is sertraline safe during pregnancy?"
6. **Out-of-Scope** (10 cases): Should refuse non-drug queries
7. **Edge Cases** (5 cases): Ambiguous queries, similar drug names

## Configuration

Key settings in `backend/app/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `chunk_size` | 500 | Tokens per chunk |
| `chunk_overlap` | 50 | Overlap between chunks |
| `top_k_results` | 5 | Final results returned |
| `retrieval_top_k` | 20 | Candidates before reranking |
| `bm25_weight` | 0.3 | Weight for BM25 in fusion |
| `vector_weight` | 0.7 | Weight for vector in fusion |
| `enable_reranking` | true | Toggle reranker |
| `reranker_model` | "cohere" | "cohere" or "cross-encoder" |

## Development Commands

```bash
# Backend Setup
cd backend
pip install -r requirements.txt

# Download FDA drug labels (25 drugs)
python scripts/download_fda.py

# Ingest documents into hybrid retrieval system (vector + BM25)
python scripts/ingest.py              # Full hybrid ingestion
python scripts/ingest.py --vector-only  # Vector store only
python scripts/ingest.py --bm25-only    # BM25 index only

# Start server
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev

# Evaluation
cd backend
python -m evaluation.runner           # Full evaluation (100 cases)
python -m evaluation.runner --quick   # Quick 10-case test
python -m evaluation.runner --compare # Compare retrieval methods
python -m evaluation.runner --no-rerank  # Test without reranking

# Generate reports
python -m evaluation.report evaluation_report_*.json -o report.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Main RAG chat endpoint (hybrid + reranking) |
| `/api/chat/debug` | POST | Chat with detailed debug info |
| `/api/chat/compare` | GET | Compare retrieval methods side-by-side |
| `/api/chat/stats` | GET | Pipeline statistics |
| `/api/search` | POST | Direct search (debugging) |
| `/api/traces` | GET | View recent query traces |
| `/api/traces/summary` | GET | Trace summary statistics |
| `/api/traces/{id}` | GET | Get specific trace |
| `/api/traces/{id}/scores` | GET | Get retrieval scores for trace |
| `/api/traces/{id}/timings` | GET | Get timing breakdown for trace |
| `/health` | GET | Health check with feature status |

## Data Sources

### Tier 1: FDA Drug Labels (25 drugs)
Real FDA-approved drug labels from openFDA API, parsed into clean sections:
- Indications, Dosage, Contraindications, Interactions, Adverse Reactions, Warnings, Special Populations

### Tier 2: Synthetic Internal Docs (5-10 docs)
Realistic internal policy documents for testing:
- Formulary Guidelines, Opioid Policy, Pediatric Dosing, Drug Shortage Alternatives, High-Alert Medications

### Tier 3: Golden Test Set (100 cases)
Curated Q&A pairs with expected answers and sources for comprehensive evaluation.

## Tracing

Every query generates a trace with:
- Unique trace ID
- Query text and timestamp
- Retrieval scores (vector + BM25)
- Reranking scores
- Step-by-step latencies
- Token usage
- Final response

Access via `/api/traces` endpoint or logs.

## Known Limitations

Document failure modes here after running evaluation:
- [ ] Retrieval failures (patterns)
- [ ] Ranking failures (patterns)
- [ ] Generation failures (patterns)

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...  # Optional, for Cohere reranking
```

## Interview Talking Points

1. **Hybrid Search**: "BM25 catches keyword matches that embeddings miss. I saw X% recall improvement on drug name queries. Use `/api/chat/compare` to see the difference."

2. **Reranking**: "Cross-encoders are slower but more accurate. I retrieve 20 candidates, rerank to 5. Precision went from X% to Y%."

3. **Evaluation**: "I have 100 test cases across 7 categories. I can show you exactly where the system fails with `python -m evaluation.runner`."

4. **Tradeoffs**: "I chose chunk size 500 because FDA labels have natural section breaks around that length. I tuned fusion weights (0.3 BM25 / 0.7 vector) by running A/B evaluations."

5. **Failure Analysis**: "X% of failures are retrieval issues (document not found), Y% are generation issues (answer missing key information). The retrieval failures cluster around ambiguous drug name queries."

6. **Observability**: "Every query is traced with `/api/traces`. I can see exactly which documents were retrieved, their scores from each retriever, and how reranking changed the order."
