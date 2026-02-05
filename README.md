# SafeRx Analytics - Drug Safety Knowledge Assistant

AI-powered drug safety assistant that helps pharmacovigilance analysts find drug interactions, side effects, and safety alerts instantly.

## Overview

SafeRx Analytics is a RAG (Retrieval-Augmented Generation) application built from scratch without LangChain or LlamaIndex. It demonstrates core RAG fundamentals: document chunking, embeddings, vector search, and LLM-based generation.

### Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend | FastAPI | Async API server |
| Frontend | Next.js 15 | React-based chat interface |
| Vector DB | ChromaDB | Document storage and retrieval |
| Embeddings | OpenAI text-embedding-3-small | Semantic search |
| LLM | OpenAI GPT-4o-mini | Response generation |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- OpenAI API key

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Download FDA drug labels
python scripts/download_fda.py

# Ingest documents into ChromaDB
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

# Configure environment
cp .env.example .env.local

# Start the development server
npm run dev
```

The frontend will be available at http://localhost:3000

## Project Structure

```
saferx/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI endpoints
│   │   ├── services/      # RAG pipeline services
│   │   └── prompts/       # System prompts
│   ├── scripts/           # Data ingestion scripts
│   ├── evaluation/        # Evaluation framework
│   └── data/              # Document storage
├── frontend/
│   └── src/
│       ├── app/           # Next.js pages
│       ├── components/    # React components
│       └── lib/           # API client
└── docker-compose.yml
```

## API Endpoints

### Chat
```
POST /api/chat
{
  "query": "What are the side effects of metformin?",
  "conversation_id": "optional-uuid"
}
```

### Search (Retrieval Only)
```
POST /api/search
{
  "query": "warfarin interactions",
  "top_k": 5
}
```

### Health Check
```
GET /health
```

## RAG Pipeline

### 1. Document Ingestion
- Load FDA drug labels (text files)
- Chunk documents (500 tokens, 50 token overlap)
- Generate embeddings via OpenAI
- Store in ChromaDB

### 2. Query Processing
- Embed user query
- Search ChromaDB for similar chunks
- Return top 5 results

### 3. Response Generation
- Build context from retrieved chunks
- Generate response with GPT-4o-mini
- Include citations to source documents

## Evaluation

Run the evaluation suite:

```bash
cd backend
python evaluation/evaluate.py
```

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| MRR | Mean Reciprocal Rank | > 0.7 |
| Recall@5 | Relevant docs in top 5 | > 0.8 |
| Keyword Match | Expected keywords in answer | > 0.7 |

## Configuration

### Backend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENAI_API_KEY | OpenAI API key | Required |
| CHROMA_PERSIST_DIR | ChromaDB storage path | ./data/chroma_db |
| EMBEDDING_MODEL | Embedding model | text-embedding-3-small |
| LLM_MODEL | LLM model | gpt-4o-mini |
| CHUNK_SIZE | Tokens per chunk | 500 |
| CHUNK_OVERLAP | Overlap tokens | 50 |
| TOP_K_RESULTS | Search results count | 5 |

### Frontend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NEXT_PUBLIC_API_URL | Backend API URL | http://localhost:8000 |

## Docker Deployment

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-your-key

# Build and run
docker-compose up --build
```

## Drug Labels Included

The system includes FDA drug labels for 20 commonly prescribed medications:

- Lisinopril, Atorvastatin, Metformin, Amlodipine, Metoprolol
- Omeprazole, Simvastatin, Losartan, Gabapentin, Hydrochlorothiazide
- Sertraline, Levothyroxine, Warfarin, Aspirin, Prednisone
- Furosemide, Clopidogrel, Fluoxetine, Tramadol, Alprazolam

## Example Queries

- "What are the side effects of metformin?"
- "Does warfarin interact with aspirin?"
- "What are the contraindications for lisinopril?"
- "Can pregnant women take sertraline?"
- "What is the recommended dosage for atorvastatin?"

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│  Frontend   │────▶│   Backend   │
│             │     │  (Next.js)  │     │  (FastAPI)  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
             ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
             │   OpenAI    │           │  ChromaDB   │           │   OpenAI    │
             │ Embeddings  │           │ Vector DB   │           │  GPT-4o-mini│
             └─────────────┘           └─────────────┘           └─────────────┘
```

## License

This project is for educational and portfolio purposes.

## Disclaimer

This tool is for informational purposes only. Always consult a healthcare professional for medical advice.
