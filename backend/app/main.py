from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.api import chat, search, traces


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"Starting SafeRx Analytics API...")
    print(f"ChromaDB path: {settings.chroma_persist_dir}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"LLM model: {settings.llm_model}")
    print(f"Hybrid search: {'enabled' if settings.enable_hybrid_search else 'disabled'}")
    print(f"Reranking: {'enabled' if settings.enable_reranking else 'disabled'} ({settings.reranker_model})")
    print(f"Tracing: {'enabled' if settings.enable_tracing else 'disabled'}")
    yield
    # Shutdown
    print("Shutting down SafeRx Analytics API...")


app = FastAPI(
    title="SafeRx Analytics API",
    description="AI-powered drug safety knowledge assistant with hybrid retrieval and observability",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(traces.router, tags=["traces"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "SafeRx Analytics API",
        "version": "2.0.0",
        "features": {
            "hybrid_search": settings.enable_hybrid_search,
            "reranking": settings.enable_reranking,
            "tracing": settings.enable_tracing,
        }
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to SafeRx Analytics API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "/api/chat",
            "chat_debug": "/api/chat/debug",
            "chat_compare": "/api/chat/compare",
            "search": "/api/search",
            "traces": "/api/traces",
            "trace_summary": "/api/traces/summary",
        }
    }
