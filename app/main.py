import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.rag_pipeline import RAGPipeline

# Global reference to pipeline
rag_pipeline: Optional[RAGPipeline] = None
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes RAG pipeline and loads initial sample documents upon startup."""
    global rag_pipeline
    rag_pipeline = RAGPipeline(
        model_name="all-MiniLM-L6-v2",
        persist_directory="./chroma_db",
        collection_name="documents_collection",
    )

    # Index sample data on startup if collection is currently empty
    norm_sample_path = os.path.abspath(SAMPLE_DATA_PATH)
    if rag_pipeline.get_document_count() == 0 and os.path.exists(norm_sample_path):
        count = rag_pipeline.index_file(norm_sample_path)
        print(f"[Lifespan] Indexed {count} sample document chunks into ChromaDB.")

    yield
    print("[Lifespan] Shutting down RAG service.")


app = FastAPI(
    title="Production RAG API",
    description="FastAPI service powered by PyTorch SentenceTransformers and ChromaDB for semantic retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request and Response Schemas ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question or search query.")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of relevant documents to retrieve.")


class ContextItem(BaseModel):
    document: str
    metadata: Dict[str, Any]
    distance: float
    similarity: float


class QueryResponse(BaseModel):
    query: str
    retrieved_contexts: List[ContextItem]
    augmented_prompt: str


class IndexResponse(BaseModel):
    status: str
    indexed_count: int
    message: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    collection_name: str
    document_count: int


# --- Endpoints ---
@app.get("/", tags=["General"])
def root():
    """Service status and quick links."""
    return {
        "service": "Production RAG API",
        "docs_url": "/docs",
        "status": "online",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Reports system health, active embedding model, and indexed document count."""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized.",
        )
    return HealthResponse(
        status="healthy",
        model_name=rag_pipeline.model_name,
        collection_name=rag_pipeline.collection_name,
        document_count=rag_pipeline.get_document_count(),
    )


@app.post("/index", response_model=IndexResponse, tags=["RAG"])
def index_documents():
    """Triggers indexing or re-indexing of sample documents from data/sample_docs.txt."""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized.",
        )

    norm_path = os.path.abspath(SAMPLE_DATA_PATH)
    if not os.path.exists(norm_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample document file not found at {norm_path}",
        )

    indexed = rag_pipeline.index_file(norm_path)
    return IndexResponse(
        status="success",
        indexed_count=indexed,
        message=f"Successfully indexed {indexed} chunks from sample_docs.txt",
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(request: QueryRequest):
    """Accepts a search query, retrieves top-k semantic matches, and returns an augmented prompt."""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized.",
        )

    result = rag_pipeline.query_and_augment(query=request.query, top_k=request.top_k)
    return QueryResponse(**result)

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer from indexed docs.")
    top_k: int = Field(default=3, ge=1, le=10, description="Max chunks to retrieve before filtering.")
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum similarity cutoff.")


class CitationItem(BaseModel):
    source: str
    chunk_index: Optional[int] = None
    similarity: float


class GroundingReport(BaseModel):
    grounded_ratio: float
    is_grounded: bool
    ungrounded_terms: List[str]


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationItem]
    retrieved_contexts: List[ContextItem]
    filtered_out_count: int
    grounding: GroundingReport


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_rag(request: AskRequest):
    """Retrieves relevant context, filters low-similarity noise, and returns a grounded answer."""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized.",
        )
    result = rag_pipeline.generate_grounded_answer(
        query=request.query,
        top_k=request.top_k,
        min_similarity=request.min_similarity,
    )
    return AskResponse(**result)
