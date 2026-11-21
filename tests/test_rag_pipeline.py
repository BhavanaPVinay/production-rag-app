import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient with FastAPI lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Tests the /health endpoint reports healthy status and model configuration."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "collection_name" in data
    assert isinstance(data["document_count"], int)
    assert data["document_count"] >= 0


def test_index_endpoint(client: TestClient):
    """Tests the /index endpoint successfully reads and indexes data/sample_docs.txt."""
    response = client.post("/index")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["indexed_count"] > 0
    assert "sample_docs.txt" in data["message"]


def test_ask_endpoint_valid_query(client: TestClient):
    """Tests the /ask endpoint retrieves context, generates an answer, citations, and grounding."""
    payload = {
        "query": "How does ChromaDB perform approximate nearest neighbor search?",
        "top_k": 3,
        "min_similarity": 0.35,
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == payload["query"]
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert isinstance(data["citations"], list)
    assert len(data["citations"]) > 0
    assert "retrieved_contexts" in data
    assert "grounding" in data
    assert "grounded_ratio" in data["grounding"]
    assert "is_grounded" in data["grounding"]


def test_ask_endpoint_filtering_unrelated_query(client: TestClient):
    """Tests that an unrelated question with low similarity correctly filters chunks."""
    payload = {
        "query": "What is the capital of Mars and astrophysics theory?",
        "top_k": 2,
        "min_similarity": 0.95,  # Very high threshold to test filtering
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "No sufficiently relevant context" in data["answer"]
    assert data["citations"] == []
