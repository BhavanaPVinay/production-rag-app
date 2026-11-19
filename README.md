# Production RAG API

A production-ready Retrieval-Augmented Generation (RAG) backend service built with **FastAPI**, **PyTorch SentenceTransformers**, and **ChromaDB**.

---

## Features

- **Semantic Embedding Generation**: Uses PyTorch and SentenceTransformers (`all-MiniLM-L6-v2`) to generate 384-dimensional dense vectors.
- **Persistent Vector Database**: Embeddings and metadata are indexed and persisted on disk using **ChromaDB** with cosine similarity.
- **Grounded `/ask` Endpoint**: Performs semantic retrieval, filters chunks below similarity thresholds, extracts answers, provides source citations, and computes a grounding score to mitigate hallucinations.
- **Augmented Prompt `/query` Endpoint**: Retrieves relevant context and builds ready-to-use LLM prompt templates.
- **Automatic Ingestion & Lifespan Management**: Automatically indexes knowledge base documents (`data/sample_docs.txt`) on service startup if the collection is empty.
- **Dockerized**: Containerized with a slim Python 3.11 base image for portable, isolated deployments.

---

## Project Structure

```text
production-rag-app/
├── Dockerfile                  # Container build definition
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── app/
│   ├── __init__.py             # Python package marker
│   ├── main.py                 # FastAPI application, schemas & endpoints
│   └── rag_pipeline.py         # Embedding, retrieval, and grounding engine
└── data/
    └── sample_docs.txt         # Knowledge base documents
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- `git`
- (Optional) Docker

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/production-rag-app.git
cd production-rag-app

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Development Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The interactive OpenAPI Swagger documentation will be available at:
**http://localhost:8000/docs**

---

## API Endpoints

### 1. Health Check
- **`GET /health`**
- Returns server status, active embedding model, collection name, and total indexed chunks.

```bash
curl -X GET http://localhost:8000/health
```

### 2. Grounded Question Answering (`/ask`)
- **`POST /ask`**
- Retrieves relevant documents, applies a minimum similarity threshold filter, extracts a grounded response, generates citations, and reports grounding metrics.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does ChromaDB perform vector search?",
    "top_k": 3,
    "min_similarity": 0.35
  }'
```

**Response Example:**
```json
{
  "query": "How does ChromaDB perform vector search?",
  "answer": "ChromaDB is an open-source, AI-native embedding database designed for developer productivity. It operates either in-memory or persisted on local storage via SQLite and Parquet files, utilizing the Hierarchical Navigable Small World (HNSW) algorithm for fast, approximate nearest neighbor (ANN) vector search.",
  "citations": [
    {
      "source": "sample_docs.txt",
      "chunk_index": 3,
      "similarity": 0.6124
    }
  ],
  "retrieved_contexts": [...],
  "filtered_out_count": 0,
  "grounding": {
    "grounded_ratio": 1.0,
    "is_grounded": true,
    "ungrounded_terms": []
  }
}
```

### 3. Prompt Augmentation (`/query`)
- **`POST /query`**
- Retrieves matching contexts and compiles an augmented prompt template suitable for sending to an LLM.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the common distance metrics used for embeddings?",
    "top_k": 2
  }'
```

### 4. Manual Indexing
- **`POST /index`**
- Re-reads and indexes `data/sample_docs.txt` into the ChromaDB collection.

```bash
curl -X POST http://localhost:8000/index
```

---

## Docker Deployment

### Build the Image
```bash
docker build -t production-rag-app .
```

### Run the Container
```bash
docker run -d -p 8000:8000 --name rag-service production-rag-app
```

Check health:
```bash
curl http://localhost:8000/health
```

