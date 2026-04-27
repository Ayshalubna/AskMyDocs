# ⚡ AskMyDocs — RAG-Powered Document Q&A

> **Ask questions about your documents using local LLMs — no API keys, no data leakage, completely free.**

[![CI/CD](https://github.com/Ayshalubna/AskMyDocs/actions/workflows/ci.yml/badge.svg)](https://github.com/Ayshalubna/AskMyDocs/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-orange.svg)](https://langchain.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📸 Demo

| Upload & Chat | Source Citations | Multi-turn Conversation |
|---|---|---|
| Drag-and-drop any PDF/DOCX/TXT | Every answer cites exact source chunks | Session memory across follow-up questions |

> **Try it locally:** `./scripts/setup.sh` → open [http://localhost:3000](http://localhost:3000)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend (Nginx)                       │
│            Drag-and-drop upload + Chat UI                │
└──────────────────────┬──────────────────────────────────┘
                       │ REST / SSE
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Application                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Documents  │  │    Query     │  │    Sessions    │  │
│  │   Router    │  │    Router    │  │    Router      │  │
│  └──────┬──────┘  └──────┬───────┘  └────────────────┘  │
│         │                │                               │
│  ┌──────▼──────┐  ┌──────▼───────────────────────────┐  │
│  │  Document   │  │          RAG Pipeline             │  │
│  │  Processor  │  │  Retrieve → Rerank → Generate     │  │
│  │  (chunking) │  └──────┬──────────────┬─────────────┘  │
│  └──────┬──────┘         │              │               │
│         │         ┌──────▼──────┐  ┌───▼──────────┐   │
│  ┌──────▼──────┐  │  FAISS      │  │  LLM Service │   │
│  │  HuggingFace│  │  Vector     │  │  (Ollama /   │   │
│  │  Embeddings │  │  Store      │  │   HF Fallback│   │
│  └─────────────┘  └─────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
         │                                    │
   ┌─────▼──────┐                    ┌────────▼──────┐
   │  FAISS     │                    │  Ollama LLM   │
   │  Index on  │                    │  (Local, Free)│
   │  Disk      │                    └───────────────┘
   └────────────┘
```

**Data flow:**
1. File upload → `DocumentProcessor` parses and splits into chunks
2. Chunks embedded via HuggingFace `all-MiniLM-L6-v2` → stored in FAISS
3. Query → MMR similarity search retrieves top-k diverse chunks
4. Retrieved context + conversation history → Ollama LLM generates answer
5. Response streamed back to UI via SSE with source citations

---

## ✨ Features

| Feature | Details |
|---|---|
| **Document Formats** | PDF, DOCX, TXT, Markdown, CSV, HTML |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine similarity) |
| **Vector Store** | FAISS with MMR diversity re-ranking |
| **LLM** | Ollama (local, zero API cost) with HuggingFace fallback |
| **Chunking** | Recursive character splitting — 512 chars, 64 overlap |
| **Sessions** | In-memory conversation history with TTL cleanup |
| **Streaming** | Server-Sent Events (SSE) for real-time token streaming |
| **Monitoring** | Prometheus metrics + Grafana dashboards |
| **API Docs** | Interactive Swagger UI at `/docs` |
| **CI/CD** | GitHub Actions: lint → test → Docker build → push to GHCR |

---

## 🚀 Quick Start (Docker — Recommended)

```bash
# 1. Clone
git clone https://github.com/Ayshalubna/AskMyDocs.git
cd AskMyDocs

# 2. One-command setup
chmod +x scripts/setup.sh && ./scripts/setup.sh
```

Open **http://localhost:3000** in your browser.

### What the setup script does:
1. Copies `.env.example` → `.env`
2. Builds all Docker images
3. Starts API, Ollama, Frontend, Prometheus, Grafana
4. Pulls the Mistral LLM model (~4 GB, one-time download)

---

## 🛠 Local Development (without Docker)

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed locally

```bash
# 1. Clone and enter
git clone https://github.com/Ayshalubna/AskMyDocs.git
cd AskMyDocs

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env: set OLLAMA_BASE_URL=http://localhost:11434

# 5. Pull a model via Ollama
ollama pull mistral

# 6. Create storage directories
mkdir -p storage/uploads storage/vectorstore

# 7. Run the API
uvicorn app.main:app --reload --port 8000

# 8. Serve the frontend
python -m http.server 3000 --directory frontend
# Then open http://localhost:3000
```

---

## 📡 API Reference

### Upload Document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@yourfile.pdf"
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics covered?",
    "top_k": 5,
    "use_mmr": true,
    "include_sources": true
  }'
```

**Response:**
```json
{
  "answer": "The document covers three main topics: ...",
  "session_id": "abc-123",
  "sources": [
    { "content": "...", "score": 0.87, "filename": "report.pdf", "page_number": 3 }
  ],
  "model_used": "mistral",
  "processing_time_ms": 1423.5
}
```

### Multi-turn Conversation
```bash
# First turn — creates session
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the document"}'
# Returns: {"session_id": "abc-123", "answer": "..."}

# Follow-up — passes session for memory
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me more about the second point", "session_id": "abc-123"}'
```

### Semantic Search (no LLM)
```bash
curl -X POST http://localhost:8000/api/v1/query/search \
  -H "Content-Type: application/json" \
  -d '{"query": "neural networks", "top_k": 5}'
```

### Filter by Document
```bash
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does this say about risks?",
    "doc_ids": ["your-doc-id-here"]
  }'
```

Full interactive API docs: **http://localhost:8000/docs**

---

## ⚙️ Configuration

Key environment variables (see `.env.example` for all):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `mistral` | LLM to use (mistral, llama3, phi3…) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Min relevance score (0–1) |
| `MMR_LAMBDA` | `0.5` | Diversity vs relevance (0 = max diversity) |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum file size |

### Switching LLM Models

```bash
# In .env
OLLAMA_MODEL=llama3       # Meta's Llama 3
OLLAMA_MODEL=phi3         # Microsoft's Phi-3 (fast, small)
OLLAMA_MODEL=mixtral      # Mistral's MoE model (best quality)
OLLAMA_MODEL=codellama    # Code-focused
```

```bash
# Pull the model first
docker-compose exec ollama ollama pull llama3
# Then restart the API
docker-compose restart api
```

---

## 📊 Performance Evaluation

See [EVALUATION.md](EVALUATION.md) for full methodology and raw data. Summary results on a standard test corpus (10 documents, 250 questions):

| Metric | Value | Notes |
|---|---|---|
| **Avg. response latency** | 1.38 s | Measured p50 over 250 queries |
| **p95 response latency** | 1.89 s | All under the 2 s SLA target |
| **Retrieval precision@5** | 0.81 | Fraction of top-5 chunks that are relevant |
| **Answer faithfulness** | 0.79 | Answers grounded in retrieved context |
| **Hallucination rate** | ~8 % | Chunk 512 + threshold 0.3 vs baseline 22 % |
| **Embedding throughput** | ~420 chunks/s | CPU, `all-MiniLM-L6-v2`, batch=32 |
| **FAISS index build** | 1.2 s / 1 k chunks | On-disk flat cosine index |

**Key tuning findings:**
- Chunk size 512 with overlap 64 reduces hallucination vs 1024-char chunks by ~65 %
- MMR lambda 0.5 gives best balance of diversity and relevance vs greedy top-k
- Similarity threshold 0.3 filters low-quality retrievals without hurting recall
- top-k = 5 outperforms top-k = 3 on multi-aspect questions with marginal latency cost

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=app --cov-report=html
open htmlcov/index.html
```

Test suite covers:

- Health and root endpoints
- Document upload (TXT, unsupported types, oversized files)
- Document CRUD and index stats
- Session lifecycle (create, get, delete)
- `DocumentProcessor` unit tests (clean text, dedup, metadata enrichment)
- Schema validation (whitespace stripping, bounds checking)
- Config and custom exceptions

---

## 📈 Monitoring

| Service | URL | Credentials |
|---|---|---|
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin |

Tracked metrics: query latency (p50/p95/p99), embedding time, FAISS search time, LLM generation time, document count, error rates.

---

## 🐳 Docker Commands

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f ollama

# Stop
docker-compose down

# Full reset (removes volumes)
docker-compose down -v

# Rebuild after code changes
docker-compose build api && docker-compose up -d api
```

---

## 🗂 Project Structure

```
AskMyDocs/
├── app/
│   ├── main.py                  # FastAPI app factory
│   ├── api/routes/
│   │   ├── documents.py         # Upload/list/delete endpoints
│   │   ├── query.py             # Q&A and search endpoints
│   │   ├── sessions.py          # Session management
│   │   └── health.py            # Health check
│   ├── core/
│   │   ├── config.py            # Pydantic settings
│   │   ├── exceptions.py        # Custom HTTP exceptions
│   │   └── logging.py           # Structured JSON logging
│   ├── models/
│   │   └── schemas.py           # Request/Response models
│   └── services/
│       ├── document_processor.py  # Multi-format file parsing + chunking
│       ├── vector_store.py        # Async FAISS operations + MMR
│       ├── llm_service.py         # Ollama / HuggingFace LLM
│       ├── rag_pipeline.py        # RAG orchestration
│       └── session_service.py     # Conversation history + TTL
├── frontend/
│   ├── index.html               # Chat UI (drag-and-drop + SSE streaming)
│   ├── Dockerfile               # Nginx container
│   └── nginx.conf               # Reverse proxy config
├── tests/
│   └── test_api.py              # Integration + unit tests
├── monitoring/
│   └── prometheus.yml           # Prometheus scrape config
├── scripts/
│   └── setup.sh                 # One-command Docker setup
├── .github/workflows/
│   └── ci.yml                   # Lint → Test → Docker build → push GHCR
├── EVALUATION.md                # Benchmarks and tuning methodology
├── docker-compose.yml
├── Dockerfile                   # Multi-stage build (builder + runtime)
├── requirements.txt
└── .env.example
```

---

## 🤝 Contributing

```bash
# Fork, clone, create branch
git checkout -b feature/your-feature

# Install dev tools
pip install black ruff pre-commit
pre-commit install

# Make changes, run tests
pytest tests/ -v

# Push and open PR
git push origin feature/your-feature
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

**Built with:** FastAPI · LangChain · FAISS · HuggingFace · Ollama · Docker · Prometheus · Grafana

