# ⚡ RAG-Powered Document Q&A

> **Ask questions about your documents using local LLMs — no API keys, no data leakage, completely free.**

[![CI/CD](https://github.com/YOUR_USERNAME/rag-docqa/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/rag-docqa/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-orange.svg)](https://langchain.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend (Nginx)                    │
│               Drag-and-drop upload + Chat UI             │
└──────────────────────┬──────────────────────────────────┘
                       │ REST / SSE
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Application                     │
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

## ✨ Features

| Feature | Details |
|---|---|
| **Document Formats** | PDF, DOCX, TXT, Markdown, CSV, HTML |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast) |
| **Vector Store** | FAISS with cosine similarity + MMR for diversity |
| **LLM** | Ollama (local) with HuggingFace fallback |
| **Chunking** | Recursive character splitting (512 tokens, 64 overlap) |
| **Sessions** | In-memory conversation history with TTL |
| **Streaming** | Server-Sent Events (SSE) for real-time responses |
| **Monitoring** | Prometheus metrics + Grafana dashboards |
| **API Docs** | Interactive Swagger UI at `/docs` |
| **CI/CD** | GitHub Actions: lint → test → Docker build → push |

## 🚀 Quick Start (Docker — Recommended)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/rag-docqa.git
cd rag-docqa

# 2. One-command setup
chmod +x scripts/setup.sh && ./scripts/setup.sh
```

That's it. Open **http://localhost:3000** in your browser.

### What the setup script does:
1. Copies `.env.example` → `.env`
2. Builds all Docker images
3. Starts API, Ollama, Frontend, Prometheus, Grafana
4. Pulls the Mistral LLM model (~4GB)

## 🛠 Local Development (without Docker)

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed locally

```bash
# 1. Clone and enter
git clone https://github.com/YOUR_USERNAME/rag-docqa.git
cd rag-docqa

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows

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

# 8. Open the frontend (just open the file in your browser)
open frontend/index.html
# Or serve it: python -m http.server 3000 --directory frontend
```

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

### Multi-turn Conversation
```bash
# First turn — creates session
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the document"}'
# Returns: {"session_id": "abc-123", "answer": "..."}

# Follow-up — uses session history
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

## ⚙️ Configuration

Key environment variables (see `.env.example` for all):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `mistral` | LLM to use (mistral, llama3, phi3…) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Min relevance score (0-1) |
| `MMR_LAMBDA` | `0.5` | Diversity vs relevance (0=max diversity) |
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

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=app --cov-report=html
open htmlcov/index.html
```

## 📊 Monitoring

| Service | URL | Credentials |
|---|---|---|
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin |

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

## 🗂 Project Structure

```
rag-docqa/
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
│       ├── document_processor.py  # Multi-format file parsing
│       ├── vector_store.py        # FAISS operations
│       ├── llm_service.py         # Ollama/HuggingFace LLM
│       ├── rag_pipeline.py        # RAG orchestration
│       └── session_service.py     # Conversation history
├── frontend/
│   ├── index.html               # Full-featured chat UI
│   ├── Dockerfile               # Nginx container
│   └── nginx.conf               # Reverse proxy config
├── tests/
│   └── test_api.py              # Integration + unit tests
├── monitoring/
│   └── prometheus.yml           # Prometheus scrape config
├── scripts/
│   └── setup.sh                 # One-command setup
├── .github/workflows/
│   └── ci.yml                   # CI/CD pipeline
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

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

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

**Built with:** FastAPI · LangChain · FAISS · HuggingFace · Ollama · Docker · Prometheus · Grafana
