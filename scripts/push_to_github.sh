#!/usr/bin/env bash
# ================================================================
# Push this project to GitHub
# Usage: ./scripts/push_to_github.sh YOUR_GITHUB_USERNAME
# ================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[STEP]${NC} $1"; }

USERNAME=${1:-"YOUR_USERNAME"}
REPO="rag-docqa"

warn "1. Go to https://github.com/new and create a repo named: $REPO"
warn "   • Set to PUBLIC (for free audience access)"
warn "   • Do NOT initialize with README (we have one)"
echo ""
read -p "Press Enter once you've created the repo on GitHub..."

info "Initializing git..."
cd "$(dirname "$0")/.."

git init
git add .
git commit -m "feat: initial RAG Document Q&A system

- FastAPI REST API with async support
- LangChain + FAISS vector store
- HuggingFace sentence-transformers embeddings
- Ollama local LLM (free, no API keys)
- Multi-format document support (PDF/DOCX/TXT/MD/CSV/HTML)
- Semantic search with MMR diversity
- Conversation sessions with history
- SSE streaming responses
- Production Docker + docker-compose
- Prometheus + Grafana monitoring
- GitHub Actions CI/CD
- Full test suite"

git branch -M main
git remote add origin "https://github.com/${USERNAME}/${REPO}.git"

info "Pushing to GitHub..."
git push -u origin main

echo ""
echo -e "${GREEN}=================================================="
echo "✅ Project pushed to GitHub!"
echo "=================================================="
echo -e "${NC}"
echo "  Repository:  https://github.com/${USERNAME}/${REPO}"
echo "  Actions:     https://github.com/${USERNAME}/${REPO}/actions"
echo ""
echo "  To enable free public access:"
echo "  • Go to Settings → Pages → Deploy from branch: main"
echo "  • The API docs will be at: https://${USERNAME}.github.io/${REPO}"
echo ""
echo "  To deploy the full stack for free:"
echo "  • Railway.app: https://railway.app (free tier)"
echo "  • Render.com:  https://render.com (free tier)"  
echo "  • Fly.io:      https://fly.io (free tier)"
echo ""
