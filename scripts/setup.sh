#!/usr/bin/env bash
# ================================================================
# RAG Document Q&A — Quick Setup Script
# Usage: ./scripts/setup.sh
# ================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

info "🚀 RAG Document Q&A Setup"
echo "=================================================="

# ── Check Docker ──
command -v docker >/dev/null 2>&1 || error "Docker is required. Install: https://docs.docker.com/get-docker/"
command -v docker-compose >/dev/null 2>&1 || command -v docker >/dev/null 2>&1 || error "docker-compose required"

# ── .env ──
if [ ! -f .env ]; then
  cp .env.example .env
  info "Created .env from .env.example"
else
  warn ".env already exists — skipping"
fi

# ── Storage dirs ──
mkdir -p storage/uploads storage/vectorstore monitoring/grafana/{dashboards,provisioning}
info "Created storage directories"

# ── Pull & build ──
info "Building Docker images…"
docker-compose build --no-cache

# ── Start ──
info "Starting services…"
docker-compose up -d

# ── Wait for API ──
info "Waiting for API to be ready…"
MAX_WAIT=120; ELAPSED=0
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
  sleep 3; ELAPSED=$((ELAPSED+3))
  [ $ELAPSED -ge $MAX_WAIT ] && error "API did not start in ${MAX_WAIT}s. Check: docker-compose logs api"
  echo -n "."
done
echo ""

# ── Pull Ollama model ──
MODEL=${OLLAMA_MODEL:-mistral}
info "Pulling Ollama model: $MODEL (this may take a few minutes)…"
docker-compose exec ollama ollama pull $MODEL || warn "Model pull failed — will retry automatically"

echo ""
echo -e "${GREEN}=================================================="
echo "✅ RAG Document Q&A is running!"
echo "=================================================="
echo -e "${NC}"
echo "  🌐 Frontend:    http://localhost:3000"
echo "  🔧 API Docs:    http://localhost:8000/docs"
echo "  ❤️  Health:      http://localhost:8000/health"
echo "  📊 Metrics:     http://localhost:9090 (Prometheus)"
echo "  📈 Dashboards:  http://localhost:3001 (Grafana)"
echo ""
echo "  Next steps:"
echo "  1. Open http://localhost:3000"
echo "  2. Upload a PDF or text file"
echo "  3. Ask questions!"
echo ""
