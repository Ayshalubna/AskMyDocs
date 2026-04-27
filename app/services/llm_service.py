"""
LLM service using Ollama with phi3 model, optimized for low-RAM environments.
"""

import asyncio
import time
from functools import partial
from typing import AsyncGenerator, List, Optional, Tuple

import structlog
from langchain.schema import Document
from langchain_community.llms import Ollama

from app.core.config import settings
from app.core.exceptions import LLMError

logger = structlog.get_logger(__name__)

# --- Tuning constants ---
# Keep top_k low in your retriever (recommended: 2-3). More docs = more tokens = slower.
# Keep chunk_size low in your splitter (recommended: 300-400 chars, overlap: 50).
CONTEXT_CHAR_LIMIT = 1200   # Hard cap on context fed to the model (~300 tokens)
HISTORY_TURNS = 1           # Only pass the last 1 exchange to save tokens

# Compact prompt — every extra word costs latency on low RAM
RAG_SYSTEM_PROMPT = """Use only the context below to answer the question. Be brief.

Context:
{context}

{history_block}Question: {question}
Answer:"""

HISTORY_BLOCK_TEMPLATE = "Previous exchange:\n{chat_history}\n\n"


class LLMService:

    def __init__(self):
        self._llm = None
        self._model_name = "phi3"

    @property
    def model_name(self) -> str:
        """Public read-only property so external code can access model_name without underscore."""
        return self._model_name

    async def initialize(self):
        """Initialize Ollama with phi3, tuned for low-memory inference."""
        try:
            self._llm = Ollama(
                model="phi3",
                # Low temperature = more deterministic, less sampling overhead
                temperature=0.1,
                # Hard limit on generated tokens — prevents runaway generation that hangs
                num_predict=200,
                # Reduce context window loaded into memory (default is 2048+)
                num_ctx=1024,
                # Two threads is enough for phi3; more causes RAM thrashing on 4GB machines
                num_thread=2,
                # Keep model loaded between calls — avoids cold-start reload delay
                keep_alive="10m",
            )
            self._model_name = "phi3"
            logger.info("llm_ready", backend="ollama", model=self._model_name)
        except Exception as e:
            raise LLMError(f"Ollama initialization failed: {e}")

    def _format_context(self, docs_with_scores: List[Tuple[Document, float]]) -> str:
        """
        Format retrieved docs into a compact context string.
        Truncates to CONTEXT_CHAR_LIMIT to prevent oversized prompts.
        """
        parts = []
        total_chars = 0
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            snippet = doc.page_content.strip()
            remaining = CONTEXT_CHAR_LIMIT - total_chars
            if remaining <= 0:
                break
            if len(snippet) > remaining:
                snippet = snippet[:remaining].rsplit(" ", 1)[0]
            parts.append(f"[{i}] {snippet}")
            total_chars += len(snippet)

        return "\n\n".join(parts)

    def _format_history(self, history: List[dict]) -> str:
        """
        Return only the last HISTORY_TURNS exchanges to keep the prompt short.
        Each turn = one user message + one assistant message = 2 list items.
        """
        if not history:
            return ""
        recent = history[-(HISTORY_TURNS * 2):]
        return "\n".join(
            f"{h['role'].capitalize()}: {h['content']}" for h in recent
        )

    def _build_prompt(
        self,
        question: str,
        docs_with_scores: List[Tuple[Document, float]],
        history: Optional[List[dict]],
    ) -> str:
        context = self._format_context(docs_with_scores)
        chat_history = self._format_history(history or [])
        history_block = (
            HISTORY_BLOCK_TEMPLATE.format(chat_history=chat_history)
            if chat_history
            else ""
        )
        return RAG_SYSTEM_PROMPT.format(
            context=context,
            history_block=history_block,
            question=question,
        )

    async def generate_answer(
        self,
        question: str,
        docs_with_scores: List[Tuple[Document, float]],
        history: Optional[List[dict]] = None,
    ) -> Tuple[str, dict]:
        """Generate a complete answer. Runs Ollama in a thread to keep FastAPI non-blocking."""
        if not self._llm:
            raise LLMError("LLM not initialized. Call initialize() first.")

        if not docs_with_scores:
            return "No relevant documents found.", {}

        prompt = self._build_prompt(question, docs_with_scores, history)

        try:
            start = time.time()

            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None, partial(self._llm.invoke, prompt)
            )

            elapsed_ms = round((time.time() - start) * 1000, 2)
            logger.info("llm_answered", time_ms=elapsed_ms, model=self._model_name)

            return answer.strip(), {
                "model": self._model_name,
                "backend": "ollama",
                "time_ms": elapsed_ms,
            }

        except Exception as e:
            raise LLMError(f"Answer generation failed: {e}")

    async def stream_answer(
        self,
        question: str,
        docs_with_scores: List[Tuple[Document, float]],
        history: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream answer tokens as they are generated.
        Preferred over generate_answer for UI responsiveness — the user sees
        the first token in ~1s even if total generation takes longer.
        """
        if not self._llm:
            raise LLMError("LLM not initialized. Call initialize() first.")

        if not docs_with_scores:
            yield "No relevant documents found."
            return

        prompt = self._build_prompt(question, docs_with_scores, history)

        try:
            async for chunk in self._llm.astream(prompt):
                yield chunk
        except Exception as e:
            raise LLMError(f"Streaming failed: {e}")

    async def health_check(self) -> dict:
        """Return current health status of the LLM service."""
        return {
            "status": "healthy" if self._llm is not None else "uninitialized",
            "model": self._model_name,
            "backend": "ollama",
        }
