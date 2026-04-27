"""
Session management service.
In-memory sessions with TTL expiry and conversation history.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4

import structlog

from app.core.config import settings
from app.core.exceptions import SessionNotFoundError
from app.models.schemas import ConversationTurn, SessionResponse

logger = structlog.get_logger(__name__)


class SessionService:
    """Thread-safe in-memory session store with TTL."""

    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()

    def create_session(self) -> str:
        session_id = str(uuid4())
        now = datetime.utcnow()
        self._sessions[session_id] = {
            "session_id": session_id,
            "created_at": now,
            "last_active": now,
            "history": [],
        }
        logger.info("session_created", session_id=session_id)
        return session_id

    def get_session(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        # Check TTL
        ttl = timedelta(hours=settings.SESSION_TTL_HOURS)
        if datetime.utcnow() - session["last_active"] > ttl:
            del self._sessions[session_id]
            raise SessionNotFoundError(session_id)
        return session

    def add_turn(self, session_id: str, question: str, answer: str, sources: list = None) -> None:
        session = self.get_session(session_id)
        session["history"].append(
            ConversationTurn(role="human", content=question).dict()
        )
        session["history"].append(
            ConversationTurn(role="assistant", content=answer, sources=sources).dict()
        )
        session["last_active"] = datetime.utcnow()
        # Trim history
        max_turns = settings.MAX_HISTORY_LENGTH * 2
        if len(session["history"]) > max_turns:
            session["history"] = session["history"][-max_turns:]

    def get_history(self, session_id: str) -> List[dict]:
        session = self.get_session(session_id)
        return session["history"]

    def get_or_create(self, session_id: Optional[str]) -> str:
        if session_id:
            try:
                self.get_session(session_id)
                return session_id
            except SessionNotFoundError:
                pass
        return self.create_session()

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[dict]:
        return [
            {
                "session_id": s["session_id"],
                "created_at": s["created_at"],
                "last_active": s["last_active"],
                "num_turns": len(s["history"]) // 2,
            }
            for s in self._sessions.values()
        ]

    async def _cleanup_loop(self) -> None:
        """Periodically remove expired sessions."""
        while True:
            await asyncio.sleep(3600)  # every hour
            ttl = timedelta(hours=settings.SESSION_TTL_HOURS)
            now = datetime.utcnow()
            expired = [
                sid
                for sid, s in self._sessions.items()
                if now - s["last_active"] > ttl
            ]
            for sid in expired:
                del self._sessions[sid]
            if expired:
                logger.info("sessions_cleaned", count=len(expired))


# Singleton
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
