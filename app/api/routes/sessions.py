"""Sessions API - manage conversation sessions."""

from fastapi import APIRouter, HTTPException

from app.models.schemas import SessionResponse, ConversationTurn
from app.services.session_service import get_session_service

router = APIRouter()


@router.post("/", status_code=201)
async def create_session():
    """Create a new conversation session."""
    svc = get_session_service()
    session_id = svc.create_session()
    return {"session_id": session_id, "message": "Session created"}


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session details and conversation history."""
    svc = get_session_service()
    try:
        session = svc.get_session(session_id)
        history = [ConversationTurn(**t) for t in session["history"]]
        return SessionResponse(
            session_id=session_id,
            created_at=session["created_at"],
            last_active=session["last_active"],
            num_turns=len(history) // 2,
            history=history,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{session_id}", status_code=200)
async def delete_session(session_id: str):
    """Delete a session and its history."""
    svc = get_session_service()
    deleted = svc.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"message": "Session deleted", "session_id": session_id}


@router.get("/")
async def list_sessions():
    """List all active sessions."""
    svc = get_session_service()
    return {"sessions": svc.list_sessions(), "total": len(svc.list_sessions())}
