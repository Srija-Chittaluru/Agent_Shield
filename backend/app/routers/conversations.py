"""/conversations API — replay a single scenario live to verify a fix."""
from fastapi import APIRouter, HTTPException

from app.core.orchestrator import replay_conversation
from app.db import build_conversation_payload, get_conversation

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/{conversation_id}/replay")
async def replay(conversation_id: int) -> dict:
    """Re-run this conversation's scenario as a NEW conversation, judge + fix, return it."""
    if get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail=f"conversation {conversation_id} not found")
    new_id = await replay_conversation(conversation_id)
    if new_id is None:
        raise HTTPException(status_code=500, detail="replay failed")
    return build_conversation_payload(new_id)
