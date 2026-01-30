"""Generic interrupt resume route.

This is used for non-confirmation interrupts (e.g., INPUT/SELECTION) created by the graph.
It resumes a paused LangGraph execution using Command(resume=...).
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from src.api.models import InterruptResumeRequest
from src.api.streaming_utils import accumulate_and_format_state_updates
from src.api.formatters import make_json_serializable

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/interrupt/resume")
async def resume_interrupt(request: InterruptResumeRequest):
    """Resume an interrupted graph execution for a session."""
    try:
        from src.api.graph_manager import get_graph

        graph = await get_graph()
        session_id = request.session_id
        config = {
            "configurable": {"thread_id": session_id, "session_id": session_id},
            "recursion_limit": 25,
        }

        resume_command = Command(resume=request.resume_data)

        async def stream_response():
            try:
                yield f"data: {json.dumps({'type': 'interrupt_resumed', 'message': '已收到输入，继续处理...'}, ensure_ascii=False)}\n\n"
                async for formatted in accumulate_and_format_state_updates(
                    graph.astream(
                        command=resume_command,
                        config=config,
                        stream_mode="updates",
                        session_id=session_id,
                    )
                ):
                    serializable = make_json_serializable(formatted)
                    json_str = json.dumps(serializable, ensure_ascii=False)
                    yield f"data: {json_str}\n\n"
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"resume_interrupt stream failed: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"resume_interrupt failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

