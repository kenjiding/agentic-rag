"""确认相关路由"""
import logging
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.models import ConfirmationResolveRequest
from src.confirmation import (
    get_confirmation_manager,
    ConfirmationNotFoundError,
    ConfirmationExpiredError,
    ConfirmationAlreadyResolvedError,
)
from src.api.streaming_utils import accumulate_and_format_state_updates
from langgraph.types import Command

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/confirmation/resolve")
async def resolve_confirmation(request: ConfirmationResolveRequest):
    """解析确认操作并恢复执行"""
    try:
        from src.api.graph_manager import get_graph

        manager = get_confirmation_manager()
        confirmation = await manager.get_confirmation(request.confirmation_id)
        if not confirmation:
            raise ValueError("确认不存在")

        session_id = confirmation.session_id
        
        logger.info(
            f"用户确认请求: confirmation_id={request.confirmation_id}, "
            f"confirmed={request.confirmed}, session_id={session_id}"
        )

        graph = await get_graph()
        config = {
            "configurable": {"thread_id": session_id, "session_id": session_id},
            "recursion_limit": 25
        }
        
        resume_data = {
            "confirmed": request.confirmed
        }
        
        result = await manager.resolve_confirmation(
            request.confirmation_id,
            request.confirmed,
        )
        
        logger.info(
            f"确认操作已执行: confirmation_id={request.confirmation_id}, "
            f"status={result.status}, success={result.execution_result is not None if result.execution_result else False}"
        )
        async def stream_response():
            """流式返回恢复执行的结果"""
            try:
                action_text = "已确认" if request.confirmed else "已取消"
                yield f"data: {json.dumps({'type': 'confirmation_resolved', 'message': f'{action_text}，正在继续处理...'}, ensure_ascii=False)}\n\n"
                
                if request.confirmed and result.execution_result:
                    success_text = result.execution_result.get("text", "操作已确认并完成！")
                    yield f"data: {json.dumps({'type': 'state_update', 'data': {'response_type': 'text', 'response_data': {}, 'content': success_text, 'role': 'assistant'}}, ensure_ascii=False)}\n\n"
                
                resume_command = Command(resume=resume_data)
                
                async for formatted in accumulate_and_format_state_updates(
                    graph.astream(
                        command=resume_command,
                        config=config,
                        stream_mode="updates",
                        session_id=session_id
                    )
                ):
                    json_str = json.dumps(formatted, ensure_ascii=False)
                    yield f"data: {json_str}\n\n"
                
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logger.error(f"流式执行任务链失败: {e}", exc_info=True)
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

    except ConfirmationNotFoundError:
        raise HTTPException(status_code=404, detail="确认不存在")
    except ConfirmationExpiredError:
        raise HTTPException(status_code=410, detail="确认已过期")
    except ConfirmationAlreadyResolvedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"解析确认失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/confirmation/pending/{session_id}")
async def get_pending_confirmation(session_id: str):
    """获取会话的待确认操作

    前端重新连接时检查是否有待确认操作
    """
    try:
        manager = get_confirmation_manager()
        confirmation = await manager.get_pending_confirmation(session_id)

        if confirmation:
            return {
                "has_pending": True,
                "confirmation": {
                    "confirmation_id": confirmation.confirmation_id,
                    "action_type": confirmation.action_type,
                    "display_message": confirmation.display_message,
                    "display_data": confirmation.display_data,
                    "expires_at": confirmation.expires_at.isoformat() if confirmation.expires_at else None,
                },
            }
        return {"has_pending": False}
    except Exception as e:
        logger.error(f"获取待确认操作失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

