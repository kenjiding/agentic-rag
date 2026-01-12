"""确认相关路由

支持 LangGraph 1.x interrupt() 机制：
- 确认后使用 Command(resume=...) 恢复执行
- interrupt() 会返回 resume 的值给 Agent
"""
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
    ConfirmationStatus,
)
from src.api.streaming_utils import accumulate_and_format_state_updates
from src.api.formatters import make_json_serializable
from langgraph.types import Command

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/confirmation/resolve")
async def resolve_confirmation(request: ConfirmationResolveRequest):
    """解析确认操作并恢复执行

    LangGraph 1.x interrupt() 机制：
    1. 使用 Command(resume=...) 恢复被中断的图
    2. interrupt() 调用会返回 resume 的值
    3. 图从中断点继续执行
    """
    try:
        from src.api.graph_manager import get_graph

        manager = get_confirmation_manager()
        confirmation = await manager.get_confirmation(request.confirmation_id)
        if not confirmation:
            raise ValueError("确认不存在")

        session_id = confirmation.session_id

        logger.info(
            f"[interrupt] 用户确认请求: confirmation_id={request.confirmation_id}, "
            f"confirmed={request.confirmed}, session_id={session_id}"
        )

        graph = await get_graph()
        config = {
            "configurable": {"thread_id": session_id, "session_id": session_id},
            "recursion_limit": 25
        }

        # 【LangGraph interrupt() 机制】resume 数据
        # 这个值会被 interrupt() 调用返回给 Agent
        resume_data = {
            "confirmed": request.confirmed,
            "confirmation_id": request.confirmation_id
        }

        # 【关键修复】只标记确认状态，不执行操作
        # 操作由 Agent 在 graph resume 后执行（符合 LangGraph interrupt/resume 机制）
        confirmation_action = await manager.get_confirmation(request.confirmation_id)
        await manager._storage.update_status(
            request.confirmation_id,
            ConfirmationStatus.CONFIRMED if request.confirmed else ConfirmationStatus.CANCELLED
        )
        
        logger.info(
            f"[interrupt] 确认状态已更新: confirmation_id={request.confirmation_id}, "
            f"confirmed={request.confirmed}"
        )


        async def stream_response():
            """流式返回恢复执行的结果"""
            try:
                action_text = "已确认" if request.confirmed else "已取消"
                yield f"data: {json.dumps({'type': 'confirmation_resolved', 'message': f'{action_text}，正在继续处理...'}, ensure_ascii=False)}\n\n"

                # 【核心】使用 Command(resume=...) 恢复被 interrupt() 暂停的图
                resume_command = Command(resume=resume_data)

                async for formatted in accumulate_and_format_state_updates(
                    graph.astream(
                        command=resume_command,
                        config=config,
                        stream_mode="updates",
                        session_id=session_id
                    )
                ):
                    # 【关键修复】使用 make_json_serializable 递归处理所有不可序列化的对象
                    # 包括 Document 对象和其他 LangChain 对象
                    serializable_formatted = make_json_serializable(formatted)
                    
                    # 额外过滤掉 messages 和 result 字段（前端不需要这些原始 LangChain 对象）
                    if isinstance(serializable_formatted, dict) and "data" in serializable_formatted:
                        data = serializable_formatted["data"]
                        filtered_data = {
                            k: v for k, v in data.items()
                            if k != 'messages' and k != 'result'
                        }
                        serializable_formatted["data"] = filtered_data
                    
                    json_str = json.dumps(serializable_formatted, ensure_ascii=False)
                    yield f"data: {json_str}\n\n"

                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"流式执行失败: {e}", exc_info=True)
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

