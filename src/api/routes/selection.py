"""选择相关路由"""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.graph_manager import get_graph
from src.api.models import SelectionResolveRequest, SelectionCancelRequest
from src.confirmation.selection_manager import get_selection_manager
from langgraph.types import Command

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/selection/resolve")
async def resolve_selection(request: SelectionResolveRequest):
    """解析用户选择并恢复执行"""
    try:
        from src.api.streaming_utils import accumulate_and_format_state_updates
        import json
        import asyncio

        # 1. 记录用户选择到 selection_manager
        manager = get_selection_manager()
        result = await manager.resolve_selection(
            request.selection_id,
            request.selected_option_id,
        )

        session_id = result.metadata.get("session_id") if result.metadata else None
        if not session_id:
            # 从 selection_action 获取 session_id
            selection_action = await manager.get_selection(request.selection_id)
            if selection_action:
                session_id = selection_action.session_id

        if not session_id:
            raise ValueError("无法获取 session_id")

        logger.info(
            f"用户选择已解析: selection_id={request.selection_id}, "
            f"selected_option_id={request.selected_option_id}, session_id={session_id}"
        )

        # 2. 准备恢复执行
        graph = await get_graph()
        config = {
            "configurable": {"thread_id": session_id, "session_id": session_id},
            "recursion_limit": 25
        }
        logger.info(f"[selection/resolve] 准备恢复执行: session_id={session_id}, thread_id={session_id}, config={config}")

        # 3. 【LangGraph 1.x】使用 Command(resume=...) 恢复图执行
        # resume 值会被 interrupt() 返回，传递给 _execute_user_selection
        resume_data = {
            "selected_option_id": request.selected_option_id
        }

        # 4. 流式响应函数
        async def stream_response():
            """流式返回恢复执行的结果"""
            try:
                yield f"data: {json.dumps({'type': 'selection_resolved', 'message': '已选择商品，正在继续处理...'}, ensure_ascii=False)}\n\n"
                
                resume_command = Command(resume=resume_data)
                
                # 使用通用工具函数处理状态更新
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

                # 发送完成信号
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

    except ValueError as e:
        # 选择不存在、已过期、或选项ID无效
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"解析选择失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/selection/cancel")
async def cancel_selection(request: SelectionCancelRequest):
    """取消选择

    用户点击取消按钮后调用此接口：
    1. 取消选择（记录到 selection_manager，状态变为 CANCELLED）
    2. 返回成功响应，前端关闭对话框

    设计说明：
    - 取消操作不需要恢复图执行
    - 用户未做出选择，任务链自然结束（等待下一个用户输入）
    - 符合 LangGraph 1.x 最佳实践：无需 interrupt/resume 处理取消
    """
    try:
        manager = get_selection_manager()

        # 执行取消
        result = await manager.cancel_selection(request.selection_id)

        logger.info(f"选择已取消: selection_id={request.selection_id}, status={result.status.value}")

        return {
            "success": True,
            "status": result.status.value,
            "selection_type": result.selection_type,
            "message": "已取消选择",
        }
    except ValueError as e:
        # 选择不存在或已处理
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"取消选择失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/selection/pending/{session_id}")
async def get_pending_selection(session_id: str):
    """获取会话的待选择操作

    2025最佳实践：使用 interrupt() 机制后，待选择信息由 selection_manager 管理
    前端重新连接时通过此接口检查是否有待选择操作
    """
    try:
        manager = get_selection_manager()
        selection = await manager.get_pending_selection(session_id)

        if selection:
            return {
                "has_pending": True,
                "selection": {
                    "selection_id": selection.selection_id,
                    "selection_type": selection.selection_type,
                    "options": selection.options,
                    "display_message": selection.display_message,
                    "metadata": selection.metadata,
                    "expires_at": selection.expires_at.isoformat() if selection.expires_at else None,
                },
            }
        return {"has_pending": False}
    except Exception as e:
        logger.error(f"获取待选择操作失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
