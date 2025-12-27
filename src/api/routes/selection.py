"""选择相关路由"""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.models import SelectionResolveRequest, SelectionCancelRequest
from src.confirmation.selection_manager import get_selection_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/selection/resolve")
async def resolve_selection(request: SelectionResolveRequest):
    """解析用户选择

    用户选择产品后调用此接口，更新任务链并触发继续执行

    返回流式响应，以便前端能够接收到后续的执行状态
    """
    try:
        from src.multi_agent.task_chain_storage import get_task_chain_storage
        from src.multi_agent.task_orchestrator import get_task_orchestrator
        from src.api.graph_manager import get_graph
        from src.api.formatters import format_state_update
        import json
        import asyncio

        # 1. 解析选择
        manager = get_selection_manager()
        result = await manager.resolve_selection(
            request.selection_id,
            request.selected_option_id,
        )

        # 2. 从 selection 中获取 session_id
        selection_action = await manager.get_selection(request.selection_id)
        if not selection_action:
            raise ValueError("选择不存在")

        session_id = selection_action.session_id

        # 获取任务链
        storage = get_task_chain_storage()
        task_chain = storage.get(session_id)

        if not task_chain:
            logger.warning(f"未找到任务链: session={session_id}")
            return {
                "success": True,
                "status": result.status.value,
                "selection_type": result.selection_type,
                "selected_option": result.selected_option,
                "message": "已选择商品",
            }

        # 3. 更新任务链的 context_data
        if "context_data" not in task_chain:
            task_chain["context_data"] = {}
        task_chain["context_data"]["selected_product_id"] = int(request.selected_option_id)

        # 4. 移动到下一步
        orchestrator = get_task_orchestrator()
        task_chain = orchestrator.move_to_next_step(task_chain)

        # 5. 保存更新后的任务链
        storage.save(session_id, task_chain)
        logger.info(f"已更新任务链并移动到下一步: session={session_id}, step={task_chain['current_step_index']}")

        # 6. 返回流式响应，继续执行任务链
        async def stream_response():
            """流式返回任务链执行结果"""
            try:
                # 发送选择成功消息
                yield f"data: {json.dumps({'type': 'selection_resolved', 'message': '已选择商品，正在为您创建订单...'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.1)

                # 获取graph并继续执行
                graph = await get_graph()

                # 构造一个虚拟的"继续"消息，触发graph继续执行任务链
                # 注意：这里不是用户的真实消息，而是系统内部的继续信号
                continue_message = "__CONTINUE_TASK_CHAIN__"

                # 流式执行graph
                async for state_update in graph.astream(
                    continue_message,
                    session_id=session_id,
                    stream_mode="updates"
                ):
                    # 格式化并发送状态更新
                    for node_name, node_data in state_update.items():
                        if isinstance(node_data, dict):
                            formatted = format_state_update(node_data)
                            yield f"data: {json.dumps(formatted, ensure_ascii=False)}\n\n"

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

    用户点击取消按钮后调用此接口
    """
    try:
        manager = get_selection_manager()
        result = await manager.cancel_selection(request.selection_id)

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

    前端重新连接时检查是否有待选择操作
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

