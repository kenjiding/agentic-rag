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

        # 1. 记录用户选择到 selection_manager（会验证选项ID并更新状态）
        manager = get_selection_manager()
        result = await manager.resolve_selection(
            request.selection_id,
            request.selected_option_id,
        )

        # 验证选择结果状态
        if result.status.value != "selected":
            raise ValueError(f"选择状态异常: {result.status.value}")

        # 2. 获取 session_id 以便恢复执行
        # SelectionAction 本身就包含 session_id，直接从 selection_action 获取
        selection_action = await manager.get_selection(request.selection_id)
        if not selection_action:
            raise ValueError(f"选择 {request.selection_id} 不存在")
        
        session_id = selection_action.session_id

        # 记录选择结果（包含完整的选项数据）
        logger.info(
            f"用户选择已解析: selection_id={request.selection_id}, "
            f"selected_option_id={request.selected_option_id}, "
            f"selection_type={result.selection_type}, "
            f"session_id={session_id}, "
            f"selected_option={result.selected_option}"
        )

        # 3. 准备恢复执行
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
    2. 清理任务链、确认数据和相关实体，避免污染后续运行环境
    3. 返回成功响应，前端关闭对话框

    设计说明：
    - 取消操作不需要恢复图执行
    - 用户未做出选择，需要清理所有相关状态数据
    - 符合 LangGraph 1.x 最佳实践：清理状态以避免数据污染
    """
    try:
        manager = get_selection_manager()

        # 1. 执行取消
        result = await manager.cancel_selection(request.selection_id)

        logger.info(f"选择已取消: selection_id={request.selection_id}, status={result.status.value}")

        # 2. 获取 session_id 以便清理状态
        # SelectionAction 本身就包含 session_id，直接从 selection_action 获取
        selection_action = await manager.get_selection(request.selection_id)
        if not selection_action:
            raise ValueError(f"选择 {request.selection_id} 不存在")
        
        session_id = selection_action.session_id

        # 3. 清理任务链、确认数据和相关实体（如果找到 session_id）
        if session_id:
            try:
                graph = await get_graph()
                config = {"configurable": {"thread_id": session_id, "session_id": session_id}}

                # 获取当前状态，以便安全地清理相关字段
                try:
                    from src.multi_agent.utils import state_to_dict
                    
                    existing_snapshot = graph.graph.get_state(config)
                    current_state = {}
                    if existing_snapshot and existing_snapshot.values:
                        current_state = state_to_dict(existing_snapshot.values)
                    
                    # 清理 entities 中的相关字段（保留其他字段）
                    # 注意：只清理选择操作产生的实体（selected_product_id），
                    # 保留用户原始意图的实体（quantity、search_keyword），
                    # 这样用户取消选择后再次购买时，仍能保留之前的购买意图
                    entities = current_state.get("entities", {}).copy()
                    entities.pop("selected_product_id", None)  # 只清理选择操作产生的实体
                    # 不清理 quantity 和 search_keyword，这些是用户原始意图的一部分

                    # 更新状态：清理任务链、确认数据和相关实体
                    graph.graph.update_state(
                        config,
                        {
                            "task_chain": None,
                            "confirmation_pending": None,
                            "pending_selection": None,
                            "entities": entities,
                            "next_action": "finish",
                        },
                        as_node="__start__"
                    )

                    logger.info(
                        f"已清理状态数据: session_id={session_id}, "
                        f"清理了 task_chain, confirmation_pending, 和 entities 中的 selected_product_id "
                        f"（保留了用户原始意图的实体：quantity={entities.get('quantity')}, search_keyword={entities.get('search_keyword')}）"
                    )
                except Exception as state_error:
                    # 状态清理失败不应该影响取消操作的返回
                    logger.warning(f"清理状态时出错（可能状态不存在）: {state_error}")

            except Exception as e:
                # Graph 操作失败不应该影响取消操作的返回
                logger.warning(f"清理状态时出错: {e}")

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
