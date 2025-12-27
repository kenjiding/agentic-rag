"""确认相关路由"""
import logging
import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.models import ConfirmationResolveRequest
from src.confirmation import (
    get_confirmation_manager,
    ConfirmationNotFoundError,
    ConfirmationExpiredError,
    ConfirmationAlreadyResolvedError,
)
from src.api.formatters import format_state_update

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/confirmation/resolve")
async def resolve_confirmation(request: ConfirmationResolveRequest):
    """解析确认操作（2025最佳实践：支持任务链模式）

    用户点击确认/取消按钮后调用此接口。

    如果是在任务链模式下，确认后会触发 graph 继续执行任务链的下一步。

    返回流式响应，以便前端能够接收到后续的执行状态。
    """
    try:
        from src.api.graph_manager import get_graph

        # 1. 解析确认
        manager = get_confirmation_manager()
        result = await manager.resolve_confirmation(
            request.confirmation_id,
            request.confirmed,
        )

        # 2. 从 confirmation 中获取 session_id
        confirmation = await manager.get_confirmation(request.confirmation_id)
        if not confirmation:
            raise ValueError("确认不存在")

        session_id = confirmation.session_id

        # 3. 检查是否在任务链模式下
        graph = await get_graph()
        config = {"configurable": {"thread_id": session_id}}

        try:
            # get_state 是同步方法，不需要 await
            current_state = graph.graph.get_state(config)
            task_chain = current_state.values.get("task_chain")

            # 如果不是任务链模式，直接返回结果
            if not task_chain:
                logger.info(f"非任务链模式，直接返回确认结果: session={session_id}")
                return {
                    "success": True,
                    "status": result.status,
                    "action_type": result.action_type,
                    "message": result.execution_result.get("text", "操作已完成") if result.execution_result else "操作已取消",
                    "data": result.execution_result,
                    "error": result.error,
                }
        except Exception as e:
            logger.warning(f"从 checkpointer 获取状态失败: {e}")
            # 降级：直接返回确认结果
            return {
                "success": True,
                "status": result.status,
                "action_type": result.action_type,
                "message": result.execution_result.get("text", "操作已完成") if result.execution_result else "操作已取消",
                "data": result.execution_result,
                "error": result.error,
            }

        # 4. 任务链模式：更新任务链状态并继续执行
        logger.info(f"任务链模式：确认操作已执行，更新任务链状态: session={session_id}")
        
        # 如果用户确认了，需要标记当前步骤为完成并移动到下一步
        # 注意：由于 use_enum_values=True，result.status 是字符串而不是枚举对象
        from src.confirmation import ConfirmationStatus
        status_value = result.status if isinstance(result.status, str) else result.status.value
        if request.confirmed and status_value == "confirmed":
            from src.multi_agent.task_orchestrator import get_task_orchestrator
            
            # 获取当前任务链
            current_index = task_chain["current_step_index"]
            steps = task_chain["steps"]
            
            if current_index < len(steps):
                current_step = steps[current_index]
                # 标记当前步骤为完成
                current_step.update({
                    "status": "completed",
                    "result_data": {
                        "message": result.execution_result.get("text", "操作已完成") if result.execution_result else "操作已完成"
                    }
                })
                
                # 移动到下一步
                orchestrator = get_task_orchestrator()
                task_chain = orchestrator.move_to_next_step(task_chain)
                
                logger.info(
                    f"任务链步骤已更新: 步骤 {current_index} 已完成, "
                    f"当前步骤索引={task_chain['current_step_index']}, "
                    f"总步骤数={len(task_chain['steps'])}"
                )
                
                # 更新 checkpointer 中的任务链状态
                update_config = {"configurable": {"thread_id": session_id}}
                state_update = {"task_chain": task_chain}
                graph.graph.update_state(update_config, state_update, as_node="task_orchestrator")
                
                # 如果任务链已完成，不需要继续执行
                if task_chain["current_step_index"] >= len(task_chain["steps"]):
                    logger.info("任务链已完成，无需继续执行")
                    # 清除任务链
                    graph.graph.update_state(update_config, {"task_chain": None}, as_node="task_orchestrator")
                    # 返回完成消息
                    return StreamingResponse(
                        iter([f"data: {json.dumps({'type': 'confirmation_resolved', 'message': '订单已创建，任务完成！'}, ensure_ascii=False)}\n\n"
                              f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"]),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )

        # 返回流式响应，继续执行任务链
        async def stream_response():
            """流式返回任务链执行结果"""
            try:
                # 发送确认成功消息
                action_text = "已确认" if request.confirmed else "已取消"
                yield f"data: {json.dumps({'type': 'confirmation_resolved', 'message': f'{action_text}，正在处理...'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.1)

                # 使用特殊消息触发 graph 继续执行任务链
                # graph.astream 期望字符串参数，而不是 HumanMessage 对象
                continue_message = "__TASK_CHAIN_CONTINUE__"

                # 流式执行graph（传递正确的 config）
                stream_config = {
                    "configurable": {"thread_id": session_id},
                    "recursion_limit": 20
                }

                async for state_update in graph.astream(
                    continue_message,
                    config=stream_config,
                    stream_mode="updates",
                    session_id=session_id
                ):
                    # 格式化并发送状态更新
                    for node_name, node_data in state_update.items():
                        if isinstance(node_data, dict):
                            try:
                                formatted = format_state_update(node_data)
                                # 确保格式化后的数据是有效的字典
                                if isinstance(formatted, dict):
                                    json_str = json.dumps(formatted, ensure_ascii=False)
                                    yield f"data: {json_str}\n\n"
                            except Exception as e:
                                logger.warning(f"格式化状态更新失败 (node={node_name}): {e}")
                                # 发送错误信息而不是跳过
                                error_data = {
                                    "type": "error",
                                    "message": f"处理节点 {node_name} 的状态更新时出错"
                                }
                                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

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

