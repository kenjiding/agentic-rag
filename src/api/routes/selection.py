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
    """解析用户选择（2025最佳实践：使用 checkpointer）

    用户选择产品后调用此接口，更新任务链并触发继续执行

    返回流式响应，以便前端能够接收到后续的执行状态
    """
    try:
        from src.multi_agent.task_orchestrator import get_task_orchestrator
        from src.api.graph_manager import get_graph
        from src.api.formatters import format_state_update
        import json
        import asyncio
        from langchain_core.messages import HumanMessage

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

        # 3. 从 checkpointer 获取当前 state（包括 task_chain）
        graph = await get_graph()
        config = {"configurable": {"thread_id": session_id}}

        try:
            # 使用 checkpointer 获取当前状态（get_state 是同步方法，不需要 await）
            current_state = graph.graph.get_state(config)
            task_chain = current_state.values.get("task_chain")

            if not task_chain:
                logger.warning(f"未找到任务链: session={session_id}")
                return {
                    "success": True,
                    "status": result.status.value,
                    "selection_type": result.selection_type,
                    "selected_option": result.selected_option,
                    "message": "已选择商品",
                }
        except Exception as e:
            logger.warning(f"从 checkpointer 获取状态失败: {e}")
            return {
                "success": True,
                "status": result.status.value,
                "selection_type": result.selection_type,
                "selected_option": result.selected_option,
                "message": "已选择商品",
            }

        # 4. 更新任务链和 entities
        orchestrator = get_task_orchestrator()

        # 更新 context_data（向后兼容）
        if "context_data" not in task_chain:
            task_chain["context_data"] = {}
        task_chain["context_data"]["selected_product_id"] = int(request.selected_option_id)

        # 更新 entities（2025最佳实践）
        entities = current_state.values.get("entities", {})
        entities["selected_product_id"] = int(request.selected_option_id)

        # 5. 移动到下一步
        task_chain = orchestrator.move_to_next_step(task_chain)

        logger.info(
            f"已更新任务链并移动到下一步: session={session_id}, "
            f"step={task_chain['current_step_index']}, "
            f"selected_product_id={request.selected_option_id}"
        )

        # 6. 返回流式响应，继续执行任务链
        async def stream_response():
            """流式返回任务链执行结果"""
            try:
                # 发送选择成功消息
                yield f"data: {json.dumps({'type': 'selection_resolved', 'message': '已选择商品，正在为您创建订单...'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.1)

                # 更新 state 中的 task_chain 和 entities
                state_update = {
                    "task_chain": task_chain,
                    "entities": entities
                }

                # 使用 checkpointer 更新状态（as_node=task_orchestrator 表示是 task_orchestrator 更新的）
                # update_state 是同步方法，不需要 await
                # 确保使用与 astream 相同的 config
                update_config = {
                    "configurable": {"thread_id": session_id}
                }
                graph.graph.update_state(update_config, state_update, as_node="task_orchestrator")

                logger.info(f"已更新 checkpointer 状态: task_chain step={task_chain['current_step_index']}, thread_id={session_id}")
                
                # 验证状态更新是否成功
                verify_state = graph.graph.get_state(update_config)
                if verify_state and verify_state.values:
                    logger.info(f"状态验证: task_chain 存在={verify_state.values.get('task_chain') is not None}")

                # 使用特殊消息触发 graph 继续执行任务链
                # graph.astream 期望字符串参数，而不是 HumanMessage 对象
                continue_message = "__TASK_CHAIN_CONTINUE__"

                # 流式执行graph（传递正确的 config，必须与 update_state 使用相同的 thread_id）
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

    【关键修复】取消选择时，也要清除 checkpointer 中的 task_chain
    避免用户后续请求被旧任务链阻塞
    """
    try:
        manager = get_selection_manager()

        # 【关键修复】取消选择前，先获取 selection 以获取 session_id
        selection = await manager.get_selection(request.selection_id)
        if not selection:
            raise ValueError(f"选择 {request.selection_id} 不存在")

        session_id = selection.session_id

        # 执行取消
        result = await manager.cancel_selection(request.selection_id)

        # 【关键修复】清除 checkpointer 中的 task_chain
        # 如果不清除，用户后续的请求会被旧任务链阻塞
        from src.api.graph_manager import get_graph

        graph = await get_graph()

        if session_id:
            config = {"configurable": {"thread_id": session_id}}
            try:
                # 获取当前状态
                current_state = graph.graph.get_state(config)
                if current_state and current_state.values:
                    task_chain = current_state.values.get("task_chain")
                    if task_chain:
                        # 清除 task_chain
                        graph.graph.update_state(config, {"task_chain": None}, as_node="task_orchestrator")
                        logger.info(f"[CANCEL] 已清除 task_chain: session_id={session_id}, chain_id={task_chain.get('chain_id')}")
            except Exception as e:
                logger.warning(f"[CANCEL] 清除 task_chain 失败: {e}")

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

