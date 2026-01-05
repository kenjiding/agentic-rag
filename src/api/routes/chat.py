"""聊天相关路由"""
import json
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.api.models import ChatRequest
from src.api.graph_manager import get_graph
from src.api.formatters import format_state_update, format_step_name, format_step_detail
from src.api.streaming_utils import accumulate_and_format_state_updates

logger = logging.getLogger(__name__)

router = APIRouter()


async def stream_chat_response(question: str, session_id: str):
    """流式生成聊天响应"""
    try:
        graph = await get_graph()
        execution_steps: list[str] = []
        step_details: list[dict] = []

        def add_step(step_name: str, detail: str = "") -> bool:
            """添加执行步骤，返回是否是新步骤"""
            if step_name and step_name not in execution_steps:
                execution_steps.append(step_name)
                step_details.append({"name": step_name, "detail": detail, "status": "running"})
                # 更新之前步骤的状态为完成
                for i in range(len(step_details) - 1):
                    step_details[i]["status"] = "completed"
                return True
            return False

        # 立即发送初始状态
        initial_step = "🚀 开始分析您的问题"
        add_step(initial_step)
        yield f"data: {json.dumps({'type': 'state_update', 'data': {'execution_steps': execution_steps, 'step_details': step_details}}, ensure_ascii=False)}\n\n"

        # 配置 checkpointer（2025最佳实践：显式传递 thread_id）
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 20
        }

        # 【关键修复】在流式处理之前，先从 checkpointer 获取完整的状态作为基础
        # 这样 accumulated_state 会包含 task_chain、entities 等关键字段
        accumulated_state = {}
        try:
            existing_snapshot = graph.graph.get_state(config)
            if existing_snapshot and existing_snapshot.values:
                # 使用现有状态作为基础（保留 task_chain 等关键数据）
                # Pydantic 模型使用 model_dump() 转换为字典，不能使用 .copy()
                from src.multi_agent.utils import state_to_dict
                accumulated_state = state_to_dict(existing_snapshot.values)
                logger.info(f"从 checkpointer 初始化 accumulated_state: task_chain={'task_chain' in accumulated_state and accumulated_state.get('task_chain') is not None}")
        except Exception as e:
            logger.warning(f"从 checkpointer 获取状态失败: {e}，使用空状态初始化")

        # 【核心修复】检查并清理过期的 confirmation_pending
        # 问题根源：confirmation 可能通过 /api/confirmation/resolve 被解析，
        # 但 checkpoint 中的 confirmation_pending 没有被清除
        state_confirmation_pending = accumulated_state.get("confirmation_pending")
        if state_confirmation_pending:
            from src.confirmation import get_confirmation_manager
            confirmation_manager = get_confirmation_manager()
            confirmation_id = state_confirmation_pending.get("confirmation_id")

            # 检查 confirmation 是否仍然有效
            try:
                pending_confirmation = await confirmation_manager.get_pending_confirmation(session_id)
                if not pending_confirmation or pending_confirmation.confirmation_id != confirmation_id:
                    # confirmation 已被解析或取消，但 checkpoint 中的状态未清除
                    logger.info(f"检测到过期的 confirmation_pending (id={confirmation_id})，清理 checkpoint 状态")
                    # 更新 checkpoint 清除 confirmation_pending
                    graph.graph.update_state(
                        config,
                        {"confirmation_pending": None},
                        as_node="__start__"
                    )
                    accumulated_state["confirmation_pending"] = None
                else:
                    logger.info(f"confirmation_pending 仍然有效 (id={confirmation_id})")
            except Exception as e:
                logger.warning(f"检查 confirmation 状态失败: {e}，清理 confirmation_pending")
                # 出错时清理，避免 UI 显示过期的确认对话框
                graph.graph.update_state(
                    config,
                    {"confirmation_pending": None},
                    as_node="__start__"
                )
                accumulated_state["confirmation_pending"] = None

        # 使用 updates 模式获取每个节点的更新
        async for state_update in graph.astream(question, config=config, stream_mode="updates", session_id=session_id):
            # LangGraph 返回的格式是 {node_name: {updated_fields}}
            for node_name, node_update in state_update.items():
                # 跳过特殊节点
                if node_name in ("__start__", "__end__"):
                    continue

                # 生成步骤名称和详情
                step_name = format_step_name(node_name, node_update)
                step_detail = format_step_detail(node_name, node_update)

                if step_name:
                    is_new_step = add_step(step_name, step_detail)
                    # 如果有新步骤，立即发送
                    if is_new_step:
                        yield f"data: {json.dumps({'type': 'state_update', 'data': {'execution_steps': execution_steps, 'step_details': step_details}}, ensure_ascii=False)}\n\n"

                # 【关键修复】在更新 accumulated_state 之前，记录当前消息数量
                # 这样可以判断 node_update 中是否有新消息
                messages_before_update = len(accumulated_state.get("messages", []))

                # 累积状态
                if isinstance(node_update, dict):
                    if "messages" in node_update and "messages" in accumulated_state:
                        # 合并 messages（去重）
                        existing_messages = accumulated_state.get("messages", [])
                        new_messages = node_update.get("messages", [])
                        existing_ids = {id(msg) if hasattr(msg, 'id') else str(msg) for msg in existing_messages}
                        for msg in new_messages:
                            msg_id = id(msg) if hasattr(msg, 'id') else str(msg)
                            if msg_id not in existing_ids:
                                existing_messages.append(msg)
                                existing_ids.add(msg_id)
                        accumulated_state["messages"] = existing_messages

                        # 【关键修复】也要合并其他关键字段（tools_used, current_agent 等）
                        for key, value in node_update.items():
                            if key != "messages":
                                # tools_used 需要合并（列表追加）
                                if key == "tools_used" and value:
                                    existing_tools = accumulated_state.get("tools_used", [])
                                    accumulated_state["tools_used"] = existing_tools + value
                                else:
                                    # 其他字段直接覆盖
                                    accumulated_state[key] = value
                    else:
                        accumulated_state.update(node_update)

                # 发送状态更新
                # 【关键修复】传递 node_update 和更新前的消息数量，从源头解决问题：只提取新消息
                formatted = format_state_update(accumulated_state, node_update, messages_before_update)
                formatted["data"]["execution_steps"] = execution_steps
                formatted["data"]["step_details"] = step_details
                yield f"data: {json.dumps(formatted, ensure_ascii=False)}\n\n"

        # 【LangGraph 1.x】流结束后检查是否有 interrupt()
        # 当 interrupt() 被调用时，流正常结束，但状态保存在 checkpointer 中
        # 需要通过 get_state() 检查是否有待处理的 interrupt
        try:
            logger.info(f"[chat路由] 流结束，检查是否有 interrupt: session_id={session_id}")
            final_snapshot = graph.graph.get_state(config)
            logger.info(f"[chat路由] checkpointer snapshot: {final_snapshot is not None}, tasks存在: {final_snapshot.tasks is not None if final_snapshot else False}, tasks长度: {len(final_snapshot.tasks) if final_snapshot and final_snapshot.tasks else 0}")
            
            if final_snapshot and final_snapshot.tasks:
                logger.info(f"[chat路由] 检查 tasks 中的 interrupt: tasks数量={len(final_snapshot.tasks)}")
                # 检查是否有待处理的 interrupt（LangGraph 1.x 将 interrupt 保存在 tasks 中）
                for i, task in enumerate(final_snapshot.tasks):
                    logger.info(f"[chat路由] 检查 task[{i}]: {type(task)}")
                    # 退出循环，只处理第一个 interrupt
                    if task.interrupts:
                        break
            else:
                logger.info(f"[chat路由] 没有 interrupt 任务")
        except Exception as e:
            logger.error(f"[chat路由] 检查 interrupt 状态失败: {e}", exc_info=True)

        # 标记所有步骤为完成
        for detail in step_details:
            detail["status"] = "completed"

        # 发送最终状态
        yield f"data: {json.dumps({'type': 'state_update', 'data': {'execution_steps': execution_steps, 'step_details': step_details}}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """清除指定会话的状态

    用于重置会话，清除 checkpointer 中保存的历史消息和状态。
    当遇到消息格式错误或需要重新开始对话时使用。
    """
    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": session_id}}

        # 尝试清除状态（通过更新为空状态）
        try:
            graph.graph.update_state(
                config,
                {
                    "messages": [],
                    "task_chain": None,
                    "confirmation_pending": None,
                    "entities": {}
                },
                as_node="__start__"
            )

            logger.info(f"已清除会话状态: {session_id}")
            return {
                "success": True,
                "message": f"会话 {session_id} 的状态已清除"
            }
        except Exception as e:
            logger.warning(f"清除会话状态时出错（可能不存在）: {e}")
            return {
                "success": True,
                "message": f"会话 {session_id} 不存在或已清除"
            }

    except Exception as e:
        logger.error(f"清除会话失败: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/chat")
async def chat(request: ChatRequest):
    """聊天接口 - 支持流式响应"""
    if request.stream:
        return StreamingResponse(
            stream_chat_response(request.message, request.session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # 非流式响应（同步）
        graph = await get_graph()

        # 配置 checkpointer（2025最佳实践：显式传递 thread_id）
        config = {
            "configurable": {"thread_id": request.session_id},
            "recursion_limit": 20
        }

        final_state = await graph.ainvoke(request.message, config=config)

        # 格式化响应
        response = format_state_update(final_state)
        return response

