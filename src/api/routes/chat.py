"""聊天相关路由"""
import json
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.api.models import ChatRequest
from src.api.graph_manager import get_graph
from src.api.formatters import format_state_update, format_step_name, format_step_detail

logger = logging.getLogger(__name__)

router = APIRouter()


async def stream_chat_response(question: str, session_id: str):
    """流式生成聊天响应"""
    try:
        graph = await get_graph()
        accumulated_state = {}
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
                # 【关键修复】传递 node_update，只提取当前轮次的工具结果
                formatted = format_state_update(accumulated_state, node_update)
                formatted["data"]["execution_steps"] = execution_steps
                formatted["data"]["step_details"] = step_details
                yield f"data: {json.dumps(formatted, ensure_ascii=False)}\n\n"

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
                    "pending_selection": None,
                    "confirmation_pending": None,
                    "entities": {},
                    "context_data": {}
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

