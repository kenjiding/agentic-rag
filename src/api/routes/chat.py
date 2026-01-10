"""聊天相关路由（2025-2026简化版）

核心原则：
- 路由层只做SSE事件封装和透传
- Agent通过ResponseModel提供完整的前端数据
- 不在路由层做状态管理、业务逻辑处理
"""
import json
import logging
from typing import cast, Any, Dict, List
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.api.models import ChatRequest
from src.api.graph_manager import get_graph
from src.api.formatters import format_step_name, format_step_detail, format_state_update

logger = logging.getLogger(__name__)


def make_json_serializable(obj: Any) -> Any:
    """递归清理对象，移除不可 JSON 序列化的字段
    
    过滤掉 LangChain 的 Document 对象和其他不可序列化的对象。
    对于包含 Document 的列表，只保留其 page_content 和 metadata（如果存在）。
    
    Args:
        obj: 要清理的对象
        
    Returns:
        可 JSON 序列化的对象
    """
    # 检查是否是 Document 对象（通过类名和属性判断）
    if hasattr(obj, '__class__'):
        class_name = obj.__class__.__name__
        # 检查是否是 LangChain Document 对象
        if class_name == 'Document' and hasattr(obj, 'page_content'):
            # 如果是 Document 对象，只返回可序列化的内容
            try:
                return {
                    "page_content": getattr(obj, 'page_content', ''),
                    "metadata": make_json_serializable(getattr(obj, 'metadata', {}))
                }
            except Exception:
                return str(obj)
        # 检查是否是其他 LangChain 消息对象（BaseMessage, AIMessage, HumanMessage等）
        elif 'Message' in class_name and hasattr(obj, 'content'):
            # 只返回 content，忽略其他不可序列化的属性
            return str(getattr(obj, 'content', ''))
    
    # 检查是否是列表
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    
    # 检查是否是字典
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    # 检查是否是基本类型（可 JSON 序列化）
    # 基本类型包括：str, int, float, bool, None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    # 尝试 JSON 序列化测试
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # 如果是不可序列化的对象，尝试转换为字符串
        try:
            return str(obj)
        except Exception:
            return None

router = APIRouter()


async def stream_chat_response(question: str, session_id: str):
    """流式生成聊天响应（2025-2026简化版）

    核心原则：
    - 路由层只做SSE事件封装和透传
    - Agent通过ResponseModel提供完整的前端数据
    - 不在路由层做状态管理、业务逻辑处理
    """
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
        add_step("🚀 开始分析您的问题")
        yield f"data: {json.dumps({'type': 'state_update', 'data': {'execution_steps': execution_steps, 'step_details': step_details}}, ensure_ascii=False)}\n\n"

        # 配置 checkpointer
        config = {
            "configurable": {
                "thread_id": session_id,
                "session_id": session_id,
            },
            "recursion_limit": 20
        }

        # 流式处理：直接透传 node_update
        async for state_update in graph.astream(question, config=config, stream_mode="updates", session_id=session_id):
            for node_name, node_update in state_update.items():
                # 跳过特殊节点
                if node_name in ("__start__", "__end__"):
                    continue

                # 【核心修复】处理 __interrupt__ 节点
                # 当 interrupt() 被调用时，LangGraph 生成一个 __interrupt__ 节点
                # node_update 是一个 tuple，包含 Interrupt 对象
                if node_name == "__interrupt__":
                    # 从 tuple 中提取 Interrupt 对象
                    interrupt_value = None
                    if isinstance(node_update, tuple) and len(node_update) > 0:
                        interrupt_obj = node_update[0]
                        if hasattr(interrupt_obj, 'value'):
                            interrupt_value = interrupt_obj.value
                        elif isinstance(interrupt_obj, dict):
                            interrupt_value = interrupt_obj
                    elif isinstance(node_update, dict):
                        interrupt_value = node_update
                    
                    # 格式化并发送 interrupt 信息
                    if isinstance(interrupt_value, dict):
                        # 构建前端数据，包含所有必要的信息
                        frontend_data = {
                            k: make_json_serializable(v) for k, v in interrupt_value.items()
                            if k != 'messages' and k != 'result' and k != 'agent_results'
                        }
                        # 添加步骤信息
                        frontend_data["execution_steps"] = execution_steps
                        frontend_data["step_details"] = step_details
                        # 确保 response_type 设置为 confirmation
                        if "response_type" not in frontend_data:
                            frontend_data["response_type"] = "confirmation"
                        
                        yield f"data: {json.dumps({'type': 'state_update', 'data': frontend_data}, ensure_ascii=False)}\n\n"
                        continue  # 跳过后续处理

                # 格式化步骤名称和详情
                step_name = format_step_name(node_name, node_update)
                step_detail = format_step_detail(node_name, node_update)

                if step_name:
                    is_new_step = add_step(step_name, step_detail)
                    # 如果有新步骤，立即发送
                    if is_new_step:
                        yield f"data: {json.dumps({'type': 'state_update', 'data': {'execution_steps': execution_steps, 'step_details': step_details}}, ensure_ascii=False)}\n\n"

                # 直接透传状态更新（Agent已通过ResponseModel提供完整数据）
                if isinstance(node_update, dict):
                    # 排除无法序列化的字段（messages包含LangChain对象，result可能包含Document对象）
                    # 前端只需要：response_type, content, role, response_data, current_agent, tools_used, metadata等
                    frontend_data = {
                        k: make_json_serializable(v) for k, v in node_update.items()
                        if k != 'messages' and k != 'result' and k != 'agent_results'
                    }
                    # 添加步骤信息到响应中
                    frontend_data["execution_steps"] = execution_steps
                    frontend_data["step_details"] = step_details
                    # 直接透传给前端
                    yield f"data: {json.dumps({'type': 'state_update', 'data': frontend_data}, ensure_ascii=False)}\n\n"

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

        # 清除状态
        try:
            graph.graph.update_state(
                config,
                {
                    "messages": [],
                    "confirmation_pending": None,
                    "entities": {},
                    "last_product_search_context": None
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

        # 配置 checkpointer
        config = {
            "configurable": {"thread_id": request.session_id},
            "recursion_limit": 20
        }

        final_state = await graph.ainvoke(request.message, config=config)

        # 格式化响应（非流式模式）
        # MultiAgentState是Pydantic模型，转换为dict
        if hasattr(final_state, 'model_dump'):
            final_state_dict = final_state.model_dump()
        else:
            final_state_dict = cast(dict, final_state)
        response = format_state_update(final_state_dict)
        return response
