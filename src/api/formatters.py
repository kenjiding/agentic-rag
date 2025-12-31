"""状态格式化和步骤格式化工具

核心原则：
1. 只提取当前节点实际调用的工具结果
2. 避免历史数据污染
3. 用户问什么就返回什么，不混入其他数据
"""
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _extract_tool_results_from_messages(messages: list, expected_data_types: list = None) -> Dict[str, Any]:
    """从消息列表中提取工具结果

    只提取最新的 ToolMessage 中的结构化数据。

    Args:
        messages: 消息列表
        expected_data_types: 期望的数据类型列表，如 ["orders"] 或 ["products"]
                           如果为 None，则提取所有类型

    Returns:
        包含 products/orders 等数据的字典
    """
    if not messages:
        return {}

    from langchain_core.messages import ToolMessage

    # 倒序遍历，找到第一个有效的 ToolMessage 即停止
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue

        try:
            content = message.content
            if not isinstance(content, str):
                continue

            tool_result = json.loads(content)
            if not isinstance(tool_result, dict):
                continue

            # 根据 expected_data_types 过滤提取
            results = {}
            data_types = expected_data_types or ["products", "orders"]

            for data_type in data_types:
                if data_type in tool_result and tool_result[data_type]:
                    results[data_type] = tool_result[data_type]

            if results:
                return results

        except (json.JSONDecodeError, TypeError):
            continue

    return {}


def _determine_expected_data_types(tools_used: list, current_agent: Optional[str] = None) -> list:
    """根据当前 agent 和工具调用确定期望的数据类型

    Args:
        tools_used: 当前节点使用的工具列表
        current_agent: 当前 agent 名称

    Returns:
        期望的数据类型列表，如 ["orders"] 或 ["products"]
    """
    # 优先根据 current_agent 确定数据类型
    if current_agent == "order_agent":
        return ["orders"]
    if current_agent == "product_agent":
        return ["products"]

    # 降级：根据最后一个工具调用判断
    if tools_used:
        order_tools = {
            "query_user_orders", "query_order_detail", "prepare_create_order",
            "confirm_create_order", "prepare_cancel_order", "confirm_cancel_order"
        }
        product_tools = {
            "search_products_tool", "get_product_detail", "get_brands", "get_categories"
        }

        last_tool = tools_used[-1] if tools_used else None
        if last_tool and isinstance(last_tool, dict):
            tool_name = last_tool.get("tool", "")
            if tool_name in order_tools:
                return ["orders"]
            if tool_name in product_tools:
                return ["products"]

    return []


def format_state_update(state_update: Dict[str, Any], node_update: Any = None, messages_count_before_update: int = 0) -> Dict[str, Any]:
    """格式化状态更新为前端友好的格式

    核心原则：
    - 只提取当前节点新产生的 ToolMessage 数据
    - 如果当前节点没有产生新的 ToolMessage，response_data 保持为空
    - 避免任何历史数据污染
    - 支持购买流程中的产品选择列表

    Args:
        state_update: 完整的累积状态（包含历史消息）
        node_update: 当前节点的更新（只包含当前轮次的变化）
        messages_count_before_update: 更新前 state_update 中的消息数量，用于判断是否有新消息

    Returns:
        格式化后的响应数据
    """
    result = {
        "type": "state_update",
        "data": {
            "response_type": "text",
            "response_data": {}
        }
    }

    # 1. 提取工具结果（仅当 node_update 有新的工具调用时）
    if node_update and isinstance(node_update, dict):
        tools_used = node_update.get("tools_used", [])
        node_messages = node_update.get("messages", [])

        # 判断是否有有效的新工具调用
        new_tools = [t for t in tools_used if t and t.get("tool")]
        if new_tools and node_messages:
            expected_data_types = _determine_expected_data_types(
                tools_used,
                node_update.get("current_agent")
            )
            tool_results = _extract_tool_results_from_messages(node_messages, expected_data_types)

            if "products" in tool_results:
                result["data"]["response_data"]["products"] = tool_results["products"]
            if "orders" in tool_results:
                result["data"]["response_data"]["orders"] = tool_results["orders"]

    # 2. 提取新增的 AI 消息内容
    messages = state_update.get("messages", [])
    new_messages = messages[messages_count_before_update:]
    if new_messages:
        from langchain_core.messages import AIMessage
        ai_messages = [msg for msg in new_messages if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_message = ai_messages[-1]
            if hasattr(last_ai_message, "content") and last_ai_message.content:
                result["data"]["content"] = last_ai_message.content
                result["data"]["role"] = "assistant"

    # 3. 处理 task_chain 中的 user_selection 步骤
    # 当 task_chain 存在且当前步骤是 user_selection 时：
    #   从 state_update 中获取 pending_selection（由 _execute_user_selection 设置）
    #   如果有 pending_selection，设置 response_type 为 selection

    task_chain = state_update.get("task_chain")
    if task_chain:
        # 辅助函数：兼容 Pydantic 模型和字典获取步骤属性
        def _get_step_attr(step, attr, default=None):
            if hasattr(step, attr):
                return getattr(step, attr)
            elif isinstance(step, dict):
                return step.get(attr, default)
            return default

        # task_chain 可能是 Pydantic 模型或字典，需要兼容处理
        if hasattr(task_chain, 'model_dump'):
            current_step_index = task_chain.current_step_index
            steps = task_chain.steps
        else:
            current_step_index = task_chain.get("current_step_index")
            steps = task_chain.get("steps", [])

        if current_step_index is not None and current_step_index < len(steps):
            current_step = steps[current_step_index]
            step_type = _get_step_attr(current_step, "step_type")

            # 检测是否是 user_selection 步骤
            if step_type == "user_selection":
                logger.info(f"检测到 task_chain 中的 user_selection 步骤，index={current_step_index}")

    # 4. 处理特殊状态（从 state_update 中获取 pending_selection）
    # 如果 result 中还没有设置 pending_selection，则从 state_update 中获取
    if result["data"].get("pending_selection") is None:
        pending_selection = state_update.get("pending_selection")
        if pending_selection:
            result["data"]["pending_selection"] = pending_selection
            result["data"]["response_type"] = "selection"
            # 移除重复的 products 数据
            result["data"]["response_data"].pop("products", None)

    # node_update 可能是 tuple（当 interrupt() 被调用时），需要类型检查
    node_confirmation = node_update.get("confirmation_pending") if isinstance(node_update, dict) else None
    confirmation_pending = node_confirmation or state_update.get("confirmation_pending")

    if confirmation_pending:
        result["data"]["confirmation_pending"] = confirmation_pending
        result["data"]["response_type"] = "confirmation"

        # 订单确认时，构建订单信息供前端使用
        if confirmation_pending.get("action_type") == "create_order":
            display_data = confirmation_pending.get("display_data", {})
            if display_data:
                result["data"]["response_data"]["order"] = {
                    "items": display_data.get("items", []),
                    "total_amount": display_data.get("total_amount", 0),
                    "user_phone": confirmation_pending.get("action_data", {}).get("user_phone", "")
                }

    # 5. 确定响应类型（仅在没有特殊状态时）
    # 检查 result 中是否有特殊状态，而不是检查 state_update
    has_pending_selection = result["data"].get("pending_selection") is not None
    has_confirmation_pending = result["data"].get("confirmation_pending") is not None

    if not has_pending_selection and not has_confirmation_pending:
        response_data = result["data"]["response_data"]
        if "orders" in response_data:
            result["data"]["response_type"] = "order_list"
        elif "products" in response_data:
            result["data"]["response_type"] = "product_list"

    # 6. 添加其他元信息
    if current_agent := state_update.get("current_agent"):
        result["data"]["current_agent"] = current_agent
    if tools_used := state_update.get("tools_used", []):
        result["data"]["tools_used"] = tools_used

    return result


def format_step_name(node_name: str, node_update: Any) -> Optional[str]:
    """格式化执行步骤名称"""
    step_map = {
        "intent_recognition": "🎯 意图识别",
        "supervisor": "🧠 路由决策",
        "rag_agent": "📚 知识检索",
        "chat_agent": "💬 对话处理",
        "product_agent": "🛍️ 商品搜索",
        "order_agent": "📦 订单管理",
        "task_orchestrator": "🔗 任务编排",
    }

    # 检查是否有路由决策信息
    if node_name == "supervisor" and isinstance(node_update, dict):
        selected_agent = node_update.get("current_agent")
        if selected_agent:
            agent_name = step_map.get(selected_agent, selected_agent)
            return f"🧠 路由到: {agent_name}"

    return step_map.get(node_name)


def format_step_detail(node_name: str, node_update: Any) -> str:
    """格式化执行步骤的详细描述"""
    detail_map = {
        "intent_recognition": "正在分析您的问题意图...",
        "supervisor": "智能路由正在选择最合适的助手...",
        "rag_agent": "正在从知识库中检索相关信息...",
        "chat_agent": "正在生成回答...",
        "product_agent": "正在搜索商品信息...",
        "order_agent": "正在查询订单信息...",
        "task_orchestrator": "正在协调多步骤任务...",
    }

    # 特殊处理：supervisor 路由决策
    if node_name == "supervisor" and isinstance(node_update, dict):
        selected_agent = node_update.get("current_agent")
        routing_reason = node_update.get("routing_reason", "")
        if selected_agent:
            agent_descriptions = {
                "rag_agent": "知识库检索助手",
                "chat_agent": "智能对话助手",
                "product_agent": "商品搜索助手",
                "order_agent": "订单管理助手",
            }
            desc = agent_descriptions.get(selected_agent, selected_agent)
            if routing_reason:
                return f"已选择 {desc}，原因：{routing_reason[:50]}..."
            return f"已选择 {desc}"

    # 检查是否有工具调用信息
    if isinstance(node_update, dict):
        tools_used = node_update.get("tools_used", [])
        if tools_used:
            tool_names = [t.get("tool", "").split("_")[-1] for t in tools_used if t.get("tool")]
            if tool_names:
                return f"正在使用工具：{', '.join(tool_names)}"

    return detail_map.get(node_name, "正在处理...")
