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

    【关键改进】：
    1. 如果指定了 expected_data_types，只提取匹配的数据类型
    2. 从最新的消息开始，只处理最近的 ToolMessage
    3. 避免历史数据污染

    Args:
        messages: 消息列表
        expected_data_types: 期望的数据类型列表，如 ["orders"] 或 ["products"]
                           如果为 None，则提取所有类型

    Returns:
        包含 products/orders 等数据的字典
    """
    results = {}

    if not messages:
        return results

    from langchain_core.messages import ToolMessage

    # 倒序遍历，找到第一个有效的 ToolMessage 即停止
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            try:
                tool_content = message.content
                if isinstance(tool_content, str):
                    try:
                        tool_result = json.loads(tool_content)
                    except:
                        continue

                    if isinstance(tool_result, dict):
                        # 【改进】根据 expected_data_types 过滤
                        if expected_data_types:
                            # 只提取期望的数据类型
                            if "products" in expected_data_types and "products" in tool_result:
                                products = tool_result.get("products", [])
                                if products:
                                    results["products"] = products
                            if "orders" in expected_data_types and "orders" in tool_result:
                                orders = tool_result.get("orders", [])
                                if orders:
                                    results["orders"] = orders
                        else:
                            # 提取所有类型
                            if "products" in tool_result:
                                products = tool_result.get("products", [])
                                if products:
                                    results["products"] = products
                            if "orders" in tool_result:
                                orders = tool_result.get("orders", [])
                                if orders:
                                    results["orders"] = orders

                        # 找到有效数据后立即停止
                        if results:
                            break
            except Exception:
                continue

    return results


def _determine_expected_data_types(tools_used: list, current_agent: str = None) -> list:
    """根据当前 agent 和工具调用确定期望的数据类型

    【关键改进】：
    1. 优先使用 current_agent 来确定数据类型（最可靠）
    2. tools_used 可能包含历史累积的工具调用，不太可靠

    Args:
        tools_used: 当前节点使用的工具列表（注意：可能是累积的）
        current_agent: 当前 agent 名称

    Returns:
        期望的数据类型列表，如 ["orders"] 或 ["products"]
    """
    # 【优先】根据 current_agent 确定数据类型（最可靠的判断依据）
    if current_agent:
        if current_agent == "order_agent":
            logger.info(f"根据 current_agent={current_agent} 确定期望数据类型: ['orders']")
            return ["orders"]
        elif current_agent == "product_agent":
            logger.info(f"根据 current_agent={current_agent} 确定期望数据类型: ['products']")
            return ["products"]

    # 降级：根据最后一个工具调用判断（只看最后一个，避免历史污染）
    if tools_used:
        order_tools = ["query_user_orders", "query_order_detail", "prepare_create_order", "confirm_create_order", "prepare_cancel_order", "confirm_cancel_order"]
        product_tools = ["search_products_tool", "get_product_detail", "get_brands", "get_categories"]

        # 只看最后一个工具调用
        last_tool = tools_used[-1] if tools_used else None
        if last_tool:
            tool_name = last_tool.get("tool", "") if isinstance(last_tool, dict) else ""
            if tool_name in order_tools:
                logger.info(f"根据最后一个工具 {tool_name} 确定期望数据类型: ['orders']")
                return ["orders"]
            if tool_name in product_tools:
                logger.info(f"根据最后一个工具 {tool_name} 确定期望数据类型: ['products']")
                return ["products"]

    logger.info("无法确定期望数据类型，返回空列表")
    return []


def format_state_update(state_update: Dict[str, Any], node_update: Dict[str, Any] = None) -> Dict[str, Any]:
    """格式化状态更新为前端友好的格式

    核心原则：
    - 只提取当前节点新产生的 ToolMessage 数据
    - 如果当前节点没有产生新的 ToolMessage，response_data 保持为空
    - 避免任何历史数据污染

    Args:
        state_update: 完整的累积状态（包含历史消息）
        node_update: 当前节点的更新（只包含当前轮次的变化）

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

    # 【调试】打印关键信息
    task_chain = state_update.get("task_chain")
    if task_chain:
        logger.info(f"[DEBUG] 当前有活跃任务链: chain_id={task_chain.get('chain_id')}, current_step={task_chain.get('current_step_index')}")
    else:
        logger.info("[DEBUG] 无活跃任务链")

    if node_update and isinstance(node_update, dict):
        logger.info(f"[DEBUG] node_update keys: {list(node_update.keys())}")
        logger.info(f"[DEBUG] node_update.tools_used: {node_update.get('tools_used', [])}")
        logger.info(f"[DEBUG] node_update.messages count: {len(node_update.get('messages', []))}")

    has_products = False
    has_orders = False
    has_new_tool_messages = False  # 标记当前节点是否产生了新的 ToolMessage

    # 【核心逻辑】判断当前节点是否产生了新的 ToolMessage
    if node_update and isinstance(node_update, dict):
        # 方法1：检查 tools_used（最可靠的判断依据）
        new_tools_used = node_update.get("tools_used", [])
        if new_tools_used:
            actual_tools = [t for t in new_tools_used if t and t.get("tool")]
            if actual_tools:
                has_new_tool_messages = True
                tool_names = [t.get("tool") for t in actual_tools]
                logger.info(f"当前节点有新的工具调用: {tool_names}")

        # 方法2：检查 node_update 的 messages 中是否有 ToolMessage
        # 即使没有 tools_used，如果有新的 ToolMessage，也应该提取数据
        if not has_new_tool_messages:
            node_messages = node_update.get("messages", [])
            from langchain_core.messages import ToolMessage
            for msg in node_messages:
                if isinstance(msg, ToolMessage):
                    has_new_tool_messages = True
                    logger.info("当前节点有新的 ToolMessage（即使没有 tools_used）")
                    break

        # 只有确认当前节点产生了新的 ToolMessage，才提取数据
        if has_new_tool_messages:
            node_messages = node_update.get("messages", [])
            if node_messages:
                # 【关键改进】根据当前工具调用确定期望的数据类型
                new_tools_used = node_update.get("tools_used", [])
                current_agent = node_update.get("current_agent")
                expected_data_types = _determine_expected_data_types(new_tools_used, current_agent)

                logger.info(f"当前节点工具: {[t.get('tool') for t in new_tools_used if t]}, agent: {current_agent}, 期望数据类型: {expected_data_types}")

                # 使用期望的数据类型过滤，避免历史数据污染
                tool_results = _extract_tool_results_from_messages(node_messages, expected_data_types)

                if "products" in tool_results:
                    result["data"]["response_data"]["products"] = tool_results["products"]
                    has_products = True
                if "orders" in tool_results:
                    result["data"]["response_data"]["orders"] = tool_results["orders"]
                    has_orders = True

                logger.info(f"从当前节点提取到工具结果: products={has_products}, orders={has_orders}")
        else:
            # 当前节点没有产生新的 ToolMessage，确保 response_data 为空
            logger.info("当前节点无新 ToolMessage，不提取任何工具结果")

    # 提取文本内容（从完整状态中获取最后一条 AI 消息）
    messages = state_update.get("messages", [])
    if messages:
        from langchain_core.messages import AIMessage

        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_message = ai_messages[-1]
            if hasattr(last_ai_message, "content") and last_ai_message.content:
                result["data"]["content"] = last_ai_message.content
                result["data"]["role"] = "assistant"

    # 提取选择等待信息（优先处理）
    pending_selection = state_update.get("pending_selection")
    if pending_selection:
        result["data"]["pending_selection"] = pending_selection
        result["data"]["response_type"] = "selection"
        # 当有pending_selection时，不在response_data中重复包含products
        if "products" in result["data"]["response_data"]:
            del result["data"]["response_data"]["products"]
            has_products = False

    # 提取确认等待信息
    if confirmation_pending := state_update.get("confirmation_pending"):
        result["data"]["confirmation_pending"] = confirmation_pending
        result["data"]["response_type"] = "confirmation"

    # 确定响应类型（仅在没有pending_selection和confirmation_pending时）
    if not pending_selection and not confirmation_pending:
        if has_orders:
            result["data"]["response_type"] = "order_list"
        elif has_products:
            result["data"]["response_type"] = "product_list"

    # 提取其他信息
    if current_agent := state_update.get("current_agent"):
        result["data"]["current_agent"] = current_agent
    if tools_used := state_update.get("tools_used", []):
        result["data"]["tools_used"] = tools_used

    return result


def format_step_name(node_name: str, node_update: Dict[str, Any]) -> Optional[str]:
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


def format_step_detail(node_name: str, node_update: Dict[str, Any]) -> str:
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
