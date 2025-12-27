"""状态格式化和步骤格式化工具"""
import json
from typing import Dict, Any, Optional


def format_state_update(state_update: Dict[str, Any]) -> Dict[str, Any]:
    """格式化状态更新为前端友好的格式

    返回统一的结构化响应：
    - 有结构化数据时：content 为简短描述，数据在 response_data 中
    - 无结构化数据时：content 为 AI 生成的完整回复
    """
    result = {
        "type": "state_update",
        "data": {
            "response_type": "text",
            "response_data": {}
        }
    }

    # 提取消息 - 查找最后一条 AI 消息
    messages = state_update.get("messages", [])
    has_products = False
    has_orders = False

    if messages:
        from langchain_core.messages import AIMessage, ToolMessage

        # 先提取工具结果中的结构化数据
        for message in messages:
            if isinstance(message, ToolMessage):
                try:
                    tool_content = message.content
                    if isinstance(tool_content, str):
                        try:
                            tool_result = json.loads(tool_content)
                        except:
                            continue

                        if isinstance(tool_result, dict):
                            if "products" in tool_result:
                                products = tool_result.get("products", [])
                                if products:
                                    result["data"]["response_data"]["products"] = products
                                    has_products = True
                            if "orders" in tool_result:
                                orders = tool_result.get("orders", [])
                                if orders:
                                    result["data"]["response_data"]["orders"] = orders
                                    has_orders = True
                except Exception:
                    pass

        # 提取文本内容
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_message = ai_messages[-1]
            if hasattr(last_ai_message, "content") and last_ai_message.content:
                ai_content = last_ai_message.content

                # 当有结构化数据时，content 已经是工具的简短描述
                # 直接使用即可，无需额外处理
                result["data"]["content"] = ai_content
                result["data"]["role"] = "assistant"
        elif isinstance(messages[-1], dict):
            result["data"]["content"] = messages[-1].get("content", "")
            result["data"]["role"] = messages[-1].get("type", "assistant")

    # 提取选择等待信息（优先处理）
    pending_selection = state_update.get("pending_selection")
    if pending_selection:
        result["data"]["pending_selection"] = pending_selection
        # 当有pending_selection时，不在response_data中重复包含products
        # 因为products已经在pending_selection.options中
        result["data"]["response_type"] = "selection"
        if "products" in result["data"]["response_data"]:
            del result["data"]["response_data"]["products"]
            has_products = False

    # 提取确认等待信息
    if confirmation_pending := state_update.get("confirmation_pending"):
        result["data"]["confirmation_pending"] = confirmation_pending
        result["data"]["response_type"] = "confirmation"

    # 确定响应类型（仅在没有pending_selection和confirmation_pending时）
    if not pending_selection and not confirmation_pending:
        if has_products and has_orders:
            result["data"]["response_type"] = "mixed"
        elif has_products:
            result["data"]["response_type"] = "product_list"
        elif has_orders:
            result["data"]["response_type"] = "order_list"

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

