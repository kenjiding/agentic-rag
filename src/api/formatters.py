"""状态格式化和步骤格式化工具

核心原则（2025-2026重构）：
1. 只做简单的透传和序列化，不做业务数据转换
2. Agent设置的response_data直接透传到前端
3. 单一数据源：Agent是response_data的唯一提供者
4. 避免任何历史数据污染
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# LangGraph interrupt node key constant
LANGGRAPH_INTERRUPT_KEY = "__interrupt__"


def format_state_update(state_update: Dict[str, Any], node_update: Any = None, messages_count_before_update: int = 0) -> Dict[str, Any]:
    """格式化状态更新为前端友好的格式

    核心原则（2025-2026终极重构）：
    - 只做SSE事件封装，零业务逻辑
    - Agent返回的node_update就是完整的前端数据
    - 直接透传，不做任何处理和转换
    - 单一数据源：ResponseModel在Agent中构建

    Args:
        state_update: 完整的累积状态（包含历史消息）
        node_update: 当前节点的更新（只包含当前轮次的变化）
        messages_count_before_update: 更新前 state_update 中的消息数量（已废弃）

    Returns:
        SSE事件封装的数据
    """
    return {
        "type": "state_update",
        "data": node_update if isinstance(node_update, dict) else {}
    }


def format_step_name(node_name: str, node_update: Any) -> Optional[str]:
    """格式化执行步骤名称（一步一步智能模式）"""
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


def format_step_detail(node_name: str, node_update: Any) -> str:
    """格式化执行步骤的详细描述（一步一步智能模式）"""
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
