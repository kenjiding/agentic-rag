"""状态格式化和步骤格式化工具

核心原则（2025-2026重构）：
1. 只做简单的透传和序列化，不做业务数据转换
2. Agent设置的response_data直接透传到前端
3. 单一数据源：Agent是response_data的唯一提供者
4. 避免任何历史数据污染
"""
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# LangGraph interrupt node key constant
LANGGRAPH_INTERRUPT_KEY = "__interrupt__"


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
    """格式化执行步骤名称
    
    展示策略（简化版）：
    - 只展示：planner → policy_gate → plan_executor（Agent执行）
    - 其他节点（Agent节点、post_action_verifier等）不展示
    - 通过 step_display 透传（节点负责自己的展示逻辑）
    """
    # === 从 step_display 透传（节点设置的展示信息）===
    if isinstance(node_update, dict):
        step_display = node_update.get("step_display")
        if step_display is not None:
            # step_display 可能是 StepDisplay 对象或 dict
            if hasattr(step_display, "show"):
                # StepDisplay 对象
                if not step_display.show:
                    return None
                return step_display.name
            elif isinstance(step_display, dict):
                # dict 形式
                if not step_display.get("show", True):
                    return None
                return step_display.get("name")
    
    # 没有 step_display 的节点不展示
    return None


def format_step_detail(node_name: str, node_update: Any) -> str:
    """格式化执行步骤的详细描述
    
    展示策略（简化版）：
    - 只展示：planner → policy_gate → plan_executor（Agent执行）
    - 通过 step_display 透传（节点负责自己的展示逻辑）
    """
    # === 从 step_display 透传（节点设置的展示信息）===
    if isinstance(node_update, dict):
        step_display = node_update.get("step_display")
        if step_display is not None:
            # step_display 可能是 StepDisplay 对象或 dict
            if hasattr(step_display, "detail"):
                # StepDisplay 对象
                return step_display.detail or "正在处理..."
            elif isinstance(step_display, dict):
                # dict 形式
                return step_display.get("detail") or "正在处理..."
    
    return "正在处理..."
