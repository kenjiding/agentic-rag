"""Multi-Agent 系统工具函数

提供共享的工具函数，避免代码重复。
"""

import logging
from typing import List, Any, TypeVar, Union
from typing import Mapping

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

logger = logging.getLogger(__name__)

T = TypeVar('T')


def get_state_value(state: Union[Mapping[str, Any], object], key: str, default: T = None) -> Union[Any, T]:
    """安全地获取状态字段值，支持字典和 Pydantic 模型

    LangGraph 的 state_snapshot.values 可能是字典或 Pydantic 实例
    这个函数统一处理两种情况，避免 AttributeError 和 KeyError

    Args:
        state: 状态对象（字典或 Pydantic 实例）
        key: 字段名
        default: 默认值（如果字段不存在）

    Returns:
        字段值或默认值

    Examples:
        >>> # Pydantic 实例
        >>> value = get_state_value(snapshot.values, "task_chain")
        >>> # 字典
        >>> value = get_state_value(snapshot.values, "task_chain")
    """
    # 如果是字典或 Mapping
    if isinstance(state, Mapping):
        return state.get(key, default)

    # 如果是 Pydantic 实例或其他对象
    try:
        return getattr(state, key, default)
    except (AttributeError, TypeError):
        return default


def state_to_dict(state: Union[Mapping[str, Any], object]) -> Mapping[str, Any]:
    """将状态转换为字典格式

    处理 Pydantic 实例和字典两种情况

    Args:
        state: 状态对象（字典或 Pydantic 实例）

    Returns:
        字典格式的状态
    """
    # 如果已经是字典或 Mapping
    if isinstance(state, Mapping):
        return state

    # 如果是 Pydantic 实例
    if hasattr(state, 'model_dump'):
        return state.model_dump()

    # 其他情况，尝试转换为字典
    try:
        return dict(state)
    except (TypeError, ValueError):
        logger.warning(f"无法将状态转换为字典: {type(state)}")
        return {}


def clean_messages_for_llm(messages: List[BaseMessage], keep_recent_n: int = 10) -> List[BaseMessage]:
    """清理消息序列，优化传递给 LLM 的上下文

    分三步处理：
    1. 只保留最近 N 条消息，控制上下文长度
    2. 移除孤立的 ToolMessage（没有对应 AIMessage 中 tool_calls 的）
    3. 移除孤立的 tool_calls（没有对应 ToolMessage 的 tool_calls）

    避免出现以下错误：
    - "tool_calls must be followed by tool messages"
    - "messages with role 'tool' must be a response to a preceeding message with 'tool_calls'"

    Args:
        messages: 原始消息列表
        keep_recent_n: 保留最近的 N 轮对话（默认10条消息）

    Returns:
        清理后的消息列表
    """
    # 步骤1：取最后 N 条消息
    recent_messages = list(messages[-keep_recent_n:]) if len(messages) > keep_recent_n else list(messages)

    # 步骤2：收集所有有效的 tool_call_id（来自 AIMessage 的 tool_calls）
    valid_tool_call_ids_from_ai = set()
    for msg in recent_messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("id"):
                    valid_tool_call_ids_from_ai.add(tc["id"])

    # 步骤3：收集所有 ToolMessage 的 tool_call_id
    tool_call_ids_from_tool_messages = set()
    for msg in recent_messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, "tool_call_id"):
            tool_call_ids_from_tool_messages.add(msg.tool_call_id)

    # 步骤4：清理孤立的消息
    cleaned_messages = []
    orphaned_tool_calls_count = 0
    orphaned_tool_messages_count = 0

    for msg in recent_messages:
        if isinstance(msg, ToolMessage):
            # 检查 ToolMessage 是否有对应的 AIMessage(tool_calls)
            if hasattr(msg, "tool_call_id") and msg.tool_call_id in valid_tool_call_ids_from_ai:
                cleaned_messages.append(msg)
            else:
                orphaned_tool_messages_count += 1
                logger.debug(f"移除孤立的 ToolMessage: tool_call_id={getattr(msg, 'tool_call_id', 'N/A')}")
        elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            # 检查每个 tool_call 是否有对应的 ToolMessage
            valid_calls = [
                tc for tc in msg.tool_calls
                if tc.get("id") in tool_call_ids_from_tool_messages
            ]
            if valid_calls:
                # 有有效的 tool_calls，保留消息
                if len(valid_calls) != len(msg.tool_calls):
                    orphaned_tool_calls_count += len(msg.tool_calls) - len(valid_calls)
                cleaned_messages.append(msg)
            else:
                # 所有 tool_calls 都是孤立的，移除 tool_calls 保留消息
                orphaned_tool_calls_count += len(msg.tool_calls)
                cleaned_messages.append(AIMessage(content=msg.content))
        else:
            # 其他消息类型，直接保留
            cleaned_messages.append(msg)

    if orphaned_tool_calls_count > 0:
        logger.warning(f"移除了 {orphaned_tool_calls_count} 个孤立的 tool_calls")
    if orphaned_tool_messages_count > 0:
        logger.warning(f"移除了 {orphaned_tool_messages_count} 个孤立的 ToolMessage")

    return cleaned_messages
