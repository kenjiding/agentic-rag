"""流式响应通用工具函数

提供统一的状态累积和格式化逻辑，避免代码重复
"""
import logging
from typing import Dict, Any, AsyncIterator
from src.api.formatters import format_state_update

logger = logging.getLogger(__name__)


async def accumulate_and_format_state_updates(
    state_updates: AsyncIterator[Dict[str, Any]],
    filter_empty: bool = True
) -> AsyncIterator[Dict[str, Any]]:
    """累积状态更新并格式化为前端友好的格式
    
    这是流式响应的通用处理逻辑，统一处理状态累积和格式化
    
    Args:
        state_updates: LangGraph 返回的状态更新迭代器
        filter_empty: 是否过滤空的 state_update
        
    Yields:
        格式化后的状态更新字典
    """
    accumulated_state = {}
    messages_before_update = 0
    
    async for state_update in state_updates:
        # 处理每个节点的更新
        for node_name, node_data in state_update.items():
            if node_name in ("__start__", "__end__"):
                continue
                
            if not isinstance(node_data, dict):
                continue
                
            try:
                # 记录更新前的消息数量
                messages_before_update = len(accumulated_state.get("messages", []))
                
                # 累积状态（合并消息去重）
                if "messages" in node_data and "messages" in accumulated_state:
                    existing_messages = accumulated_state.get("messages", [])
                    new_messages = node_data.get("messages", [])
                    existing_ids = {id(msg) if hasattr(msg, 'id') else str(msg) for msg in existing_messages}
                    
                    for msg in new_messages:
                        msg_id = id(msg) if hasattr(msg, 'id') else str(msg)
                        if msg_id not in existing_ids:
                            existing_messages.append(msg)
                            existing_ids.add(msg_id)
                    
                    accumulated_state["messages"] = existing_messages
                    
                    # 合并其他字段
                    for key, value in node_data.items():
                        if key != "messages":
                            if key == "tools_used" and value:
                                existing_tools = accumulated_state.get("tools_used", [])
                                accumulated_state["tools_used"] = existing_tools + value
                            else:
                                accumulated_state[key] = value
                else:
                    accumulated_state.update(node_data)
                
                # 格式化状态更新
                formatted = format_state_update(accumulated_state, node_data, messages_before_update)
                
                if isinstance(formatted, dict):
                    # 过滤空的 state_update（可选）
                    if filter_empty:
                        data = formatted.get("data", {})
                        has_content = bool(data.get("content"))
                        has_response_data = bool(data.get("response_data", {}))
                        has_special_state = bool(
                            data.get("pending_selection") or 
                            data.get("confirmation_pending")
                        )
                        
                        if has_content or has_response_data or has_special_state:
                            yield formatted
                        else:
                            logger.debug(f"跳过空的 state_update (node={node_name})")
                    else:
                        yield formatted
                        
            except Exception as e:
                logger.warning(f"格式化状态更新失败 (node={node_name}): {e}", exc_info=True)

