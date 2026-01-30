"""Context rendering utilities for prompts."""
import json
from typing import Any, Dict, List, Optional


def render_context_bundle(context_bundle: Optional[Dict[str, Any]]) -> str:
    """Render structured context bundle in human-readable format with minimal token usage.
    
    Extracts and formats only the essential information from the context bundle,
    significantly reducing token consumption compared to raw JSON serialization.
    """
    if not context_bundle:
        return "上下文: 无"

    short_term = context_bundle.get("short_term_context", {})
    task_input = context_bundle.get("task_input", {})
    
    sections = []
    
    # 1. 当前查询
    current_query = short_term.get("current_query", task_input.get("query", ""))
    if current_query:
        sections.append(f"当前问题: {current_query}")
    
    # 2. 对话阶段
    phase = short_term.get("conversation_phase") or task_input.get("conversation_phase", "")
    if phase:
        phase_display = {
            "idle": "空闲",
            "greeting": "问候",
            "product_selecting": "商品选择中",
            "order_management": "订单管理中",
            "consultation": "咨询中",
        }.get(phase, phase)
        sections.append(f"对话阶段: {phase_display}")
    
    # 3. 关键实体（精炼）
    entities = short_term.get("key_entities", {})
    if entities:
        entity_parts = []
        # 单个商品ID（已锁定一方）
        if entities.get("product_id"):
            entity_parts.append(f"商品ID(单): {entities['product_id']}")
        # 多商品ID（对比/多商品场景）
        if entities.get("product_ids"):
            entity_parts.append(f"商品ID(多): {entities['product_ids']}")
        if entities.get("order_id"):
            entity_parts.append(f"订单ID: {entities['order_id']}")
        if entities.get("search_keyword"):
            entity_parts.append(f"搜索词: {entities['search_keyword']}")
        if entity_parts:
            sections.append("关键实体: " + " | ".join(entity_parts))
    
    # 4. 对话历史（精炼格式，最近3轮）
    history = short_term.get("conversation_history", [])
    if history:
        sections.append("\n对话历史:")
        # 只展示最近3轮，避免过多历史
        recent_history = history[-3:] if len(history) > 3 else history
        for idx, round_data in enumerate(recent_history, 1):
            human_msg = round_data.get("human", "")
            ai_msg = round_data.get("ai", "")
            tool_calls = round_data.get("tool_calls", [])
            
            # 用户问题
            sections.append(f"  [{idx}] 用户: {human_msg}")
            
            # AI回复（截断过长内容）
            if ai_msg:
                ai_display = ai_msg if len(ai_msg) <= 100 else ai_msg[:100] + "..."
                sections.append(f"      AI: {ai_display}")
            
            # 工具调用（简化格式）
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    # 只显示关键参数
                    key_args = _extract_key_args(tool_name, tool_args)
                    sections.append(f"      🔧 {tool_name}({key_args})")
    
    # 5. 最近工具调用结果（精炼）
    recent_tools = short_term.get("recent_tool_calls", [])
    if recent_tools:
        sections.append("\n工具调用结果:")
        # 只展示最近3个
        recent_tools_limited = recent_tools[-3:] if len(recent_tools) > 3 else recent_tools
        for idx, tool_result in enumerate(recent_tools_limited, 1):
            tool_name = tool_result.get("name", "unknown")
            summary = tool_result.get("summary", "")
            # 提取关键信息
            result_display = _format_tool_result(tool_name, summary)
            sections.append(f"  [{idx}] {tool_name}: {result_display}")
    
    return "\n".join(sections)


def _extract_key_args(tool_name: str, args: Dict[str, Any]) -> str:
    """Extract key arguments by directly outputting key-value pairs.
    
    Trusts field names as semantic: simply outputs all non-empty key-value pairs.
    No intelligent inference needed - field names already carry semantic meaning.
    """
    if not args:
        return "..."
    
    # Filter out empty/null values
    non_empty = {k: v for k, v in args.items() if v not in (None, "", [], {})}
    if not non_empty:
        return "..."
    
    # Format key-value pairs directly (trust field names as semantic)
    parts = []
    max_parts = 3
    
    for key, value in non_empty.items():
        if len(parts) >= max_parts:
            break
        
        # Format based on value type, but keep it simple
        if isinstance(value, list):
            parts.append(f"{key}={len(value)}项")
        elif isinstance(value, bool):
            if value:
                parts.append(key)  # Just show key name for True flags
            # Skip False values
        elif isinstance(value, str):
            # Truncate long strings
            str_value = value if len(value) <= 20 else value[:20] + "..."
            parts.append(f"{key}={str_value}")
        else:
            parts.append(f"{key}={value}")
    
    if parts:
        return ", ".join(parts)
    
    return "..."


def _format_tool_result(tool_name: str, summary: str) -> str:
    """Format tool result by automatically analyzing data structure.
    
    Completely data-driven: automatically extracts meaningful information
    based on data types and patterns, without enumerating field names.
    """
    if not summary:
        return "无结果"
    
    try:
        data = json.loads(summary)
        return _format_data_structure(data)
    
    except (json.JSONDecodeError, TypeError, AttributeError):
        # JSON解析失败或非JSON数据，直接截断
        summary_str = str(summary)
        return summary_str if len(summary_str) <= 80 else summary_str[:80] + "..."


def _format_data_structure(data: Any, max_depth: int = 2) -> str:
    """Recursively format any data structure by analyzing its type and content.
    
    Automatically extracts meaningful fields based on:
    - Data types (list, dict, primitives)
    - Field name patterns (contains "id", "name", "status", etc.)
    - Value types and significance
    """
    if data is None:
        return "无"
    
    # Pattern 1: List/Array - format items
    if isinstance(data, list):
        if not data:
            return "空列表"
        
        # Format first 3 items
        formatted_items = []
        for item in data[:3]:
            if isinstance(item, dict):
                # Extract key info from dict item
                item_str = _extract_key_info_from_dict(item)
                formatted_items.append(item_str)
            else:
                formatted_items.append(str(item))
        
        result = "; ".join(formatted_items)
        if len(data) > 3:
            result += f" (共{len(data)}项)"
        return result
    
    # Pattern 2: Dictionary/Object - extract meaningful fields
    if isinstance(data, dict):
        # Find arrays in dict (most important pattern)
        array_fields = {k: v for k, v in data.items() if isinstance(v, list) and v}
        if array_fields:
            # Format first array found
            first_key, first_array = next(iter(array_fields.items()))
            array_str = _format_data_structure(first_array, max_depth - 1)
            return f"{first_key}: {array_str}"
        
        # Find nested dicts with arrays
        for key, value in data.items():
            if isinstance(value, dict):
                nested_arrays = {k: v for k, v in value.items() if isinstance(v, list) and v}
                if nested_arrays:
                    nested_key, nested_array = next(iter(nested_arrays.items()))
                    array_str = _format_data_structure(nested_array, max_depth - 1)
                    return f"{key}.{nested_key}: {array_str}"
        
        # Extract key information from dict
        return _extract_key_info_from_dict(data)
    
    # Pattern 3: Primitive values
    if isinstance(data, (str, int, float, bool)):
        return str(data)
    
    # Fallback
    data_str = json.dumps(data, ensure_ascii=False)
    return data_str if len(data_str) <= 80 else data_str[:80] + "..."


def _extract_key_info_from_dict(obj: Dict[str, Any]) -> str:
    """Extract key information by directly outputting key-value pairs.
    
    Trusts field names as semantic: simply outputs all non-empty key-value pairs.
    No intelligent inference needed - field names already carry semantic meaning.
    """
    if not obj:
        return "{}"
    
    # Filter non-empty values
    non_empty = {k: v for k, v in obj.items() if v not in (None, "", [], {})}
    if not non_empty:
        return "{}"
    
    # Simply output key-value pairs (trust field names as semantic)
    parts = []
    max_parts = 4  # Show more fields for result objects
    
    for key, value in non_empty.items():
        if len(parts) >= max_parts:
            break
        
        # Format based on value type, keep it simple
        if isinstance(value, list):
            parts.append(f"{key}={len(value)}项")
        elif isinstance(value, dict):
            # For nested dicts, show key and indicate it's an object
            parts.append(f"{key}={{...}}")
        elif isinstance(value, str):
            # Truncate long strings
            str_value = value if len(value) <= 30 else value[:30] + "..."
            parts.append(f"{key}={str_value}")
        elif isinstance(value, bool):
            if value:
                parts.append(key)  # Just show key name for True
            # Skip False
        else:
            parts.append(f"{key}={value}")
    
    if parts:
        return ", ".join(parts)
    
    # Fallback: show first few key-value pairs
    first_few = list(non_empty.items())[:3]
    return ", ".join(f"{k}={v}" for k, v in first_few)
