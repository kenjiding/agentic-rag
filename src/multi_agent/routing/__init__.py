"""智能路由模块 - LLM驱动的路由决策系统

本模块实现了基于LLM intent classification的智能路由引擎，
从"硬编码匹配"转向"LLM智能判断"。

核心组件：
- IntentParser: 直接读取LLM输出的业务意图字段（无需模式匹配）
- BusinessIntent: 业务意图对象

2025-2026 企业级最佳实践：
- 充分利用LLM的语义理解能力
- 规则仅用于高频确定性场景
- 复杂场景交给LLM兜底
- 无硬编码模式匹配
"""

from src.multi_agent.routing.intent_parser import IntentParser, BusinessIntent, IntentReasoningParser

__all__ = [
    "IntentParser",  # 新的类名（推荐使用）
    "IntentReasoningParser",  # 旧类名（向后兼容）
    "BusinessIntent",
]
