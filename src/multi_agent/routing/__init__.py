"""智能路由模块 - LLM驱动的路由决策系统

本模块实现了基于LLM reasoning的智能路由引擎，从"规则驱动"转向"LLM智能驱动"。

核心组件：
- IntentReasoningParser: 解析LLM生成的reasoning字段，提取业务意图
- ContextAwareEntityCleaner: 对话阶段感知的实体清理器
- RoutingDecisionLogger: 路由决策日志格式化工具

2025-2026 企业级最佳实践：
- 充分利用LLM的语义理解能力
- 规则仅用于高频确定性场景
- 复杂场景交给LLM兜底
"""

from src.multi_agent.routing.intent_parser import IntentReasoningParser, BusinessIntent
from src.multi_agent.routing.entity_cleaner import ContextAwareEntityCleaner
from src.multi_agent.routing.decision_logger import RoutingDecisionLogger

__all__ = [
    "IntentReasoningParser",
    "BusinessIntent",
    "ContextAwareEntityCleaner",
    "RoutingDecisionLogger",
]
