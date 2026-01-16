"""LLM-Driven Routing Engine - 基于业务意图的智能路由

核心设计原则：
1. LLM优先：直接使用LLM输出的business_intent_type字段进行路由决策
2. 信任LLM：如果LLM已经识别出明确的业务意图，直接使用，无需通过实体硬编码判断
3. 实体补充：实体（product_id、order_id等）仅作为上下文验证，不用于主要路由判断
4. 可解释性：每个路由决策都有明确的reason和method

架构改进：
- 移除基于product_ids数量、order_id存在等硬编码规则
- 优先使用business_intent.intent_type进行路由映射
- 实体仅用于增强决策信心或提供上下文信息
"""
import logging
from typing import Any, Dict, Optional, Tuple

from src.multi_agent.state import MultiAgentState
from src.multi_agent.constants import ActionName, AgentName
from src.multi_agent.routing.intent_parser import IntentReasoningParser

logger = logging.getLogger(__name__)


class RoutingEngine:
    """基于业务意图的智能路由引擎

    设计理念：
    - LLM优先：直接使用LLM输出的business_intent_type字段
    - 信任LLM：如果LLM已识别意图，直接路由，无需实体硬编码验证
    - 实体补充：实体仅用于增强决策信心，不用于主要判断
    - 可解释性：每个路由决策都有明确的reason和method

    决策流程：
    1. 读取LLM输出的business_intent_type和suggested_action
    2. 基于intent_type直接映射到对应的action和agent
    3. 实体信息仅用于增强决策信心或提供上下文
    4. 如果LLM未识别出明确意图，返回None交给supervisor的LLM路由处理
    """

    # 业务意图类型到动作和Agent的映射（与IntentParser中的映射保持一致）
    INTENT_TO_ROUTING: Dict[str, Tuple[ActionName, AgentName]] = {
        "social_chat": (ActionName.CHAT, AgentName.CHAT_AGENT),
        "general_chat": (ActionName.CHAT, AgentName.CHAT_AGENT),
        "product_search": (ActionName.PRODUCT_SEARCH, AgentName.PRODUCT_AGENT),
        "product_comparison": (ActionName.CONSULTATION, AgentName.CONSULTATION_AGENT),
        "order_management": (ActionName.ORDER_MANAGEMENT, AgentName.ORDER_AGENT),
    }

    def __init__(self):
        """初始化路由引擎"""
        self.intent_parser = IntentReasoningParser()

    def route(self, state: MultiAgentState) -> Optional[Dict[str, Any]]:
        """
        智能路由决策：基于LLM输出的业务意图类型进行路由

        Args:
            state: 多Agent系统状态

        Returns:
            路由决策字典，如果无规则匹配返回None（交给LLM兜底）
        """
        query_intent = state.query_intent or {}
        entities = state.entities or {}

        # Step 1: 直接读取LLM输出的业务意图类型
        business_intent = self.intent_parser.parse_business_intent(query_intent)

        # Step 2: 优先基于business_intent_type进行路由（核心逻辑）
        # 如果LLM已经识别出明确的业务意图，直接使用，无需通过实体硬编码判断
        if business_intent.intent_type in self.INTENT_TO_ROUTING:
            action, agent = self.INTENT_TO_ROUTING[business_intent.intent_type]
            
            # 构建路由原因（包含实体信息作为上下文，但不用于判断）
            routing_reason = self._build_routing_reason(
                business_intent.intent_type,
                business_intent.suggested_action,
                entities
            )

            decision = {
                "next_action": action,
                "selected_agent": agent,
                "routing_reason": routing_reason,
                "confidence": business_intent.confidence,
                "routing_method": "llm_intent_classification",
                "business_intent": business_intent.intent_type,
            }
            
            logger.info(
                f"[RoutingEngine] Route: {action}→{agent} | "
                f"Method: llm_intent_classification | "
                f"Intent: {business_intent.intent_type} | "
                f"Confidence: {business_intent.confidence:.2f}"
            )
            return decision

        # Step 3: 如果LLM未识别出明确意图，返回None交给supervisor的LLM路由处理
        logger.info(
            f"[RoutingEngine] LLM未识别出明确业务意图（intent_type={business_intent.intent_type}），"
            f"交给supervisor的LLM路由处理"
        )
        return None

    def _build_routing_reason(
        self,
        intent_type: str,
        suggested_action: Optional[str],
        entities: Dict[str, Any]
    ) -> str:
        """
        构建路由原因说明，包含实体信息作为上下文

        Args:
            intent_type: 业务意图类型
            suggested_action: LLM建议的动作
            entities: 提取的实体信息

        Returns:
            路由原因说明字符串
        """
        reason_parts = [f"LLM识别为{intent_type}意图"]
        
        # 添加LLM建议的动作（如果存在且与映射一致）
        if suggested_action:
            reason_parts.append(f"（建议动作：{suggested_action}）")
        
        # 添加实体信息作为上下文（不用于判断，仅提供信息）
        entity_context = []
        if entities.get("product_id"):
            entity_context.append(f"product_id={entities['product_id']}")
        if entities.get("product_ids"):
            entity_context.append(f"product_ids={entities['product_ids']}")
        if entities.get("order_id"):
            entity_context.append(f"order_id={entities['order_id']}")
        if entities.get("search_keyword"):
            entity_context.append(f"search_keyword={entities['search_keyword']}")
        
        if entity_context:
            reason_parts.append(f"（实体上下文：{', '.join(entity_context)}）")
        
        return " | ".join(reason_parts)
