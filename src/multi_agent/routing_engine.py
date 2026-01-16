"""Code-first routing engine with deterministic rules."""
import json
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, ToolMessage

from src.multi_agent.state import MultiAgentState
from src.multi_agent.constants import ActionName, AgentName


class RoutingEngine:
    """Deterministic routing rules with explicit priorities."""

    def route(self, state: MultiAgentState) -> Optional[Dict[str, Any]]:
        """Return routing decision or None if no rule matched."""
        entities = state.entities or {}
        query_intent = state.query_intent or {}
        conversation_phase = state.conversation_phase

        product_id = entities.get("product_id")
        product_ids = entities.get("product_ids")
        order_id = entities.get("order_id")
        search_keyword = entities.get("search_keyword")

        intent_type = query_intent.get("intent_type")
        reasoning = (query_intent.get("reasoning") or "").lower()

        # Rule 1: order management if order_id or explicit order intent
        if order_id or "订单查询意图" in reasoning or "订单取消意图" in reasoning:
            return self._decision(
                action=ActionName.ORDER_MANAGEMENT,
                agent=AgentName.ORDER_AGENT,
                reason="检测到订单相关实体/意图",
                confidence=0.95,
                method="rule_based",
            )

        # Rule 1b: order intent inferred from user text (without order_id)
        if self._looks_like_order_request(state):
            return self._decision(
                action=ActionName.ORDER_MANAGEMENT,
                agent=AgentName.ORDER_AGENT,
                reason="从用户表达推断订单管理需求",
                confidence=0.75,
                method="rule_based",
            )

        # Rule 2: product comparison if multiple product_ids
        if isinstance(product_ids, list) and len(product_ids) >= 2:
            return self._decision(
                action=ActionName.CONSULTATION,
                agent=AgentName.CONSULTATION_AGENT,
                reason="检测到多个产品ID，进入对比/咨询",
                confidence=0.9,
                method="rule_based",
            )

        # Rule 3: order creation when product_id is selected
        if product_id:
            return self._decision(
                action=ActionName.ORDER_MANAGEMENT,
                agent=AgentName.ORDER_AGENT,
                reason="已选定产品，进入订单流程",
                confidence=0.9,
                method="rule_based",
            )

        # Rule 4: product search when purchase intent or search keyword exists
        if search_keyword or intent_type == "comparison":
            return self._decision(
                action=ActionName.PRODUCT_SEARCH,
                agent=AgentName.PRODUCT_AGENT,
                reason="检测到商品搜索/对比意图但未选定产品",
                confidence=0.8,
                method="rule_based",
            )

        # Rule 5: phase-based routing hints
        if conversation_phase == "product_selecting":
            return self._decision(
                action=ActionName.PRODUCT_SEARCH,
                agent=AgentName.PRODUCT_AGENT,
                reason="对话阶段为产品选择，继续商品搜索",
                confidence=0.7,
                method="rule_based",
            )

        return None

    def _looks_like_order_request(self, state: MultiAgentState) -> bool:
        """Heuristic: order-related request without explicit order_id."""
        last_user = self._get_last_user_text(state)
        if not last_user:
            return False
        if "订单" in last_user:
            return True
        order_keywords = ["退款", "退货", "物流", "发货", "售后", "改地址", "改收货"]
        return any(keyword in last_user for keyword in order_keywords)

    @staticmethod
    def _get_last_user_text(state: MultiAgentState) -> str:
        for msg in reversed(state.messages or []):
            if isinstance(msg, HumanMessage):
                return msg.content or ""
        return ""

    @staticmethod
    def _decision(action: ActionName, agent: AgentName, reason: str, confidence: float, method: str) -> Dict[str, Any]:
        return {
            "next_action": action,
            "selected_agent": agent,
            "routing_reason": reason,
            "confidence": confidence,
            "routing_method": method,
        }
