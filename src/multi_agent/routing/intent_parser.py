"""Intent Parser - 直接读取LLM输出的业务意图字段

核心理念：让LLM在意图识别阶段直接输出结构化的业务意图类型，
而不是后续用硬编码模式匹配reasoning字段。

这是真正智能的方案：充分利用LLM的语义理解能力，避免硬编码规则。
"""
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BusinessIntent(BaseModel):
    """业务意图（直接来自LLM输出）

    从QueryIntent.business_intent_type字段直接读取，无需模式匹配。
    """
    intent_type: str = Field(
        ...,
        description="业务意图类型：social_chat, order_management, product_comparison, product_search, general_chat"
    )
    suggested_action: str = Field(
        default="chat",
        description="LLM建议的next_action"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="意图识别置信度（0-1）"
    )

    def __str__(self) -> str:
        return (
            f"BusinessIntent(type={self.intent_type}, "
            f"conf={self.confidence:.2f})"
        )


class IntentParser:
    """意图解析器 - 直接读取LLM输出，无需硬编码匹配

    核心理念：
    - LLM在意图识别阶段已经输出了business_intent_type
    - 我们只需要直接读取，不需要任何模式匹配
    - 这是真正智能的、可扩展的方案

    与之前的方案对比：
    - 旧方案：用硬编码模式列表匹配reasoning字符串（容易出错）
    - 新方案：直接读取LLM输出的结构化字段（智能、准确、可扩展）
    """

    @classmethod
    def parse_business_intent(cls, query_intent: Dict[str, Any]) -> BusinessIntent:
        """
        从query_intent中读取业务意图（直接读取，无需模式匹配）

        Args:
            query_intent: 意图识别结果（QueryIntent转字典）

        Returns:
            BusinessIntent: 业务意图对象
        """
        # 直接从LLM输出读取，但必须做类型/空值归一化，避免Pydantic校验错误导致路由降级
        raw_type = query_intent.get("business_intent_type", "general_chat")
        business_intent_type = raw_type if isinstance(raw_type, str) and raw_type.strip() else "general_chat"

        raw_conf = query_intent.get("confidence", 0.6)
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))

        raw_action = query_intent.get("suggested_next_action")
        suggested_action = raw_action if isinstance(raw_action, str) and raw_action.strip() else None

        # 如果LLM没有显式给出 suggested_next_action（或给了None），按intent_type推导一个稳定的默认值
        if suggested_action is None:
            suggested_action = {
                "social_chat": "chat",
                "general_chat": "chat",
                "product_search": "product_search",
                "product_comparison": "consultation",
                "order_management": "order_management",
            }.get(business_intent_type, "chat")

        logger.info(
            f"[IntentParser] 读取业务意图: "
            f"type={business_intent_type}, "
            f"confidence={confidence:.2f}"
        )

        return BusinessIntent(
            intent_type=business_intent_type,
            suggested_action=suggested_action,
            confidence=confidence
        )

    @classmethod
    def is_social_chat(cls, query_intent: Dict[str, Any]) -> bool:
        """
        快速判断是否为社交对话

        Args:
            query_intent: 意图识别结果

        Returns:
            bool: 是否为社交对话
        """
        # 直接判断business_intent_type，无需额外字段
        return query_intent.get("business_intent_type") == "social_chat"


# 保留旧的类名作为别名，保持向后兼容
IntentReasoningParser = IntentParser
