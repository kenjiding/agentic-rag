"""Intent Reasoning Parser - 解析LLM生成的reasoning字段

核心创新：充分利用LLM已生成的reasoning信息，提取结构化业务意图。

从"规则驱动"转向"LLM智能驱动"的关键组件。
"""
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BusinessIntent(BaseModel):
    """解析后的业务意图

    从QueryIntent.reasoning字段中提取的结构化业务意图。
    """
    intent_type: str = Field(
        ...,
        description="业务意图类型：social_chat, order_management, product_comparison, product_search, general_chat等"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="意图识别置信度（0-1）"
    )
    signals: List[str] = Field(
        default_factory=list,
        description="触发该意图的关键信号（如['gratitude', 'greeting']）"
    )
    raw_reasoning: str = Field(
        ...,
        description="原始reasoning文本"
    )

    def __str__(self) -> str:
        return f"BusinessIntent(type={self.intent_type}, confidence={self.confidence:.2f}, signals={self.signals})"


class IntentReasoningParser:
    """意图推理解析器 - 解析LLM生成的reasoning字段

    核心理念：
    - LLM已经在reasoning中分析了用户意图（如"用户表达了感谢"）
    - 我们不需要重新分析，只需要解析这些信息
    - 充分利用LLM的智能判断能力

    优先级（从高到低）：
    1. 社交表达（感谢/问候）- 最高优先级，避免误路由
    2. 订单业务意图（LLM已明确标注）
    3. 产品对比
    4. 产品搜索
    5. 默认为通用对话
    """

    # 模式1: 社交表达（最高优先级）
    SOCIAL_PATTERNS = [
        "用户表达了感谢", "表示感谢", "感谢语", "问候",
        "致谢", "礼貌用语", "社交表达", "闲聊",
        "一般性对话", "没有具体业务需求", "没有提出具体的查询需求",
        "纯粹感谢", "礼貌性感谢", "感谢帮助", "致谢语"
    ]

    # 模式2: 订单业务意图（LLM已明确标注）
    ORDER_INTENT_PATTERNS = [
        "订单查询意图", "订单取消意图", "订单管理意图", "订单相关操作",
        "查询订单", "取消订单", "订单咨询"
    ]

    # 模式3: 产品对比
    COMPARISON_PATTERNS = [
        "对比", "比较", "多个产品", "产品对比", "型号对比",
        "产品比较", "差异", "区别", "哪个好"
    ]

    # 模式4: 产品搜索
    PRODUCT_SEARCH_PATTERNS = [
        "购买需求", "搜索产品", "产品查询", "查找商品", "产品推荐",
        "想买", "想要", "购买", "选购"
    ]

    @classmethod
    def parse_business_intent(cls, query_intent: Dict[str, Any]) -> BusinessIntent:
        """
        从reasoning中提取业务意图

        Args:
            query_intent: 意图识别结果（QueryIntent转字典）

        Returns:
            BusinessIntent: 解析后的业务意图
        """
        reasoning = (query_intent.get("reasoning") or "").lower()
        intent_type = query_intent.get("intent_type", "")

        # Pattern 1: 社交表达/感谢（最高优先级）
        # 这是最关键的修复：避免"谢谢"被误路由到product_agent
        for pattern in cls.SOCIAL_PATTERNS:
            if pattern.lower() in reasoning:
                logger.info(f"[IntentParser] 检测到社交表达: '{pattern}'")
                return BusinessIntent(
                    intent_type="social_chat",
                    confidence=0.95,
                    signals=["gratitude_or_greeting"],
                    raw_reasoning=reasoning
                )

        # Pattern 2: 订单业务意图（LLM已明确标注）
        for pattern in cls.ORDER_INTENT_PATTERNS:
            if pattern.lower() in reasoning:
                logger.info(f"[IntentParser] 检测到订单业务意图: '{pattern}'")
                return BusinessIntent(
                    intent_type="order_management",
                    confidence=0.9,
                    signals=["explicit_order_intent"],
                    raw_reasoning=reasoning
                )

        # Pattern 3: 产品对比
        if intent_type == "comparison" or any(p in reasoning for p in cls.COMPARISON_PATTERNS):
            logger.info(f"[IntentParser] 检测到产品对比意图")
            return BusinessIntent(
                intent_type="product_comparison",
                confidence=0.85,
                signals=["comparison_keywords"],
                raw_reasoning=reasoning
            )

        # Pattern 4: 产品搜索（仅当有明确搜索信号）
        if any(p in reasoning for p in cls.PRODUCT_SEARCH_PATTERNS):
            logger.info(f"[IntentParser] 检测到产品搜索意图")
            return BusinessIntent(
                intent_type="product_search",
                confidence=0.8,
                signals=["product_search_signals"],
                raw_reasoning=reasoning
            )

        # Pattern 5: 默认为通用对话
        logger.info(f"[IntentParser] 未检测到明确业务意图，使用默认: general_chat")
        return BusinessIntent(
            intent_type="general_chat",
            confidence=0.6,
            signals=["no_specific_intent"],
            raw_reasoning=reasoning
        )

    @classmethod
    def is_social_chat(cls, query_intent: Dict[str, Any]) -> bool:
        """
        快速判断是否为社交对话（感谢/问候/闲聊）

        Args:
            query_intent: 意图识别结果

        Returns:
            bool: 是否为社交对话
        """
        business_intent = cls.parse_business_intent(query_intent)
        return business_intent.intent_type == "social_chat"

    @classmethod
    def is_order_management(cls, query_intent: Dict[str, Any]) -> bool:
        """
        快速判断是否为订单管理意图

        Args:
            query_intent: 意图识别结果

        Returns:
            bool: 是否为订单管理
        """
        business_intent = cls.parse_business_intent(query_intent)
        return business_intent.intent_type == "order_management"
