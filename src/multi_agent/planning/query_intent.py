"""Unified query intent models for the multi-agent system.

This replaces the legacy src/intent module (which is being removed).

Design goals:
- Pydantic models compatible with LangChain structured output
- Business-oriented fields for routing
- Retrieval-oriented fields for RAG decomposition
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# Retrieval pipeline options (kept minimal and explicit)
PipelineOption = Literal["semantic", "keyword", "hybrid", "rerank"]

# Decomposition types
DecompositionType = Literal["comparison", "multi_hop", "information_needs", "dimensional", "other"]

# High-level intent type (general)
IntentType = Literal[
    "factual",
    "comparison",
    "analytical",
    "procedural",
    "causal",
    "temporal",
    "multi_hop",
    "other",
]

ComplexityLevel = Literal["simple", "moderate", "complex"]

# Order-management sub-intents
OrderIntent = Literal["query", "cancel", "create", "other"]


class Entities(BaseModel):
    general_entities: List[str] = Field(default_factory=list)
    time_points: List[str] = Field(default_factory=list)
    quantity: Optional[int] = Field(default=None, ge=1)
    search_keyword: Optional[str] = Field(default=None)
    product_id: Optional[int] = Field(default=None, ge=1)
    # For comparison / multi-product tasks.
    # Keep it optional and permissive: planner may fill it when IDs are known from context,
    # and downstream product_agent can also populate it deterministically from tool results.
    product_ids: List[int] = Field(default_factory=list)
    order_id: Optional[str] = Field(default=None)


class SubQuery(BaseModel):
    query: str
    purpose: str = ""
    recommended_strategy: List[PipelineOption] = Field(default_factory=lambda: ["semantic"])
    recommended_k: int = 5
    order: int = 0
    depends_on: List[int] = Field(default_factory=list)


class QueryIntent(BaseModel):
    # Core intent
    intent_type: IntentType = Field(
        ...,
        description=(
            "通用意图类型（用于问题形态/推理类型，不用于业务路由）。"
            "必须是以下之一：factual/comparison/analytical/procedural/causal/temporal/multi_hop/other。"
            "注意：不要填写 business_intent_type（例如 order_management/product_search）。"
        ),
    )
    complexity: ComplexityLevel = Field(
        ...,
        description="复杂度：simple/moderate/complex（与业务路由无关）",
    )

    # Decomposition
    needs_decomposition: bool
    decomposition_type: Optional[DecompositionType] = None
    decomposition_reason: str = ""
    sub_queries: List[SubQuery] = Field(default_factory=list)

    # Entities (slot filling)
    entities: Entities = Field(default_factory=Entities)

    # Retrieval strategy
    recommended_retrieval_strategy: List[PipelineOption] = Field(default_factory=list)
    recommended_k: int = 5
    needs_multi_round_retrieval: bool = False

    # Meta
    confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Planner confidence score in [0,1]. Defaults to 0.6 if omitted by LLM.",
    )
    reasoning: str = ""

    # Business routing (for multi-agent)
    business_intent_type: Literal[
        "social_chat",
        "order_management",
        "product_comparison",
        "product_search",
        "general_chat",
    ] = Field(
        ...,
        description=(
            "业务意图类型（用于多Agent路由）。"
            "可选：social_chat/order_management/product_comparison/product_search/general_chat。"
        ),
    )

    # Business sub-intents (explicit, to avoid reasoning-string matching)
    order_intent: Optional[OrderIntent] = Field(
        default=None,
        description=(
            "订单管理子意图，仅当 business_intent_type=order_management 时使用："
            "query(查询订单)/cancel(取消订单)/create(创建订单)/other(其他订单相关)。"
        ),
    )

    # Extensible bucket for future planner signals
    extra: Dict[str, Any] = Field(default_factory=dict)

