"""Query intent data models.

General-purpose models for representing query intent and decomposition.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.intent.models.types import PipelineOption, DecompositionType, IntentType, ComplexityLevel


class Entities(BaseModel):
    """实体信息模型 - 统一存放所有实体
    
    用于从用户消息中提取结构化的实体信息。
    使用 Pydantic 模型确保类型安全和 OpenAI structured output 兼容性。
    """
    general_entities: List[str] = Field(
        default_factory=list,
        description="通用实体（人名、地名、时间、组织等）"
    )
    time_points: List[str] = Field(
        default_factory=list,
        description="时间点（年份、日期等）"
    )
    user_phone: Optional[str] = Field(
        default=None,
        description="用户手机号（11位，1开头）",
        pattern=r"^1[3-9]\d{9}$"
    )
    quantity: Optional[int] = Field(
        default=None,
        description="购买数量，如果查询中包含数量信息则提取（如：买2件、要3个）",
        ge=1
    )
    search_keyword: Optional[str] = Field(
        default=None,
        description="搜索关键词（品牌名、产品名或型号），用于商品模糊搜索。注意：只提取核心关键词，不要包含'产品'、'商品'、'东西'等通用词汇"
    )


class SubQuery(BaseModel):
    """Sub-query structure for decomposed queries.

    Represents a single query in a decomposed query strategy.
    Each sub-query can have its own retrieval strategy and execution order.
    """

    query: str = Field(description="子查询文本")

    purpose: str = Field(
        default="",
        description="该子查询的目的说明（如：获取基础事实、验证假设、补充细节等）"
    )

    recommended_strategy: List[PipelineOption] = Field(
        default_factory=lambda: ["semantic"],
        description="该子查询的推荐检索策略（根据子查询的特点独立选择）"
    )

    recommended_k: int = Field(
        default=5,
        description="该子查询的推荐检索数量"
    )

    order: int = Field(
        default=0,
        description="执行顺序：0表示可并��执行，>0表示需按顺序执行（数字越小越先执行）"
    )

    depends_on: List[int] = Field(
        default_factory=list,
        description="依赖的子查询索引列表（用于多跳查询，表示需要先执行哪些查询）"
    )


class QueryIntent(BaseModel):
    """Query intent structure (Unified Information Extraction Framework).

    Supports universal query decomposition mechanism that can automatically
    determine whether to decompose a query based on its complexity and characteristics.
    """

    # Main intent type
    intent_type: IntentType = Field(description="查询的主要意图类型")

    # Query complexity
    complexity: ComplexityLevel = Field(
        description="查询复杂度：simple(单一信息点), moderate(2-3个信息点), complex(多信息点或需要多步推理)"
    )

    # ==================== Universal Query Decomposition Mechanism ====================

    # Whether query decomposition is needed (core decision field)
    needs_decomposition: bool = Field(
        description="""是否需要将原查询分解为多个子查询。

**需要分解的信号**（满足任一条件即可）：
1. 复杂度为 complex
2. 包含多个独立的信息需求点（如"介绍X的原理、应用和前景"）
3. 需要对比多个对象/时间点（comparison）
4. 需要多步推理（multi_hop）
5. 需要从多个维度分析（analytical）
6. 需要按时间段查询（temporal跨度大）
7. 涉及因果链条（causal有多个层次）

**不需要分解的信号**：
1. 简单的单一事实查询
2. 查询已经足够具体明确
3. procedural类型查询（步骤是内容本身，不是检索单位）
4. 分解会导致上下文信息丢失
"""
    )

    # Decomposition type (when needs_decomposition=True)
    decomposition_type: Optional[DecompositionType] = Field(
        default=None,
        description="分解类型，当 needs_decomposition=True 时必须指定"
    )

    # Reason for decomposition
    decomposition_reason: Optional[str] = Field(
        default="",
        description="为什么需要分解的简要说明（如果 needs_decomposition=True）"
    )

    # Universal sub-query list
    sub_queries: List[SubQuery] = Field(
        default_factory=list,
        description="""分解后的子查询列表。每个子查询包含：
- query: 子查询文本
- purpose: 该子查询的目的
- recommended_strategy: 推荐的检索策略
- recommended_k: 推荐的检索数量
- order: 执行顺序（0=可并行，>0=按顺序）
- depends_on: 依赖的子查询索引
"""
    )

    # ==================== Slot Filling Information ====================

    # Extracted key information (slots) - 统一实体模型
    # 包含通用实体和业务实体，统一存放在 entities 模型中
    entities: Entities = Field(
        default_factory=Entities,
        description="提取的实体信息，包含通用实体和业务实体，与 state['entities'] 结构一致"
    )

    # ==================== Retrieval Strategy Recommendations ====================

    # Retrieval strategy recommendations (for original query or when not decomposing)
    recommended_retrieval_strategy: List[PipelineOption] = Field(
        default_factory=list,
        description="推荐的检索策略（当不分解时使用，分解时参考各子查询的策略）"
    )

    # Recommended retrieval count
    recommended_k: int = Field(
        default=5,
        description="推荐的检索文档数量（当不分解时使用）"
    )

    # Whether multi-round retrieval is needed
    needs_multi_round_retrieval: bool = Field(
        default=False,
        description="是否需要多轮检索（对于多跳查询或需要迭代细化的查询）"
    )

    # ==================== Meta Information ====================

    # Confidence level
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="意图识别的置信度（0-1）"
    )

    # Reasoning process explanation
    reasoning: str = Field(
        description="意图识别和分解决策的推理过程（使用与查询相同的语言）"
    )
