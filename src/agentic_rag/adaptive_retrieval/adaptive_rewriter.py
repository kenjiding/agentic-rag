"""自适应查询重写器模块

该模块根据检索失败分析结果，动态生成多种查询变体，
以从不同角度尝试检索，避免多轮检索使用相同查询的问题。

重写策略：
1. EXPANSION - 扩展：添加同义词和相关概念
2. REFINEMENT - 精炼：增加限定条件，提高精确度
3. GENERALIZATION - 泛化：移除限定条件，扩大范围
4. PERSPECTIVE_SHIFT - 换角度：从不同角度重新表述
5. DECOMPOSITION - 分解：将复杂查询拆分
6. SYNTHESIS - 综合：基于已有信息生成新查询
7. ENTITY_FOCUS - 实体聚焦：聚焦于特定实体

2025 最佳实践：
- 基于失败原因的针对性重写
- 多策略组合生成多样化查询变体
- 保留原始意图的同时改变表述方式
"""

from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agentic_rag.threshold_config import ThresholdConfig
from .failure_analyzer import FailureType, FailureAnalysisResult


class RewriteStrategy(str, Enum):
    """查询重写策略枚举"""
    EXPANSION = "expansion"              # 扩展
    REFINEMENT = "refinement"            # 精炼
    GENERALIZATION = "generalization"    # 泛化
    PERSPECTIVE_SHIFT = "perspective_shift"  # 换角度
    DECOMPOSITION = "decomposition"      # 分解
    SYNTHESIS = "synthesis"              # 综合
    ENTITY_FOCUS = "entity_focus"        # 实体聚焦


class QueryVariantOutput(BaseModel):
    """LLM 生成的查询变体输出"""

    variants: List[Dict[str, str]] = Field(
        default_factory=list,  # 默认空列表，避免 LLM 未返回时验证失败
        description="生成的查询变体列表，每个包含 query 和 strategy"
    )

    reasoning: str = Field(
        default="",  # 默认空字符串
        description="生成变体的推理过程"
    )


@dataclass
class QueryVariant:
    """查询变体"""

    # 变体查询文本
    query: str

    # 使用的重写策略
    strategy: RewriteStrategy

    # 变体目的说明
    purpose: str = ""

    # 预期的检索策略
    recommended_retrieval_strategy: List[str] = field(
        default_factory=lambda: ["semantic"]
    )

    # 预期的 k 值
    recommended_k: int = 5

    # 优先级 (0 最高)
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "query": self.query,
            "strategy": self.strategy.value,
            "purpose": self.purpose,
            "recommended_retrieval_strategy": self.recommended_retrieval_strategy,
            "recommended_k": self.recommended_k,
            "priority": self.priority
        }


# 失败类型到推荐重写策略的映射
FAILURE_TO_STRATEGY_MAP: Dict[FailureType, List[RewriteStrategy]] = {
    FailureType.NO_RESULTS: [
        RewriteStrategy.GENERALIZATION,
        RewriteStrategy.EXPANSION,
        RewriteStrategy.PERSPECTIVE_SHIFT
    ],
    FailureType.LOW_RELEVANCE: [
        RewriteStrategy.REFINEMENT,
        RewriteStrategy.ENTITY_FOCUS,
        RewriteStrategy.PERSPECTIVE_SHIFT
    ],
    FailureType.INCOMPLETE: [
        RewriteStrategy.DECOMPOSITION,
        RewriteStrategy.EXPANSION,
        RewriteStrategy.SYNTHESIS
    ],
    FailureType.REDUNDANT: [
        RewriteStrategy.PERSPECTIVE_SHIFT,
        RewriteStrategy.SYNTHESIS,
        RewriteStrategy.ENTITY_FOCUS
    ],
    FailureType.SHALLOW: [
        RewriteStrategy.REFINEMENT,
        RewriteStrategy.DECOMPOSITION,
        RewriteStrategy.SYNTHESIS
    ],
    FailureType.MISALIGNED: [
        RewriteStrategy.PERSPECTIVE_SHIFT,
        RewriteStrategy.REFINEMENT,
        RewriteStrategy.ENTITY_FOCUS
    ],
    FailureType.UNKNOWN: [
        RewriteStrategy.EXPANSION,
        RewriteStrategy.PERSPECTIVE_SHIFT
    ]
}


class AdaptiveQueryRewriter:
    """自适应查询重写器

    根据失败分析结果，动态生成多种查询变体。
    支持基于规则和基于 LLM 的重写。
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        threshold_config: Optional[ThresholdConfig] = None,
        max_variants: int = 5
    ):
        """
        初始化自适应重写器

        Args:
            llm: LLM 实例
            threshold_config: 阈值配置
            max_variants: 最大生成变体数量
        """
        self.threshold_config = threshold_config or ThresholdConfig.default()
        self.max_variants = max_variants

        if llm is None:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.llm = llm

    def rewrite(
        self,
        query: str,
        failure_analysis: FailureAnalysisResult,
        query_intent: Optional[Dict[str, Any]] = None,
        previous_variants: Optional[List[QueryVariant]] = None,
        context: Optional[str] = None
    ) -> List[QueryVariant]:
        """
        根据失败分析生成查询变体

        Args:
            query: 原始查询
            failure_analysis: 失败分析结果
            query_intent: 查询意图
            previous_variants: 之前使用过的变体（避免重复）
            context: 上下文信息（来自之前的检索结果）

        Returns:
            查询变体列表
        """
        # 确定要使用的重写策略
        strategies = self._select_strategies(failure_analysis)

        # 生成变体
        variants = self._generate_variants(
            query=query,
            strategies=strategies,
            failure_analysis=failure_analysis,
            query_intent=query_intent,
            previous_variants=previous_variants,
            context=context
        )

        # 去重和排序
        variants = self._deduplicate_and_rank(variants, previous_variants)

        return variants[:self.max_variants]

    def _select_strategies(
        self,
        failure_analysis: FailureAnalysisResult
    ) -> List[RewriteStrategy]:
        """根据失败分析选择重写策略"""

        strategies: List[RewriteStrategy] = []

        # 根据主要失败类型选择策略
        primary_strategies = FAILURE_TO_STRATEGY_MAP.get(
            failure_analysis.primary_failure,
            [RewriteStrategy.EXPANSION]
        )
        strategies.extend(primary_strategies)

        # 根据其他失败类型补充策略
        for failure_type in failure_analysis.failure_types:
            if failure_type != failure_analysis.primary_failure:
                additional = FAILURE_TO_STRATEGY_MAP.get(failure_type, [])
                for s in additional:
                    if s not in strategies:
                        strategies.append(s)

        # 根据严重程度调整
        if failure_analysis.severity >= 0.8:
            # 高严重程度时，优先使用激进策略
            if RewriteStrategy.PERSPECTIVE_SHIFT not in strategies:
                strategies.insert(0, RewriteStrategy.PERSPECTIVE_SHIFT)

        return strategies[:4]  # 最多使用 4 种策略

    def _generate_variants(
        self,
        query: str,
        strategies: List[RewriteStrategy],
        failure_analysis: FailureAnalysisResult,
        query_intent: Optional[Dict[str, Any]],
        previous_variants: Optional[List[QueryVariant]],
        context: Optional[str]
    ) -> List[QueryVariant]:
        """使用 LLM 生成查询变体"""

        # 准备策略说明
        strategy_descriptions = self._get_strategy_descriptions(strategies)

        # 准备失败分析信息
        failure_info = f"""
失败类型: {[ft.value for ft in failure_analysis.failure_types]}
主要问题: {failure_analysis.primary_failure.value}
严重程度: {failure_analysis.severity:.2f}
缺失方面: {failure_analysis.missing_aspects}
改进建议: {failure_analysis.suggested_refinements}
替代角度: {failure_analysis.alternative_angles}
"""

        # 准备意图信息
        intent_info = ""
        if query_intent:
            intent_info = f"""
意图类型: {query_intent.get('intent_type', '未知')}
实体: {query_intent.get('entities', [])}
时间点: {query_intent.get('time_points', [])}
"""

        # 准备已使用变体信息（避免重复）
        used_queries = ""
        if previous_variants:
            used_queries = "\n".join([f"- {v.query}" for v in previous_variants])

        # 准备上下文信息
        context_info = ""
        if context:
            context_info = f"\n已获取的上下文信息（可用于生成更精确的变体）:\n{context[:500]}..."

        template = """你是一个专业的查询优化专家。请根据失败分析结果，生成多个不同角度的查询变体。

# 原始查询
{query}

# 失败分析
{failure_info}

# 意图信息
{intent_info}

# 已使用过的查询（请避免生成类似的）
{used_queries}
{context_info}

# 重写策略说明
{strategy_descriptions}

# 生成要求

1. 针对每种策略，生成1-2个查询变体
2. 每个变体必须与原查询有明显不同
3. 保持原始查询的核心意图
4. 使用与原始查询相同的语言
5. 避免生成与已使用查询相似的变体

请生成查询变体，每个变体包含：
- query: 变体查询文本
- strategy: 使用的策略（{strategy_list}）

输出JSON："""

        prompt = ChatPromptTemplate.from_template(template)

        try:
            # 使用 function_calling 方法以获得更好的兼容性
            structured_llm = self.llm.with_structured_output(
                QueryVariantOutput,
                method="function_calling"
            )
            chain = prompt | structured_llm

            output: QueryVariantOutput = chain.invoke({
                "query": query,
                "failure_info": failure_info,
                "intent_info": intent_info or "（无意图信息）",
                "used_queries": used_queries or "（无）",
                "context_info": context_info,
                "strategy_descriptions": strategy_descriptions,
                "strategy_list": ", ".join([s.value for s in strategies])
            })

            # 转换为 QueryVariant 对象
            variants = []
            for i, v in enumerate(output.variants):
                try:
                    strategy = RewriteStrategy(v.get("strategy", "expansion"))
                except ValueError:
                    strategy = RewriteStrategy.EXPANSION

                # 根据策略确定推荐的检索策略
                retrieval_strategy = self._get_recommended_retrieval_strategy(strategy)

                variants.append(QueryVariant(
                    query=v.get("query", query),
                    strategy=strategy,
                    purpose=v.get("purpose", ""),
                    recommended_retrieval_strategy=retrieval_strategy,
                    recommended_k=self._get_recommended_k(strategy),
                    priority=i
                ))

            # 如果 LLM 返回空列表，回退到规则生成
            if not variants:
                print("[自适应重写器] LLM 返回空变体列表，回退到规则生成")
                return self._rule_based_rewrite(query, strategies, failure_analysis)

            return variants

        except Exception as e:
            print(f"[自适应重写器] LLM 生成错误: {e}")
            # 回退到规则生成
            return self._rule_based_rewrite(query, strategies, failure_analysis)

    def _rule_based_rewrite(
        self,
        query: str,
        strategies: List[RewriteStrategy],
        failure_analysis: FailureAnalysisResult
    ) -> List[QueryVariant]:
        """基于规则的回退重写"""

        variants = []

        for i, strategy in enumerate(strategies):
            if strategy == RewriteStrategy.EXPANSION:
                # 简单扩展：添加 "相关" 或 "详细"
                variants.append(QueryVariant(
                    query=f"{query}的相关信息和详细内容",
                    strategy=strategy,
                    purpose="扩展查询范围",
                    recommended_retrieval_strategy=["hybrid"],
                    recommended_k=8,
                    priority=i
                ))

            elif strategy == RewriteStrategy.GENERALIZATION:
                # 泛化：提取核心概念
                words = query.split()
                if len(words) > 3:
                    # 保留前半部分关键词
                    core_words = words[:len(words)//2 + 1]
                    variants.append(QueryVariant(
                        query=" ".join(core_words),
                        strategy=strategy,
                        purpose="泛化查询",
                        recommended_retrieval_strategy=["semantic"],
                        recommended_k=10,
                        priority=i
                    ))

            elif strategy == RewriteStrategy.PERSPECTIVE_SHIFT:
                # 换角度：添加不同的疑问词
                perspectives = ["什么是", "如何理解", "关于", "有关"]
                for p in perspectives[:1]:
                    variants.append(QueryVariant(
                        query=f"{p}{query}",
                        strategy=strategy,
                        purpose="换角度提问",
                        recommended_retrieval_strategy=["hybrid"],
                        recommended_k=5,
                        priority=i
                    ))

            elif strategy == RewriteStrategy.REFINEMENT:
                # 精炼：基于失败分析的建议
                if failure_analysis.suggested_refinements:
                    suggestion = failure_analysis.suggested_refinements[0]
                    variants.append(QueryVariant(
                        query=f"{query}（{suggestion}）",
                        strategy=strategy,
                        purpose="精炼查询",
                        recommended_retrieval_strategy=["rerank"],
                        recommended_k=5,
                        priority=i
                    ))

            elif strategy == RewriteStrategy.ENTITY_FOCUS:
                # 实体聚焦：如果有实体信息
                if failure_analysis.missing_aspects:
                    aspect = failure_analysis.missing_aspects[0]
                    variants.append(QueryVariant(
                        query=aspect,
                        strategy=strategy,
                        purpose="聚焦缺失方面",
                        recommended_retrieval_strategy=["semantic"],
                        recommended_k=5,
                        priority=i
                    ))

        return variants

    def _get_strategy_descriptions(
        self,
        strategies: List[RewriteStrategy]
    ) -> str:
        """获取策略说明"""

        descriptions = {
            RewriteStrategy.EXPANSION: "扩展(expansion): 添加同义词、相关概念，扩大查询覆盖范围",
            RewriteStrategy.REFINEMENT: "精炼(refinement): 增加限定条件，提高查询精确度",
            RewriteStrategy.GENERALIZATION: "泛化(generalization): 移除限定条件，扩大检索范围",
            RewriteStrategy.PERSPECTIVE_SHIFT: "换角度(perspective_shift): 从完全不同的角度重新表述问题",
            RewriteStrategy.DECOMPOSITION: "分解(decomposition): 将复杂查询拆分为多个简单子查询",
            RewriteStrategy.SYNTHESIS: "综合(synthesis): 基于已有信息生成更精确的查询",
            RewriteStrategy.ENTITY_FOCUS: "实体聚焦(entity_focus): 聚焦于查询中的特定实体"
        }

        return "\n".join([
            f"- {descriptions.get(s, str(s))}"
            for s in strategies
        ])

    def _get_recommended_retrieval_strategy(
        self,
        rewrite_strategy: RewriteStrategy
    ) -> List[str]:
        """根据重写策略推荐检索策略"""

        strategy_map = {
            RewriteStrategy.EXPANSION: ["hybrid"],
            RewriteStrategy.REFINEMENT: ["rerank"],
            RewriteStrategy.GENERALIZATION: ["semantic"],
            RewriteStrategy.PERSPECTIVE_SHIFT: ["hybrid"],
            RewriteStrategy.DECOMPOSITION: ["semantic"],
            RewriteStrategy.SYNTHESIS: ["rerank"],
            RewriteStrategy.ENTITY_FOCUS: ["semantic", "rerank"]
        }

        return strategy_map.get(rewrite_strategy, ["semantic"])

    def _get_recommended_k(self, rewrite_strategy: RewriteStrategy) -> int:
        """根据重写策略推荐 k 值"""

        k_map = {
            RewriteStrategy.EXPANSION: 8,
            RewriteStrategy.REFINEMENT: 5,
            RewriteStrategy.GENERALIZATION: 10,
            RewriteStrategy.PERSPECTIVE_SHIFT: 7,
            RewriteStrategy.DECOMPOSITION: 5,
            RewriteStrategy.SYNTHESIS: 5,
            RewriteStrategy.ENTITY_FOCUS: 5
        }

        return k_map.get(rewrite_strategy, 5)

    def _deduplicate_and_rank(
        self,
        variants: List[QueryVariant],
        previous_variants: Optional[List[QueryVariant]]
    ) -> List[QueryVariant]:
        """去重和排序"""

        # 收集已使用的查询
        used_queries = set()
        if previous_variants:
            for v in previous_variants:
                used_queries.add(v.query.lower().strip())

        # 去重
        unique_variants = []
        seen_queries = set()

        for v in variants:
            normalized = v.query.lower().strip()

            # 跳过已使用的
            if normalized in used_queries:
                continue

            # 跳过重复的
            if normalized in seen_queries:
                continue

            seen_queries.add(normalized)
            unique_variants.append(v)

        # 按优先级排序
        unique_variants.sort(key=lambda x: x.priority)

        return unique_variants

    def generate_for_missing_aspects(
        self,
        missing_aspects: List[str],
        original_query: str
    ) -> List[QueryVariant]:
        """为缺失的信息方面生成专门的查询变体"""

        variants = []

        for i, aspect in enumerate(missing_aspects):
            variants.append(QueryVariant(
                query=aspect,
                strategy=RewriteStrategy.ENTITY_FOCUS,
                purpose=f"补充缺失信息: {aspect[:50]}",
                recommended_retrieval_strategy=["semantic"],
                recommended_k=5,
                priority=i
            ))

        return variants
