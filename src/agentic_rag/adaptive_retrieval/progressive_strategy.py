"""渐进式检索策略模块

该模块管理多轮检索过程中的策略升级和参数调整，
确保每一轮检索都采用不同的配置以获得更好的结果。

策略升级路径：
Round 1: semantic (精确匹配) → k=5
Round 2: hybrid (多策略融合) → k=8, 放宽阈值
Round 3: hybrid + rerank + 查询变体 → k=10
Round 4: 多变体并行检索 + Web Search → k=15
Round 5: 最大努力模式 → 所有策略

2025 最佳实践：
- 渐进式参数升级
- 多策略组合
- 失败驱动的策略选择
- 动态阈值调整
"""

from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

from src.agentic_rag.threshold_config import ThresholdConfig
from .failure_analyzer import FailureType, FailureAnalysisResult
from .adaptive_rewriter import QueryVariant, RewriteStrategy


@dataclass
class RetrievalRoundConfig:
    """单轮检索配置"""

    # 轮次编号 (0-based)
    round: int

    # 检索策略列表
    strategies: List[str] = field(default_factory=lambda: ["semantic"])

    # 检索数量
    k: int = 5

    # 质量阈值（低于此值触发下一轮）
    quality_threshold: float = 0.7

    # 是否启用查询改写
    enable_rewrite: bool = False

    # 使用的查询变体
    query_variants: List[QueryVariant] = field(default_factory=list)

    # 是否启用重排序
    enable_rerank: bool = False

    # 是否允许 Web Search
    allow_web_search: bool = False

    # 描述
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "round": self.round,
            "strategies": self.strategies,
            "k": self.k,
            "quality_threshold": self.quality_threshold,
            "enable_rewrite": self.enable_rewrite,
            "query_variants": [v.to_dict() for v in self.query_variants],
            "enable_rerank": self.enable_rerank,
            "allow_web_search": self.allow_web_search,
            "description": self.description
        }


class ProgressiveRetrievalStrategy:
    """渐进式检索策略管理器

    管理多轮检索的策略升级，根据失败分析动态调整配置。
    """

    def __init__(
        self,
        threshold_config: Optional[ThresholdConfig] = None,
        max_rounds: int = 5
    ):
        """
        初始化策略管理器

        Args:
            threshold_config: 阈值配置
            max_rounds: 最大检索轮次
        """
        self.threshold_config = threshold_config or ThresholdConfig.default()
        self.max_rounds = max_rounds

        # 初始化默认策略升级路径
        self._init_default_progression()

    def _init_default_progression(self):
        """初始化默认的策略升级路径"""

        base_threshold = self.threshold_config.retrieval.quality_threshold

        self.default_progression = [
            # Round 0: 基础语义检索
            RetrievalRoundConfig(
                round=0,
                strategies=["semantic"],
                k=self.threshold_config.retrieval.default_k,
                quality_threshold=base_threshold,
                enable_rewrite=False,
                enable_rerank=False,
                allow_web_search=False,
                description="基础语义检索"
            ),

            # Round 1: 混合检索 + 放宽阈值
            RetrievalRoundConfig(
                round=1,
                strategies=["hybrid"],
                k=self.threshold_config.retrieval.default_k + 3,
                quality_threshold=base_threshold - 0.1,
                enable_rewrite=True,
                enable_rerank=False,
                allow_web_search=False,
                description="混合检索 + 查询改写"
            ),

            # Round 2: 混合检索 + 重排序 + 查询变体
            RetrievalRoundConfig(
                round=2,
                strategies=["hybrid", "rerank"],
                k=self.threshold_config.retrieval.default_k + 5,
                quality_threshold=base_threshold - 0.15,
                enable_rewrite=True,
                enable_rerank=True,
                allow_web_search=True,
                description="混合检索 + 重排序 + Web Search"
            ),

            # Round 3: 多策略并行 + Web Search
            RetrievalRoundConfig(
                round=3,
                strategies=["semantic", "hybrid", "rerank"],
                k=self.threshold_config.retrieval.default_k + 8,
                quality_threshold=base_threshold - 0.2,
                enable_rewrite=True,
                enable_rerank=True,
                allow_web_search=True,
                description="多策略并行 + 多变体检索"
            ),

            # Round 4: 最大努力模式
            RetrievalRoundConfig(
                round=4,
                strategies=["semantic", "hybrid", "rerank", "bm25"],
                k=self.threshold_config.retrieval.default_k + 10,
                quality_threshold=0.3,  # 最低阈值
                enable_rewrite=True,
                enable_rerank=True,
                allow_web_search=True,
                description="最大努力模式"
            )
        ]

    def get_round_config(
        self,
        round: int,
        failure_analysis: Optional[FailureAnalysisResult] = None,
        query_variants: Optional[List[QueryVariant]] = None
    ) -> RetrievalRoundConfig:
        """
        获取指定轮次的检索配置

        Args:
            round: 轮次编号 (0-based)
            failure_analysis: 失败分析结果（用于动态调整）
            query_variants: 可用的查询变体

        Returns:
            该轮次的检索配置
        """
        # 确保轮次在有效范围内
        round = min(round, len(self.default_progression) - 1)

        # 获取基础配置
        base_config = self.default_progression[round]

        # 如果没有失败分析，直接返回基础配置
        if failure_analysis is None:
            config = RetrievalRoundConfig(
                round=round,
                strategies=base_config.strategies.copy(),
                k=base_config.k,
                quality_threshold=base_config.quality_threshold,
                enable_rewrite=base_config.enable_rewrite,
                enable_rerank=base_config.enable_rerank,
                allow_web_search=base_config.allow_web_search,
                description=base_config.description
            )
            if query_variants:
                config.query_variants = query_variants
            return config

        # 根据失败分析动态调整配置
        return self._adjust_for_failure(
            base_config=base_config,
            failure_analysis=failure_analysis,
            query_variants=query_variants
        )

    def _adjust_for_failure(
        self,
        base_config: RetrievalRoundConfig,
        failure_analysis: FailureAnalysisResult,
        query_variants: Optional[List[QueryVariant]]
    ) -> RetrievalRoundConfig:
        """根据失败分析调整配置"""

        strategies = base_config.strategies.copy()
        k = base_config.k
        quality_threshold = base_config.quality_threshold
        enable_rerank = base_config.enable_rerank

        # 根据主要失败类型调整
        primary_failure = failure_analysis.primary_failure

        if primary_failure == FailureType.NO_RESULTS:
            # 无结果：大幅扩展检索范围
            k = min(k + 5, 20)
            quality_threshold = max(quality_threshold - 0.15, 0.3)
            if "bm25" not in strategies:
                strategies.append("bm25")

        elif primary_failure == FailureType.LOW_RELEVANCE:
            # 低相关性：启用重排序，增加 k
            enable_rerank = True
            k = min(k + 3, 15)
            if "rerank" not in strategies:
                strategies.append("rerank")

        elif primary_failure == FailureType.REDUNDANT:
            # 冗余：强制使用查询变体
            if not query_variants:
                # 使用换角度策略
                pass  # 将在外部通过 query_variants 传入

        elif primary_failure == FailureType.INCOMPLETE:
            # 不完整：增加 k，多策略
            k = min(k + 5, 15)
            if "hybrid" not in strategies:
                strategies.insert(0, "hybrid")

        elif primary_failure == FailureType.SHALLOW:
            # 浅层：使用重排序提高质量
            enable_rerank = True
            if "rerank" not in strategies:
                strategies.append("rerank")

        # 根据严重程度进一步调整
        if failure_analysis.severity >= 0.8:
            k = min(k + 3, 20)
            quality_threshold = max(quality_threshold - 0.1, 0.3)

        return RetrievalRoundConfig(
            round=base_config.round,
            strategies=strategies,
            k=k,
            quality_threshold=quality_threshold,
            enable_rewrite=True,  # 失败后总是启用改写
            query_variants=query_variants or [],
            enable_rerank=enable_rerank,
            allow_web_search=base_config.allow_web_search,
            description=f"{base_config.description} (已根据失败分析调整)"
        )

    def should_continue(
        self,
        round: int,
        retrieval_quality: float,
        failure_analysis: Optional[FailureAnalysisResult] = None
    ) -> bool:
        """
        判断是否应该继续下一轮检索

        Args:
            round: 当前轮次
            retrieval_quality: 当前检索质量
            failure_analysis: 失败分析结果

        Returns:
            是否继续
        """
        # 超过最大轮次
        if round >= self.max_rounds - 1:
            return False

        # 获取当前轮次的质量阈值
        config = self.get_round_config(round)

        # 质量已达标
        if retrieval_quality >= config.quality_threshold:
            return False

        # 如果有失败分析，检查是否需要继续
        if failure_analysis:
            # 如果检测到冗余，需要更激进的策略
            if FailureType.REDUNDANT in failure_analysis.failure_types:
                return True

            # 如果建议重新识别意图，可能需要继续
            if failure_analysis.needs_intent_reclassification:
                return True

        return True

    def get_strategy_summary(self, round: int) -> str:
        """获取策略摘要"""
        if round < len(self.default_progression):
            config = self.default_progression[round]
            return f"Round {round}: {config.description} (k={config.k}, strategies={config.strategies})"
        return f"Round {round}: 超出预定义范围"

    def get_all_strategies(self) -> List[str]:
        """获取所有策略描述"""
        return [
            self.get_strategy_summary(i)
            for i in range(len(self.default_progression))
        ]


class AdaptiveRetrievalOrchestrator:
    """自适应检索编排器

    整合失败分析、查询重写和渐进策略，提供统一的自适应检索接口。
    """

    def __init__(
        self,
        threshold_config: Optional[ThresholdConfig] = None
    ):
        """
        初始化编排器

        Args:
            threshold_config: 阈值配置
        """
        self.threshold_config = threshold_config or ThresholdConfig.default()
        self.strategy_manager = ProgressiveRetrievalStrategy(threshold_config)

        # 延迟导入以避免循环依赖
        self._failure_analyzer = None
        self._query_rewriter = None

    @property
    def failure_analyzer(self):
        """延迟初始化失败分析器"""
        if self._failure_analyzer is None:
            from .failure_analyzer import RetrievalFailureAnalyzer
            self._failure_analyzer = RetrievalFailureAnalyzer(
                threshold_config=self.threshold_config
            )
        return self._failure_analyzer

    @property
    def query_rewriter(self):
        """延迟初始化查询重写器"""
        if self._query_rewriter is None:
            from .adaptive_rewriter import AdaptiveQueryRewriter
            self._query_rewriter = AdaptiveQueryRewriter(
                threshold_config=self.threshold_config
            )
        return self._query_rewriter

    def plan_next_round(
        self,
        query: str,
        current_round: int,
        retrieved_docs: List,
        retrieval_quality: float,
        query_intent: Optional[Dict[str, Any]] = None,
        retrieval_history: Optional[List] = None,
        previous_variants: Optional[List[QueryVariant]] = None
    ) -> Optional[RetrievalRoundConfig]:
        """
        规划下一轮检索

        Args:
            query: 原始查询
            current_round: 当前轮次
            retrieved_docs: 当前检索结果
            retrieval_quality: 当前检索质量
            query_intent: 查询意图
            retrieval_history: 检索历史
            previous_variants: 之前使用的查询变体

        Returns:
            下一轮配置，如果不需要继续则返回 None
        """
        # 分析当前失败原因
        failure_analysis = self.failure_analyzer.analyze(
            query=query,
            retrieved_docs=retrieved_docs,
            retrieval_quality=retrieval_quality,
            query_intent=query_intent,
            retrieval_history=retrieval_history,
            iteration=current_round
        )

        # 判断是否需要继续
        if not self.strategy_manager.should_continue(
            round=current_round,
            retrieval_quality=retrieval_quality,
            failure_analysis=failure_analysis
        ):
            return None

        # 生成查询变体
        query_variants = self.query_rewriter.rewrite(
            query=query,
            failure_analysis=failure_analysis,
            query_intent=query_intent,
            previous_variants=previous_variants
        )

        # 获取下一轮配置
        next_round = current_round + 1
        config = self.strategy_manager.get_round_config(
            round=next_round,
            failure_analysis=failure_analysis,
            query_variants=query_variants
        )

        return config

    def get_failure_analysis(
        self,
        query: str,
        retrieved_docs: List,
        retrieval_quality: float,
        query_intent: Optional[Dict[str, Any]] = None,
        retrieval_history: Optional[List] = None,
        iteration: int = 0
    ) -> FailureAnalysisResult:
        """获取失败分析结果"""
        return self.failure_analyzer.analyze(
            query=query,
            retrieved_docs=retrieved_docs,
            retrieval_quality=retrieval_quality,
            query_intent=query_intent,
            retrieval_history=retrieval_history,
            iteration=iteration
        )
