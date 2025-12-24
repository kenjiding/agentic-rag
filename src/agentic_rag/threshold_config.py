"""阈值配置模块

该模块定义了 Agentic RAG 系统中所有硬编码阈值的配置类。
所有阈值都集中在这里管理，便于统一调整和优化。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalThresholds:
    """检索相关阈值配置"""
    # 检索质量阈值
    quality_threshold: float = 0.7  # 检索质量阈值（默认）
    quality_threshold_simple_query: float = 0.65  # 简单查询的阈值（<5词）
    quality_threshold_complex_query: float = 0.75  # 复杂查询的阈值（>15词）
    
    # 检索数量
    default_k: int = 5  # 默认检索文档数量
    
    # 检索策略切换阈值
    quality_for_hybrid_search: float = 0.7  # 低于此值切换到混合检索
    
    # 查询改写相关
    context_length_for_rewrite: int = 200  # 用于查询改写的上下文长度
    min_query_keywords: int = 2  # 查询改写验证：最少关键词数


@dataclass
class GenerationThresholds:
    """生成相关阈值配置"""
    # 答案质量阈值
    answer_quality_threshold: float = 0.7  # 答案质量阈值
    
    # 答案长度检查
    min_answer_length: int = 30  # 最小答案长度（字符数）
    min_answer_length_strict: int = 50  # 严格模式下的最小答案长度
    
    # 答案质量快速检查
    low_quality_threshold: float = 0.3  # 低质量阈值（用于快速判断）


@dataclass
class EvaluationThresholds:
    """评估相关阈值配置"""
    # 文本重叠度权重
    jaccard_weight: float = 0.6  # Jaccard相似度权重
    coverage_weight: float = 0.4  # 覆盖率权重

    # 多样性奖励
    max_expected_docs: int = 4  # 最大期望文档数（用于计算多样性奖励）

    # 文档长度评估
    short_doc_penalty: float = 0.9  # 短文档惩罚因子
    long_doc_penalty: float = 0.95  # 长文档惩罚因子
    optimal_length_bonus: float = 1.1  # 最佳长度奖励因子


@dataclass
class DetectorThresholds:
    """信息需求检测器阈值配置"""
    # 嵌入向量检测阈值
    embedding_similarity_threshold: float = 0.75  # 嵌入相似度阈值（默认）
    embedding_similarity_threshold_fallback: float = 0.7  # 回退阈值
    
    # 答案长度检查
    min_answer_length_quick: int = 30  # 快速检查的最小答案长度
    min_answer_length_embedding: int = 50  # 嵌入检测的最小答案长度
    
    # 快速检查阈值
    low_quality_threshold: float = 0.3  # 低质量阈值（用于快速判断是否需要更多信息）


@dataclass
class DecisionThresholds:
    """决策节点阈值配置"""
    # 质量阈值
    retrieval_quality_threshold: float = 0.7  # 检索质量阈值
    answer_quality_threshold: float = 0.7  # 答案质量阈值
    
    # 迭代控制
    max_retrieval_iterations: int = 2  # 最大检索迭代次数（用于控制检索重试）
    min_iterations_for_fallback: int = 2  # 最小迭代次数（用于回退策略）


@dataclass
class RetrieverThresholds:
    """检索器内部阈值配置"""
    # LLM温度
    llm_temperature: float = 0.0  # 检索器使用的LLM温度
    
    # 查询改写
    max_query_words: int = 20  # 查询改写后的最大词数
    context_length_limit: int = 500  # 上下文长度限制（用于查询改写）
    min_query_keywords_for_validation: int = 2  # 查询改写验证：最少关键词数（用于判断改写是否偏离原意）


@dataclass
class GeneratorThresholds:
    """生成器内部阈值配置"""
    # LLM温度
    default_temperature: float = 0.1  # 生成器默认温度


@dataclass
class IntentClassificationThresholds:
    """意图识别阈值配置"""
    # LLM温度
    llm_temperature: float = 0.0  # 意图识别使用的LLM温度（低温度保证稳定性）

    # 意图识别开关
    enable_intent_classification: bool = True  # 是否启用意图识别

    # 置信度阈值
    min_confidence: float = 0.7  # 最小置信度（低于此值使用回退策略）

    def to_intent_config(self):
        """转换为通用IntentConfig

        用于集成通用的intent模块。
        """
        from src.intent.config import IntentConfig

        return IntentConfig(
            llm_temperature=self.llm_temperature,
            enable_intent_classification=self.enable_intent_classification,
            min_confidence=self.min_confidence,
        )


@dataclass
class AdaptiveRetrievalThresholds:
    """自适应检索阈值配置（简化版）

    只保留核心配置：
    - 失败分析相关配置
    - 渐进式策略相关配置
    - 动态意图重识别相关配置
    """

    # 失败分析相关
    enable_failure_analysis: bool = True  # 是否启用失败分析
    enable_llm_deep_analysis: bool = True  # 是否启用 LLM 深度分析
    redundancy_threshold: float = 0.7  # 冗余检测阈值（重叠度超过此值视为冗余）

    # 渐进式策略相关
    enable_progressive_strategy: bool = True  # 是否启用渐进式策略
    max_retrieval_rounds: int = 5  # 最大检索轮次
    k_increment_per_round: int = 3  # 每轮 k 值增量

    # 动态意图重识别相关
    enable_intent_reclassification: bool = True  # 是否启用动态意图重识别
    max_reclassification_count: int = 2  # 最大重识别次数
    reclassification_trigger_round: int = 2  # 触发重识别的轮次阈值


@dataclass
class ThresholdConfig:
    """统一的阈值配置类
    
    包含所有模块的阈值配置，便于统一管理和调整。
    """
    # 各模块阈值配置
    retrieval: RetrievalThresholds = None
    generation: GenerationThresholds = None
    evaluation: EvaluationThresholds = None
    detector: DetectorThresholds = None
    decision: DecisionThresholds = None
    retriever: RetrieverThresholds = None
    generator: GeneratorThresholds = None
    intent_classification: IntentClassificationThresholds = None
    adaptive_retrieval: AdaptiveRetrievalThresholds = None  # 自适应检索配置

    def __post_init__(self):
        """初始化后处理：为None的配置使用默认值"""
        if self.retrieval is None:
            self.retrieval = RetrievalThresholds()
        if self.generation is None:
            self.generation = GenerationThresholds()
        if self.evaluation is None:
            self.evaluation = EvaluationThresholds()
        if self.detector is None:
            self.detector = DetectorThresholds()
        if self.decision is None:
            self.decision = DecisionThresholds()
        if self.retriever is None:
            self.retriever = RetrieverThresholds()
        if self.generator is None:
            self.generator = GeneratorThresholds()
        if self.intent_classification is None:
            self.intent_classification = IntentClassificationThresholds()
        if self.adaptive_retrieval is None:
            self.adaptive_retrieval = AdaptiveRetrievalThresholds()
    
    @classmethod
    def default(cls) -> "ThresholdConfig":
        """默认配置：平衡各种因素"""
        return cls()
    
    @classmethod
    def strict(cls) -> "ThresholdConfig":
        """严格配置：更高的阈值，适合对质量要求高的场景"""
        return cls(
            retrieval=RetrievalThresholds(
                quality_threshold=0.8,
                quality_threshold_simple_query=0.75,
                quality_threshold_complex_query=0.85,
                quality_for_hybrid_search=0.8
            ),
            generation=GenerationThresholds(
                answer_quality_threshold=0.8,
                min_answer_length=50,
                min_answer_length_strict=80
            ),
            decision=DecisionThresholds(
                retrieval_quality_threshold=0.8,
                answer_quality_threshold=0.8
            ),
            detector=DetectorThresholds(
                embedding_similarity_threshold=0.8,
                embedding_similarity_threshold_fallback=0.75
            )
        )
    
    @classmethod
    def lenient(cls) -> "ThresholdConfig":
        """宽松配置：较低的阈值，适合快速迭代场景"""
        return cls(
            retrieval=RetrievalThresholds(
                quality_threshold=0.6,
                quality_threshold_simple_query=0.55,
                quality_threshold_complex_query=0.65,
                quality_for_hybrid_search=0.6
            ),
            generation=GenerationThresholds(
                answer_quality_threshold=0.6,
                min_answer_length=20,
                min_answer_length_strict=30
            ),
            decision=DecisionThresholds(
                retrieval_quality_threshold=0.6,
                answer_quality_threshold=0.6
            ),
            detector=DetectorThresholds(
                embedding_similarity_threshold=0.7,
                embedding_similarity_threshold_fallback=0.65
            )
        )
    
    def to_dict(self) -> dict:
        """转换为字典格式（用于序列化）"""
        return {
            "retrieval": {
                "quality_threshold": self.retrieval.quality_threshold,
                "quality_threshold_simple_query": self.retrieval.quality_threshold_simple_query,
                "quality_threshold_complex_query": self.retrieval.quality_threshold_complex_query,
                "default_k": self.retrieval.default_k,
                "quality_for_hybrid_search": self.retrieval.quality_for_hybrid_search,
                "context_length_for_rewrite": self.retrieval.context_length_for_rewrite,
                "min_query_keywords": self.retrieval.min_query_keywords,
            },
            "generation": {
                "answer_quality_threshold": self.generation.answer_quality_threshold,
                "min_answer_length": self.generation.min_answer_length,
                "min_answer_length_strict": self.generation.min_answer_length_strict,
                "low_quality_threshold": self.generation.low_quality_threshold,
            },
            "evaluation": {
                "jaccard_weight": self.evaluation.jaccard_weight,
                "coverage_weight": self.evaluation.coverage_weight,
                "max_expected_docs": self.evaluation.max_expected_docs,
                "short_doc_penalty": self.evaluation.short_doc_penalty,
                "long_doc_penalty": self.evaluation.long_doc_penalty,
                "optimal_length_bonus": self.evaluation.optimal_length_bonus,
            },
            "detector": {
                "embedding_similarity_threshold": self.detector.embedding_similarity_threshold,
                "embedding_similarity_threshold_fallback": self.detector.embedding_similarity_threshold_fallback,
                "min_answer_length_quick": self.detector.min_answer_length_quick,
                "min_answer_length_embedding": self.detector.min_answer_length_embedding,
                "low_quality_threshold": self.detector.low_quality_threshold,
            },
            "decision": {
                "retrieval_quality_threshold": self.decision.retrieval_quality_threshold,
                "answer_quality_threshold": self.decision.answer_quality_threshold,
                "max_retrieval_iterations": self.decision.max_retrieval_iterations,
                "min_iterations_for_fallback": self.decision.min_iterations_for_fallback,
            },
            "retriever": {
                "llm_temperature": self.retriever.llm_temperature,
                "max_query_words": self.retriever.max_query_words,
                "context_length_limit": self.retriever.context_length_limit,
                "min_query_keywords_for_validation": self.retriever.min_query_keywords_for_validation,
            },
            "generator": {
                "default_temperature": self.generator.default_temperature,
            },
            "intent_classification": {
                "llm_temperature": self.intent_classification.llm_temperature,
                "enable_intent_classification": self.intent_classification.enable_intent_classification,
                "min_confidence": self.intent_classification.min_confidence,
            },
            "adaptive_retrieval": {
                "enable_failure_analysis": self.adaptive_retrieval.enable_failure_analysis,
                "enable_llm_deep_analysis": self.adaptive_retrieval.enable_llm_deep_analysis,
                "redundancy_threshold": self.adaptive_retrieval.redundancy_threshold,
                "enable_progressive_strategy": self.adaptive_retrieval.enable_progressive_strategy,
                "max_retrieval_rounds": self.adaptive_retrieval.max_retrieval_rounds,
                "k_increment_per_round": self.adaptive_retrieval.k_increment_per_round,
                "enable_intent_reclassification": self.adaptive_retrieval.enable_intent_reclassification,
                "max_reclassification_count": self.adaptive_retrieval.max_reclassification_count,
                "reclassification_trigger_round": self.adaptive_retrieval.reclassification_trigger_round,
            },
        }

