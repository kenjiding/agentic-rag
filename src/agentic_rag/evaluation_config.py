"""检索质量评估配置"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalQualityWeights:
    """
    检索质量评估权重配置
    
    注意：
    - diversity_bonus_weight 在代码中实际上不使用（diversity_bonus 是直接相加的）
    - 保留此字段是为了配置的完整性和未来扩展
    - 权重验证只检查 avg_score_weight + max_score_weight 的总和
    """
    avg_score_weight: float = 0.6  # 平均分数权重
    max_score_weight: float = 0.3  # 最高分数权重
    diversity_bonus_weight: float = 0.1  # 多样性奖励权重（保留字段，实际不使用）
    keyword_coverage_weight: float = 0.0  # 关键词覆盖率权重（可选）
    length_score_weight: float = 0.0  # 文档长度评分权重（可选）
    
    def validate(self) -> None:
        """
        验证权重总和
        
        注意：只验证 avg_score_weight + max_score_weight
        因为 diversity_bonus 是直接相加的，不在权重计算中
        """
        # 只验证基础权重（avg + max），因为 diversity_bonus 是直接相加的
        base_weights_total = self.avg_score_weight + self.max_score_weight
        
        # 基础权重应该在合理范围内（0.8-1.0），因为 diversity_bonus 会直接加上去
        if not (0.8 <= base_weights_total <= 1.0):
            raise ValueError(
                f"基础权重总和（avg + max）应在 0.8-1.0 之间，当前为 {base_weights_total:.2f}"
            )
    
    @classmethod
    def balanced(cls) -> "RetrievalQualityWeights":
        """平衡配置：重视整体质量"""
        return cls(
            avg_score_weight=0.7,
            max_score_weight=0.2,
            diversity_bonus_weight=0.1
        )
    
    @classmethod
    def peak_focused(cls) -> "RetrievalQualityWeights":
        """峰值聚焦配置：重视单个高质量结果"""
        return cls(
            avg_score_weight=0.4,
            max_score_weight=0.5,
            diversity_bonus_weight=0.1
        )
    
    @classmethod
    def diversity_focused(cls) -> "RetrievalQualityWeights":
        """多样性聚焦配置：重视多个相关文档"""
        return cls(
            avg_score_weight=0.6,
            max_score_weight=0.2,
            diversity_bonus_weight=0.2
        )


@dataclass
class RetrievalQualityConfig:
    """检索质量评估配置"""
    # 基础阈值
    default_threshold: float = 0.75
    
    # 动态阈值调整
    dynamic_threshold: bool = True
    simple_query_threshold: float = 0.7  # 简单查询（<5词）
    complex_query_threshold: float = 0.75  # 复杂查询（>15词）
    
    # 权重配置
    weights: RetrievalQualityWeights = None
    
    # 评估选项
    enable_diversity_bonus: bool = True
    diversity_threshold: float = 0.8  # 高质量文档的最低分数阈值
    max_diversity_bonus: float = 0.2  # 多样性奖励的最大值
    
    # 文档长度评估
    enable_length_evaluation: bool = False
    min_doc_length: int = 50  # 最小文档长度（词数）
    max_doc_length: int = 500  # 最大文档长度（词数）
    optimal_length_range: tuple = (50, 500)  # 最佳长度范围
    
    # 关键词匹配
    enable_keyword_coverage: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if self.weights is None:
            self.weights = RetrievalQualityWeights()
        self.weights.validate()
    
    def get_threshold(self, query: Optional[str] = None) -> float:
        """
        获取评估阈值（支持动态调整）
        
        Args:
            query: 查询文本（用于动态调整阈值）
            
        Returns:
            评估阈值（0-1之间的浮点数）
        """
        if not self.dynamic_threshold or query is None:
            return self.default_threshold
        
        query_length = len(query.split())
        if query_length < 5:
            return self.simple_query_threshold
        elif query_length > 15:
            return self.complex_query_threshold
        else:
            return self.default_threshold
    
    @classmethod
    def default(cls) -> "RetrievalQualityConfig":
        """默认配置：平衡各种因素"""
        return cls()
    
    @classmethod
    def strict(cls) -> "RetrievalQualityConfig":
        """严格配置：更高的阈值，适合对质量要求高的场景"""
        return cls(
            default_threshold=0.8,
            simple_query_threshold=0.75,
            complex_query_threshold=0.85,
            dynamic_threshold=True
        )
    
    @classmethod
    def lenient(cls) -> "RetrievalQualityConfig":
        """宽松配置：较低的阈值，适合快速迭代场景"""
        return cls(
            default_threshold=0.6,
            simple_query_threshold=0.55,
            complex_query_threshold=0.65,
            dynamic_threshold=True
        )
    
    @classmethod
    def research_focused(cls) -> "RetrievalQualityConfig":
        """研究导向配置：重视多样性和整体质量"""
        return cls(
            default_threshold=0.7,
            weights=RetrievalQualityWeights.diversity_focused(),
            enable_diversity_bonus=True,
            max_diversity_bonus=0.25  # 更大的多样性奖励
        )
    
    @classmethod
    def fact_focused(cls) -> "RetrievalQualityConfig":
        """事实查询配置：重视单个高质量结果"""
        return cls(
            default_threshold=0.75,
            weights=RetrievalQualityWeights.peak_focused(),
            enable_diversity_bonus=True,
            max_diversity_bonus=0.15  # 较小的多样性奖励
        )
