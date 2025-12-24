"""检索质量评估器

该模块提供了检索质量评估的核心功能，包括：
- EvaluationResult: 评估结果的数据结构
- RetrievalQualityEvaluator: 检索质量评估器，实现多种评估策略

设计原则：
1. 可配置：通过 RetrievalQualityConfig 灵活配置评估参数
2. 可扩展：支持添加新的评估指标（如关键词覆盖率、文档长度等）
3. 高效：优先使用语义相似度，失败时自动回退到文本重叠度
4. 可解释：提供详细的评估结果和调试信息
"""
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
import numpy as np

from src.agentic_rag.evaluation_config import (
    RetrievalQualityConfig,
    RetrievalQualityWeights
)
from src.agentic_rag.threshold_config import ThresholdConfig, EvaluationThresholds


@dataclass
class EvaluationResult:
    """
    评估结果数据结构
    
    包含评估的所有关键信息，便于分析和调试
    """
    final_score: float  # 最终综合评分 (0-1)
    meets_threshold: bool  # 是否满足阈值
    avg_score: float  # 平均相关性分数
    max_score: float  # 最高相关性分数
    diversity_bonus: float  # 多样性奖励值
    relevance_scores: List[float]  # 每个文档的相关性分数列表
    details: Optional[Dict] = None  # 详细信息（可选，用于调试和分析）
    
    def to_dict(self) -> dict:
        """
        转换为字典格式
        
        Returns:
            包含所有评估结果的字典
        """
        result = {
            "final_score": self.final_score,
            "meets_threshold": self.meets_threshold,
            "avg_score": self.avg_score,
            "max_score": self.max_score,
            "diversity_bonus": self.diversity_bonus,
            "relevance_scores": self.relevance_scores
        }
        if self.details:
            result["details"] = self.details
        return result
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"EvaluationResult("
            f"score={self.final_score:.3f}, "
            f"meets_threshold={self.meets_threshold}, "
            f"docs={len(self.relevance_scores)})"
        )


class RetrievalQualityEvaluator:
    """检索质量评估器"""
    
    def __init__(
        self,
        embeddings,  # 嵌入模型
        config: Optional[RetrievalQualityConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None
    ):
        """
        初始化评估器
        
        Args:
            embeddings: 嵌入模型（用于计算语义相似度）
            config: 评估配置
            threshold_config: 阈值配置（用于获取评估相关的阈值）
        """
        self.embeddings = embeddings
        self.config = config or RetrievalQualityConfig()
        # 获取评估阈值配置（如果提供）
        self.eval_thresholds = threshold_config.evaluation if threshold_config else EvaluationThresholds()
    
    def evaluate(
        self,
        query: str,
        retrieved_docs: List[Document],
        threshold: Optional[float] = None,
        include_details: bool = False
    ) -> EvaluationResult:
        """
        评估检索质量
        
        Args:
            query: 查询文本
            retrieved_docs: 检索到的文档
            threshold: 质量阈值（如果为None，使用配置中的阈值）
            include_details: 是否包含详细信息
            
        Returns:
            EvaluationResult: 评估结果
        """
        if not retrieved_docs:
            return EvaluationResult(
                final_score=0.0,
                meets_threshold=False,
                avg_score=0.0,
                max_score=0.0,
                diversity_bonus=0.0,
                relevance_scores=[],
                details={"reason": "no_documents"} if include_details else None
            )
        
        # 获取阈值
        if threshold is None:
            threshold = self.config.get_threshold(query)
        
        # 计算每个文档的相关性分数
        relevance_scores = self._calculate_relevance_scores(query, retrieved_docs)
        
        # 计算基础指标
        avg_score = np.mean(relevance_scores) if relevance_scores else 0.0
        max_score = max(relevance_scores) if relevance_scores else 0.0
        
        # 计算多样性奖励
        diversity_bonus = self._calculate_diversity_bonus(relevance_scores)
        
        # 计算综合分数
        final_score = self._calculate_comprehensive_score(
            avg_score=avg_score,
            max_score=max_score,
            diversity_bonus=diversity_bonus,
            relevance_scores=relevance_scores,
            retrieved_docs=retrieved_docs
        )
        
        # 构建详细信息（如果需要）
        details = None
        if include_details:
            details = self._build_details(
                query=query,
                relevance_scores=relevance_scores,
                avg_score=avg_score,
                max_score=max_score,
                diversity_bonus=diversity_bonus,
                threshold=threshold
            )
        
        return EvaluationResult(
            final_score=final_score,
            meets_threshold=final_score >= threshold,
            avg_score=avg_score,
            max_score=max_score,
            diversity_bonus=diversity_bonus,
            relevance_scores=relevance_scores,
            details=details
        )
    
    def _calculate_relevance_scores(
        self,
        query: str,
        retrieved_docs: List[Document]
    ) -> List[float]:
        """计算所有文档的相关性分数"""
        scores = []
        for doc in retrieved_docs:
            score = self._calculate_relevance_score(query, doc.page_content)
            scores.append(score)
        return scores
    
    def _calculate_relevance_score(self, query: str, doc_content: str) -> float:
        """
        计算查询与文档的相关性分数
        
        策略：优先使用语义相似度，如果失败则回退到文本重叠度
        
        Args:
            query: 查询文本
            doc_content: 文档内容
            
        Returns:
            相关性分数 (0-1)
        """
        # 优先使用语义相似度（更准确）
        try:
            return self._semantic_relevance_score(query, doc_content)
        except Exception as e:
            # 如果语义相似度计算失败（例如嵌入模型出错），使用文本重叠度作为回退
            return self._simple_overlap_score(query, doc_content)
    
    def _semantic_relevance_score(self, query: str, doc_content: str) -> float:
        """使用嵌入向量计算语义相似度"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            doc_embedding = self.embeddings.embed_query(doc_content)

            query_vec = np.array(query_embedding)
            doc_vec = np.array(doc_embedding)

            # 计算点积
            dot_product = np.dot(query_vec, doc_vec)
            # 计算向量长度
            norm_query = np.linalg.norm(query_vec)
            norm_doc = np.linalg.norm(doc_vec)

            if norm_query == 0 or norm_doc == 0:
                return 0.0

            # 计算余弦相似度
            cosine_sim = dot_product / (norm_query * norm_doc)
            # 将 [-1, 1] 映射到 [0, 1]
            return (cosine_sim + 1) / 2
        except Exception:
            return self._simple_overlap_score(query, doc_content)
    
    def _simple_overlap_score(self, query: str, doc_content: str) -> float:
        """简单的文本重叠度计算（改进版）"""
        query_words = set(query.lower().split())
        doc_words = set(doc_content.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & doc_words)
        
        # Jaccard 相似度
        union = len(query_words | doc_words)
        jaccard = overlap / union if union > 0 else 0.0
        
        # 覆盖率
        coverage = overlap / len(query_words) if len(query_words) > 0 else 0.0
        
        # 综合评分（使用配置的权重）
        score = jaccard * self.eval_thresholds.jaccard_weight + coverage * self.eval_thresholds.coverage_weight
        
        # 文档长度奖励（如果启用）
        if self.config.enable_length_evaluation:
            length_penalty = self._calculate_length_penalty(doc_content)
            score *= length_penalty
        
        return min(1.0, max(0.0, score))
    
    def _calculate_diversity_bonus(self, relevance_scores: List[float]) -> float:
        """
        计算多样性奖励
        
        奖励逻辑：
        - 统计相关性分数 >= diversity_threshold 的文档数量
        - 每个高质量文档贡献 bonus_per_doc
        - 最多达到 max_diversity_bonus
        
        Args:
            relevance_scores: 所有文档的相关性分数列表
            
        Returns:
            多样性奖励值 (0 到 max_diversity_bonus)
        """
        if not self.config.enable_diversity_bonus or not relevance_scores:
            return 0.0
        
        # 统计高质量文档数量
        high_quality_count = sum(
            1 for score in relevance_scores 
            if score >= self.config.diversity_threshold
        )
        
        # 计算每个文档的奖励值（使用配置的最大期望文档数）
        max_expected_docs = self.eval_thresholds.max_expected_docs
        bonus_per_doc = self.config.max_diversity_bonus / max_expected_docs
        
        # 计算总奖励（不超过最大值）
        bonus = min(
            self.config.max_diversity_bonus,
            high_quality_count * bonus_per_doc
        )
        
        return bonus
    
    def _calculate_length_penalty(self, doc_content: str) -> float:
        """计算文档长度惩罚/奖励因子"""
        if not self.config.enable_length_evaluation:
            return 1.0
        
        doc_length = len(doc_content.split())
        min_len, max_len = self.config.optimal_length_range
        
        if doc_length < self.config.min_doc_length:
            # 文档太短，轻微惩罚（使用配置的惩罚因子）
            return self.eval_thresholds.short_doc_penalty
        elif doc_length > self.config.max_doc_length:
            # 文档太长，轻微惩罚（使用配置的惩罚因子）
            return self.eval_thresholds.long_doc_penalty
        elif min_len <= doc_length <= max_len:
            # 最佳长度范围，轻微奖励（使用配置的奖励因子）
            return self.eval_thresholds.optimal_length_bonus
        else:
            return 1.0
    
    def _calculate_comprehensive_score(
        self,
        avg_score: float,
        max_score: float,
        diversity_bonus: float,
        relevance_scores: List[float],
        retrieved_docs: List[Document]
    ) -> float:
        """
        计算综合评分
        
        评分公式（与原始代码保持一致）：
        final_score = (avg_score × avg_weight) + (max_score × max_weight) + diversity_bonus
        
        注意：
        - diversity_bonus 直接相加（不是加权），因为它已经是小的奖励值（0-0.2）
        - 这与原始代码保持一致，确保向后兼容
        - 最终的权重总和是 (avg_weight + max_weight) + diversity_bonus，可能超过1.0
        - 通过 min(1.0, final_score) 确保分数在 [0, 1] 范围内
        """
        weights = self.config.weights
        
        # 基础评分：平均分和最高分的加权组合
        base_score = (
            avg_score * weights.avg_score_weight +
            max_score * weights.max_score_weight
        )
        
        # 多样性奖励：直接相加（与原始代码保持一致）
        # diversity_bonus 的范围是 [0, max_diversity_bonus]（通常是 0-0.2）
        # 作为一个小奖励直接加上去，而不是加权
        # 这样保持了与原始代码的兼容性
        
        # 关键词覆盖率（如果启用）
        keyword_component = 0.0
        if self.config.enable_keyword_coverage and weights.keyword_coverage_weight > 0:
            keyword_coverage = self._calculate_keyword_coverage(
                query=None,  # 需要从上下文中获取
                retrieved_docs=retrieved_docs
            )
            keyword_component = keyword_coverage * weights.keyword_coverage_weight
        
        # 文档长度评分（如果启用）
        length_component = 0.0
        if self.config.enable_length_evaluation and weights.length_score_weight > 0:
            avg_length_score = self._calculate_average_length_score(retrieved_docs)
            length_component = avg_length_score * weights.length_score_weight
        
        # 综合所有组件
        # 注意：diversity_bonus 直接相加，不是加权（保持与原始代码一致）
        final_score = (
            base_score + 
            diversity_bonus +  # 直接相加，不是加权！
            keyword_component + 
            length_component
        )
        
        # 确保分数在 [0, 1] 范围内
        return min(1.0, max(0.0, final_score))
    
    def _calculate_keyword_coverage(
        self,
        query: Optional[str],
        retrieved_docs: List[Document]
    ) -> float:
        """
        计算关键词覆盖率（如果启用）
        
        TODO: 实现关键词覆盖率计算
        目前返回 0.0，表示不使用此功能
        """
        # 如果未启用关键词覆盖率，返回默认值
        if not self.config.enable_keyword_coverage:
            return 0.0
        
        # TODO: 实现关键词提取和覆盖率计算
        # 1. 从查询中提取关键词
        # 2. 检查文档中包含多少关键词
        # 3. 计算覆盖率
        return 0.0
    
    def _calculate_average_length_score(self, retrieved_docs: List[Document]) -> float:
        """计算文档长度的平均评分"""
        if not retrieved_docs:
            return 1.0
        
        length_scores = [
            self._calculate_length_penalty(doc.page_content)
            for doc in retrieved_docs
        ]
        return np.mean(length_scores) if length_scores else 1.0
    
    def _build_details(
        self,
        query: str,
        relevance_scores: List[float],
        avg_score: float,
        max_score: float,
        diversity_bonus: float,
        threshold: float
    ) -> Dict:
        """
        构建评估的详细信息
        
        用于调试、日志记录和结果分析
        """
        return {
            "query": query,
            "doc_count": len(relevance_scores),
            "scores": {
                "avg": avg_score,
                "max": max_score,
                "min": min(relevance_scores) if relevance_scores else 0.0,
                "std": float(np.std(relevance_scores)) if relevance_scores else 0.0
            },
            "diversity_bonus": diversity_bonus,
            "threshold": threshold,
            "weights": {
                "avg": self.config.weights.avg_score_weight,
                "max": self.config.weights.max_score_weight,
                "diversity": self.config.weights.diversity_bonus_weight,
                "keyword": self.config.weights.keyword_coverage_weight,
                "length": self.config.weights.length_score_weight
            },
            "config": {
                "dynamic_threshold": self.config.dynamic_threshold,
                "enable_diversity_bonus": self.config.enable_diversity_bonus,
                "enable_length_evaluation": self.config.enable_length_evaluation,
                "enable_keyword_coverage": self.config.enable_keyword_coverage
            }
        }
