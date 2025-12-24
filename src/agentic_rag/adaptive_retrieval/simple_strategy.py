"""简化的渐进式检索策略

只保留核心功能：
1. 基于轮次的简单策略升级
2. 基于失败分析的策略调整
3. 去掉查询重写等复杂功能
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from src.agentic_rag.threshold_config import ThresholdConfig
from .failure_analyzer import FailureAnalysisResult


@dataclass
class SimpleRetrievalConfig:
    """简化的检索配置"""
    strategies: List[str]
    k: int
    enable_rewrite: bool = False
    description: str = ""


class SimpleProgressiveStrategy:
    """简化的渐进式策略管理器
    
    只保留核心功能：
    - 基于轮次的策略升级
    - 基于失败分析的简单调整
    """
    
    def __init__(self, threshold_config: Optional[ThresholdConfig] = None):
        self.threshold_config = threshold_config or ThresholdConfig.default()
        adaptive_config = self.threshold_config.adaptive_retrieval
        self.max_rounds = adaptive_config.max_retrieval_rounds if adaptive_config else 5
        self.k_increment = adaptive_config.k_increment_per_round if adaptive_config else 3
        self.base_k = self.threshold_config.retrieval.default_k
    
    def get_round_config(
        self,
        round: int,
        failure_analysis: Optional[FailureAnalysisResult] = None
    ) -> SimpleRetrievalConfig:
        """获取指定轮次的检索配置
        
        Args:
            round: 轮次编号（0-based）
            failure_analysis: 失败分析结果（可选）
        
        Returns:
            检索配置
        """
        # 基础策略升级路径
        if round == 0:
            strategies = ["semantic"]
            k = self.base_k
            description = "首轮：语义检索"
        elif round == 1:
            strategies = ["hybrid"]
            k = min(self.base_k + self.k_increment, 12)
            description = "第2轮：混合检索，扩大k值"
        elif round == 2:
            strategies = ["hybrid"]
            k = min(self.base_k + self.k_increment * 2, 15)
            description = "第3轮：混合检索，进一步扩大k值"
        else:
            strategies = ["hybrid"]
            k = min(self.base_k + self.k_increment * round, 20)
            description = f"第{round+1}轮：混合检索，最大k值"
        
        # 第2轮及以后默认启用改写
        enable_rewrite = (round >= 1)
        
        # 基于失败分析的简单调整
        if failure_analysis:
            primary_failure = failure_analysis.primary_failure.value
            
            # 如果是无结果，尝试扩大k值
            if primary_failure == "no_results":
                k = min(k + 5, 20)
                description += " (无结果，扩大k值)"
            
            # 如果是低相关性，确保启用查询改写
            elif primary_failure == "low_relevance":
                enable_rewrite = True
                description += " (低相关性，启用改写)"
        
        return SimpleRetrievalConfig(
            strategies=strategies,
            k=k,
            enable_rewrite=enable_rewrite,
            description=description
        )
    
    def should_continue(
        self,
        round: int,
        retrieval_quality: float,
        failure_analysis: Optional[FailureAnalysisResult] = None
    ) -> bool:
        """判断是否应该继续下一轮检索"""
        # 超过最大轮次
        if round >= self.max_rounds - 1:
            return False
        
        # 质量已达标
        quality_threshold = self.threshold_config.retrieval.quality_threshold
        if retrieval_quality >= quality_threshold:
            return False
        
        # 如果有失败分析，检查是否需要继续
        if failure_analysis:
            # 如果严重程度很高，继续尝试
            if failure_analysis.severity > 0.8:
                return True
            
            # 如果是无结果，继续尝试
            if failure_analysis.primary_failure.value == "no_results":
                return True
        
        return True

