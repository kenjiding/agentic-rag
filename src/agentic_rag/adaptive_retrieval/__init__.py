"""自适应检索策略模块（简化版）

核心组件：
1. RetrievalFailureAnalyzer - 检索失败分析器（核心功能）
2. SimpleProgressiveStrategy - 简化的渐进式策略管理器

简化原则：
- 保留失败分析器（最有价值）
- 简化渐进策略（只保留核心升级逻辑）
- 去掉查询重写等复杂功能
"""

from .failure_analyzer import (
    RetrievalFailureAnalyzer,
    FailureType,
    FailureAnalysisResult
)
from .simple_strategy import (
    SimpleProgressiveStrategy,
    SimpleRetrievalConfig
)

__all__ = [
    # 失败分析器（核心）
    "RetrievalFailureAnalyzer",
    "FailureType",
    "FailureAnalysisResult",
    # 简化的渐进式策略
    "SimpleProgressiveStrategy",
    "SimpleRetrievalConfig",
]
