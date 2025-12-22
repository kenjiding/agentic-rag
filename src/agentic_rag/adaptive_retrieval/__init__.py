"""自适应检索策略模块

该模块实现了企业级的自适应多轮检索策略，解决了传统 RAG 系统中
"多轮检索使用相同问题和策略导致结果重复"的问题。

核心组件：
1. RetrievalFailureAnalyzer - 检索失败分析器
2. AdaptiveQueryRewriter - 自适应查询重写器
3. ProgressiveRetrievalStrategy - 渐进式检索策略管理器

2025 最佳实践：
- 失败原因分析驱动的动态策略调整
- 多维度查询变体生成
- 渐进式参数升级
- 支持动态意图重识别
"""

from .failure_analyzer import (
    RetrievalFailureAnalyzer,
    FailureType,
    FailureAnalysisResult
)
from .adaptive_rewriter import (
    AdaptiveQueryRewriter,
    RewriteStrategy,
    QueryVariant
)
from .progressive_strategy import (
    ProgressiveRetrievalStrategy,
    RetrievalRoundConfig
)

__all__ = [
    # 失败分析器
    "RetrievalFailureAnalyzer",
    "FailureType",
    "FailureAnalysisResult",
    # 自适应重写器
    "AdaptiveQueryRewriter",
    "RewriteStrategy",
    "QueryVariant",
    # 渐进式策略
    "ProgressiveRetrievalStrategy",
    "RetrievalRoundConfig",
]
