"""智能检索器实现 - 2025 最佳实践版

基于 2025 年最新研究实现的检索策略：
1. 真正的混合检索 (BM25 + Dense + RRF 融合)
2. Cross-encoder 重排序
3. 多策略检索融合

参考:
- https://www.pinecone.io/learn/advanced-rag-techniques/
- https://humanloop.com/blog/rag-architectures
"""
from src.agentic_rag.retriever.intelligent_retriever import IntelligentRetriever
from src.agentic_rag.retriever.bm25_retriever import BM25Retriever
from src.agentic_rag.retriever.reranker import CrossEncoderReranker
from src.agentic_rag.retriever.fusion import reciprocal_rank_fusion

__all__ = [
    "IntelligentRetriever",
    "BM25Retriever",
    "CrossEncoderReranker",
    "reciprocal_rank_fusion",
]

