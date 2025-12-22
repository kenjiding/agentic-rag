"""文档分块模块 - 2025 最佳实践版

基于 2025 年最新研究实现的文档分块策略：
1. 语义分块 (Semantic Chunking) - 基于嵌入相似度的断点检测
2. 结构感知分块 - 针对 Markdown/HTML 的结构化分块
3. 递归字符分块 - 作为回退策略

参考:
- https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025
- Semantic Chunking 可提高召回率 9%
"""
from src.agentic_rag.splitter.semantic_chunker import SemanticChunker
from src.agentic_rag.splitter.structure_aware_chunker import StructureAwareChunker
from src.agentic_rag.splitter.docs_splitter import DocsSplitter

__all__ = [
    "SemanticChunker",
    "StructureAwareChunker",
    "DocsSplitter",
    "create_semantic_splitter",
    "create_structure_aware_splitter",
]


# 便捷函数
def create_semantic_splitter(
    embeddings=None,
    **kwargs
) -> SemanticChunker:
    """创建语义分块器的便捷函数"""
    from langchain_openai import OpenAIEmbeddings
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return SemanticChunker(embeddings=embeddings, **kwargs)


def create_structure_aware_splitter(**kwargs) -> StructureAwareChunker:
    """创建结构感知分块器的便捷函数"""
    return StructureAwareChunker(**kwargs)

