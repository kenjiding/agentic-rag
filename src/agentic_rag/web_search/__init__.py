"""Web Search 工具 - Corrective RAG 回退机制

基于 2025 年 Corrective RAG (CRAG) 论文实现：
当本地知识库检索质量不足时，回退到 Web 搜索获取外部信息。

参考:
- https://arxiv.org/abs/2401.15884 (Corrective RAG)
- https://arxiv.org/abs/2501.09136 (Agentic RAG Survey)
"""
from src.agentic_rag.web_search.web_search_tool import WebSearchTool
from src.agentic_rag.web_search.crag_handler import CorrectiveRAGHandler

__all__ = [
    "WebSearchTool",
    "CorrectiveRAGHandler",
    "create_web_search_tool",
    "create_crag_handler",
]


# 便捷函数
def create_web_search_tool(**kwargs) -> WebSearchTool:
    """创建 Web 搜索工具"""
    return WebSearchTool(**kwargs)


def create_crag_handler(**kwargs) -> CorrectiveRAGHandler:
    """创建 CRAG 处理器"""
    return CorrectiveRAGHandler(**kwargs)

