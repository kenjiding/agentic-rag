"""Web Search Tool - 基于 DuckDuckGo Search (DDGS)

使用最新的 ddgs 库实现 Web 搜索功能，替代之前的 MCP 方案。

2025 最佳实践：
- 使用稳定的 Python 库，避免外部依赖问题
- 支持同步和异步调用
- 提供清晰的错误处理
"""
from typing import Annotated, List, Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import Field
import logging

logger = logging.getLogger(__name__)

# 导入 DDGS（最新版本使用 ddgs 包）
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
    logger.warning("无法导入 DDGS，请确保已安装 ddgs 包: pip install ddgs")


def create_web_search_tool() -> List:
    """
    创建 Web 搜索工具
    
    Returns:
        LangChain Tool 列表
    """
    if DDGS is None:
        logger.error("DDGS 未安装，无法创建 Web 搜索工具")
        return []
    
    try:
        # 创建 DDGS 实例
        ddgs = DDGS()
        
        @tool
        def web_search(
            query: Annotated[
                str,
                Field(
                    description="搜索查询字符串",
                    examples=["What is the capital of France?", "最新iPhone价格", "Python异步编程"]
                )
            ],
            max_results: Annotated[
                int,
                Field(
                    default=5,
                    description="最大返回结果数量",
                    examples=[5, 10, 20]
                )
            ] = 5
        ) -> str:
            """
            使用 DuckDuckGo 搜索网络信息
            
            此工具可以搜索最新的网络信息，适用于：
            - 获取实时信息（新闻、股价、天气等）
            - 查找不在知识库中的信息
            - 补充 RAG 检索结果的不足
            
            Args:
                query: 搜索查询字符串
                max_results: 最大返回结果数量（默认5）
            
            Returns:
                格式化的搜索结果字符串，包含标题、URL 和摘要
            """
            try:
                logger.info(f"执行 Web 搜索: {query}")
                
                # 执行搜索（新版本 ddgs 使用 query 作为位置参数）
                results = list(ddgs.text(
                    query=query,
                    max_results=max_results,
                    safesearch='moderate'
                ))
                
                if not results:
                    return f"未找到与 '{query}' 相关的搜索结果。"
                
                # 格式化结果
                formatted_results = []
                for i, result in enumerate(results, 1):
                    title = result.get('title', '无标题')
                    url = result.get('href', '无URL')
                    body = result.get('body', '无摘要')
                    
                    formatted_results.append(
                        f"[结果 {i}]\n"
                        f"标题: {title}\n"
                        f"URL: {url}\n"
                        f"摘要: {body}\n"
                    )
                
                result_text = "\n".join(formatted_results)
                logger.info(f"Web 搜索完成，找到 {len(results)} 个结果")
                
                return result_text
                
            except Exception as e:
                error_msg = f"Web 搜索执行失败: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg
        
        return [web_search]
        
    except Exception as e:
        logger.error(f"创建 Web 搜索工具失败: {e}", exc_info=True)
        return []


async def get_web_search_tools() -> List:
    """
    异步获取 Web 搜索工具（兼容旧接口）
    
    Returns:
        LangChain Tool 列表
    """
    return create_web_search_tool()
