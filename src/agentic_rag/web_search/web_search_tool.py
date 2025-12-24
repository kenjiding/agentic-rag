"""Web 搜索工具实现

2025 最佳实践 (Corrective RAG)：
当本地检索质量不足时，使用 Web 搜索获取外部信息
"""
import re
from typing import List, Optional
from langchain_core.documents import Document

# 尝试导入 Web Search 库（优先使用新包名 ddgs）
try:
    from ddgs import DDGS
    HAS_DUCKDUCKGO = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DUCKDUCKGO = True
    except ImportError:
        HAS_DUCKDUCKGO = False
        print("[警告] ddgs 或 duckduckgo-search 未安装，Web Search 将不可用。运行: uv add ddgs")


def _detect_query_language(query: str) -> tuple[str, bool]:
    """
    检测查询语言，返回适合的搜索区域和是否为中文

    Args:
        query: 查询文本

    Returns:
        (DuckDuckGo 区域代码, 是否为中文查询)
    """
    # 检测中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    chinese_chars = len(chinese_pattern.findall(query))
    total_chars = len(query.replace(" ", ""))

    if total_chars > 0 and chinese_chars / total_chars > 0.3:
        return "cn-zh", True  # 中文区域

    # 默认全球搜索
    return "wt-wt", False


def _is_result_language_match(results: list, is_chinese_query: bool) -> bool:
    """
    检查搜索结果的语言是否与查询语言匹配

    Args:
        results: 搜索结果列表
        is_chinese_query: 查询是否为中文

    Returns:
        是否匹配
    """
    if not results:
        return True  # 空结果不进行语言检查
    
    # 对于非中文查询，不进行语言检查
    if not is_chinese_query:
        return True

    # 检查结果中是否包含中文
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    chinese_result_count = 0
    total_chars_count = 0
    chinese_chars_count = 0

    checked_results = results[:min(3, len(results))]  # 检查前3个结果
    
    for result in checked_results:
        title = result.get("title", "")
        body = result.get("body", "")
        text = title + " " + body
        
        # 统计中文字符数量
        chinese_chars = len(chinese_pattern.findall(text))
        total_chars = len(text.replace(" ", ""))
        
        chinese_chars_count += chinese_chars
        total_chars_count += total_chars
        
        # 如果这个结果包含中文，计数+1
        if chinese_pattern.search(text):
            chinese_result_count += 1

    # 更宽松的匹配策略：
    # 1. 如果至少有一个结果包含中文，认为匹配
    # 2. 或者如果中文字符占比超过10%，认为匹配
    if chinese_result_count > 0:
        return True
    
    if total_chars_count > 0 and chinese_chars_count / total_chars_count > 0.1:
        return True
    
    # 如果所有检查都失败，记录详细信息用于调试
    print(f"[Web Search] 语言检查详情: 检查了{len(checked_results)}个结果, "
          f"包含中文的结果数={chinese_result_count}, "
          f"中文字符占比={chinese_chars_count/total_chars_count if total_chars_count > 0 else 0:.2%}")
    return False


class WebSearchTool:
    """Web 搜索工具

    2025 最佳实践 (Corrective RAG)：
    当本地检索质量不足时，使用 Web 搜索获取外部信息
    """

    def __init__(
        self,
        max_results: int = 5,
        region: str = "auto",  # auto: 自动检测语言, 或指定如 cn-zh, wt-wt 等
        safesearch: str = "moderate",
        time_range: Optional[str] = None,  # d: day, w: week, m: month, y: year
    ):
        """
        初始化 Web 搜索工具

        Args:
            max_results: 最大结果数
            region: 搜索区域 (auto=自动检测, cn-zh=中国, wt-wt=全球)
            safesearch: 安全搜索级别
            time_range: 时间范围
        """
        self.max_results = max_results
        self.region = region
        self.safesearch = safesearch
        self.time_range = time_range
        self.available = HAS_DUCKDUCKGO

        if not self.available:
            print("[Web Search] DuckDuckGo 不可用，Web 搜索功能禁用")

    def search(self, query: str, max_results: Optional[int] = None) -> List[Document]:
        """
        执行 Web 搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数（覆盖默认值）

        Returns:
            包含搜索结果的 Document 列表
        """
        if not self.available:
            print("[Web Search] Web 搜索不可用")
            return []

        k = max_results or self.max_results

        # 自动检测查询语言并选择合适的区域
        search_region = self.region
        is_chinese_query = False
        if search_region == "auto":
            search_region, is_chinese_query = _detect_query_language(query)
            print(f"[Web Search] 自动检测区域: {search_region}, 中文查询: {is_chinese_query}")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region=search_region,
                    safesearch=self.safesearch,
                    timelimit=self.time_range,
                    max_results=k
                ))

                # 注意：不再进行基础关键词过滤，因为：
                # 1. DuckDuckGo 本身已经做了相关性排序
                # 2. 后续的 refine_web_results 会用 LLM 进行精确评估
                # 3. 简单的关键词匹配容易误过滤相关结果
                
                # 检查结果语言是否匹配
                if not _is_result_language_match(results, is_chinese_query):
                    print(f"[Web Search] ⚠️ 结果语言不匹配，返回空结果")
                    print(f"[Web Search] 调试信息: 查询='{query}', 区域={search_region}, "
                          f"结果数={len(results)}, 前3个结果标题: {[r.get('title', '')[:50] for r in results[:3]]}")
                    # 对于中文查询，如果语言不匹配，仍然返回结果（但标记为可能不相关）
                    # 让后续的精炼步骤来处理，而不是直接丢弃
                    # 这样可以避免因为语言检测过于严格而丢失有用信息
                    if is_chinese_query and len(results) > 0:
                        print(f"[Web Search] 中文查询语言不匹配，但仍返回结果供后续精炼")
                        # 继续处理，不返回空
                    else:
                        return []

            documents = []
            for i, result in enumerate(results):
                content = f"{result.get('title', '')}\n\n{result.get('body', '')}"
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": result.get("href", ""),
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "search_rank": i + 1,
                        "source_type": "web_search"
                    }
                )
                documents.append(doc)

            print(f"[Web Search] 搜索 '{query}' 返回 {len(documents)} 个结果")
            return documents

        except Exception as e:
            print(f"[Web Search] 搜索失败: {e}")
            return []

    def search_with_context(
        self,
        query: str,
        context: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[Document]:
        """
        带上下文的搜索（可以优化查询）

        Args:
            query: 原始查询
            context: 上下文信息（之前的检索结果等）
            max_results: 最大结果数

        Returns:
            搜索结果文档列表
        """
        # 简单实现：直接使用原始查询
        # 高级实现可以使用 LLM 优化查询
        return self.search(query, max_results)

