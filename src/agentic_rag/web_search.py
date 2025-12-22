"""Web Search 工具 - Corrective RAG 回退机制

基于 2025 年 Corrective RAG (CRAG) 论文实现：
当本地知识库检索质量不足时，回退到 Web 搜索获取外部信息。

参考:
- https://arxiv.org/abs/2401.15884 (Corrective RAG)
- https://arxiv.org/abs/2501.09136 (Agentic RAG Survey)
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 尝试导入 Web Search 库
try:
    from duckduckgo_search import DDGS
    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False
    print("[警告] duckduckgo-search 未安装，Web Search 将不可用。运行: uv add duckduckgo-search")


class WebSearchTool:
    """Web 搜索工具

    2025 最佳实践 (Corrective RAG)：
    当本地检索质量不足时，使用 Web 搜索获取外部信息
    """

    def __init__(
        self,
        max_results: int = 5,
        region: str = "wt-wt",  # 全球
        safesearch: str = "moderate",
        time_range: Optional[str] = None,  # d: day, w: week, m: month, y: year
    ):
        """
        初始化 Web 搜索工具

        Args:
            max_results: 最大结果数
            region: 搜索区域
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

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region=self.region,
                    safesearch=self.safesearch,
                    timelimit=self.time_range,
                    max_results=k
                ))

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


class CorrectiveRAGHandler:
    """Corrective RAG 处理器

    实现 CRAG 论文中的核心逻辑：
    1. 评估检索质量
    2. 根据质量决定是否需要 Web 搜索
    3. 融合本地和 Web 搜索结果
    """

    def __init__(
        self,
        web_search_tool: Optional[WebSearchTool] = None,
        llm: Optional[ChatOpenAI] = None,
        quality_threshold: float = 0.5,
        max_web_results: int = 3
    ):
        """
        初始化 CRAG 处理器

        Args:
            web_search_tool: Web 搜索工具
            llm: 用于查询优化和知识精炼的 LLM
            quality_threshold: 触发 Web 搜索的质量阈值
            max_web_results: 最大 Web 搜索结果数
        """
        self.web_search = web_search_tool or WebSearchTool()
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.quality_threshold = quality_threshold
        self.max_web_results = max_web_results

    def should_trigger_web_search(
        self,
        retrieval_quality: float,
        iteration_count: int,
        max_iterations: int = 2
    ) -> bool:
        """
        判断是否应该触发 Web 搜索

        CRAG 策略：
        - 如果检索质量低于阈值
        - 且已经尝试了足够次数的本地检索
        - 则触发 Web 搜索

        Args:
            retrieval_quality: 当前检索质量分数
            iteration_count: 当前迭代次数
            max_iterations: 触发 Web 搜索前的最大本地检索迭代次数

        Returns:
            是否应该触发 Web 搜索
        """
        # 质量很低且已经尝试过本地检索
        if retrieval_quality < self.quality_threshold and iteration_count >= max_iterations:
            return True

        # 质量极低时立即触发
        if retrieval_quality < 0.3:
            return True

        return False

    def optimize_query_for_web(self, query: str, context: Optional[str] = None) -> str:
        """
        优化查询以适合 Web 搜索

        Args:
            query: 原始查询
            context: 上下文信息

        Returns:
            优化后的查询
        """
        template = """你是一个搜索专家。请将以下查询优化为更适合 Web 搜索的形式。

原始查询：{query}

要求：
1. 提取核心关键词
2. 移除过于具体或限制性的词汇
3. 添加必要的上下文词汇以提高搜索精度
4. 保持查询简洁（不超过 10 个词）
5. 只返回优化后的查询，不要解释

优化后的查询："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        try:
            response = chain.invoke({"query": query})
            optimized = response.content.strip()
            print(f"[CRAG] 查询优化: '{query}' -> '{optimized}'")
            return optimized
        except Exception as e:
            print(f"[CRAG] 查询优化失败: {e}")
            return query

    def perform_web_search(
        self,
        query: str,
        optimize_query: bool = True
    ) -> List[Document]:
        """
        执行 Web 搜索

        Args:
            query: 查询文本
            optimize_query: 是否优化查询

        Returns:
            Web 搜索结果文档列表
        """
        if not self.web_search.available:
            print("[CRAG] Web 搜索不可用")
            return []

        search_query = query
        if optimize_query:
            search_query = self.optimize_query_for_web(query)

        return self.web_search.search(search_query, self.max_web_results)

    def refine_web_results(
        self,
        query: str,
        web_docs: List[Document]
    ) -> List[Document]:
        """
        精炼 Web 搜索结果（CRAG 知识精炼）

        过滤掉不相关的结果，提取关键信息

        Args:
            query: 原始查询
            web_docs: Web 搜索结果

        Returns:
            精炼后的文档列表
        """
        if not web_docs:
            return []

        # 简单实现：保留所有结果
        # 高级实现可以使用 LLM 评估每个结果的相关性
        return web_docs

    def merge_results(
        self,
        local_docs: List[Document],
        web_docs: List[Document],
        local_weight: float = 0.6
    ) -> List[Document]:
        """
        融合本地和 Web 搜索结果

        Args:
            local_docs: 本地检索结果
            web_docs: Web 搜索结果
            local_weight: 本地结果权重（0-1）

        Returns:
            融合后的文档列表
        """
        # 标记来源
        for doc in local_docs:
            if "source_type" not in doc.metadata:
                doc.metadata["source_type"] = "local_kb"

        for doc in web_docs:
            doc.metadata["source_type"] = "web_search"

        # 简单融合：交替放置
        merged = []
        local_idx = 0
        web_idx = 0

        # 根据权重决定交替比例
        local_ratio = int(local_weight * 10)
        web_ratio = 10 - local_ratio

        while local_idx < len(local_docs) or web_idx < len(web_docs):
            # 添加本地文档
            for _ in range(local_ratio):
                if local_idx < len(local_docs):
                    merged.append(local_docs[local_idx])
                    local_idx += 1

            # 添加 Web 文档
            for _ in range(web_ratio):
                if web_idx < len(web_docs):
                    merged.append(web_docs[web_idx])
                    web_idx += 1

        return merged

    def process(
        self,
        query: str,
        local_docs: List[Document],
        retrieval_quality: float,
        iteration_count: int
    ) -> Dict[str, Any]:
        """
        CRAG 主处理流程

        Args:
            query: 查询文本
            local_docs: 本地检索结果
            retrieval_quality: 检索质量分数
            iteration_count: 当前迭代次数

        Returns:
            处理结果，包含：
            - documents: 最终文档列表
            - used_web_search: 是否使用了 Web 搜索
            - web_results_count: Web 搜索结果数量
        """
        result = {
            "documents": local_docs,
            "used_web_search": False,
            "web_results_count": 0
        }

        # 检查是否需要 Web 搜索
        if self.should_trigger_web_search(retrieval_quality, iteration_count):
            print(f"[CRAG] 触发 Web 搜索 (质量={retrieval_quality:.2f}, 迭代={iteration_count})")

            # 执行 Web 搜索
            web_docs = self.perform_web_search(query)

            if web_docs:
                # 精炼结果
                refined_docs = self.refine_web_results(query, web_docs)

                # 融合结果
                merged_docs = self.merge_results(local_docs, refined_docs)

                result["documents"] = merged_docs
                result["used_web_search"] = True
                result["web_results_count"] = len(refined_docs)

                print(f"[CRAG] Web 搜索完成，融合后共 {len(merged_docs)} 个文档")

        return result


# 便捷函数
def create_web_search_tool(**kwargs) -> WebSearchTool:
    """创建 Web 搜索工具"""
    return WebSearchTool(**kwargs)


def create_crag_handler(**kwargs) -> CorrectiveRAGHandler:
    """创建 CRAG 处理器"""
    return CorrectiveRAGHandler(**kwargs)
