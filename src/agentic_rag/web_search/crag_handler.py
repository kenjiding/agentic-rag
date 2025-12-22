"""Corrective RAG 处理器实现

实现 CRAG 论文中的核心逻辑：
1. 评估检索质量
2. 根据质量决定是否需要 Web 搜索
3. 融合本地和 Web 搜索结果
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.agentic_rag.web_search.web_search_tool import WebSearchTool


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
1. **保持查询的核心意图和关键实体不变**
2. 使用与原查询相同的语言（中文查询保持中文）
3. 转换为适合搜索引擎的形式（关键词组合）
4. 保留重要的限定词（时间、地点、数量等）
5. 如果是问句，转换为陈述式关键词
6. 保持查询简洁（5-15 个词）
7. 只返回优化后的查询，不要解释

示例：
- "2019和2020年苹果营收对比" -> "苹果公司 2019年 2020年 营收 财报"
- "特斯拉为什么成功" -> "特斯拉 成功原因 分析"
- "量子计算的原理和应用" -> "量子计算 原理 应用 介绍"

优化后的查询："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        try:
            response = chain.invoke({"query": query})
            optimized = response.content.strip()
            # 去除可能的引号
            optimized = optimized.strip('"\'')
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
        执行 Web 搜索（带重试机制）

        如果优化后的查询返回空结果或不相关结果，
        会尝试使用原始查询重新搜索。

        Args:
            query: 查询文本
            optimize_query: 是否优化查询

        Returns:
            Web 搜索结果文档列表
        """
        if not self.web_search.available:
            print("[CRAG] Web 搜索不可用")
            return []

        # 第一次尝试：使用优化后的查询
        search_query = query
        # if optimize_query:
        #     search_query = self.optimize_query_for_web(query)

        results = self.web_search.search(search_query, self.max_web_results)

        # 如果结果为空且使用了优化查询，尝试原始查询
        if not results and optimize_query and search_query != query:
            print(f"[CRAG] 优化查询无结果，尝试原始查询: '{query}'")
            results = self.web_search.search(query, self.max_web_results)

        return results

    def refine_web_results(
        self,
        query: str,
        web_docs: List[Document]
    ) -> List[Document]:
        """
        精炼 Web 搜索结果（CRAG 知识精炼）

        使用 LLM 评估每个结果的相关性，过滤掉不相关的结果

        Args:
            query: 原始查询
            web_docs: Web 搜索结果

        Returns:
            精炼后的文档列表
        """
        if not web_docs:
            return []

        # 使用 LLM 评估每个结果的相关性
        relevance_template = """你是一个信息相关性评估专家。请判断以下搜索结果是否与用户查询相关。

用户查询：{query}

搜索结果：
标题：{title}
内容：{content}

评估标准：
1. 内容是否直接回答或涉及用户查询的主题
2. 信息是否有助于回答用户的问题
3. 内容质量是否可靠（非广告、非垃圾信息）

请只回答 "相关" 或 "不相关"，不要解释。"""

        refined_docs = []
        for doc in web_docs:
            try:
                title = doc.metadata.get("title", "")
                content = doc.page_content[:500]  # 限制长度

                prompt = ChatPromptTemplate.from_template(relevance_template)
                chain = prompt | self.llm

                response = chain.invoke({
                    "query": query,
                    "title": title,
                    "content": content
                })

                result = response.content.strip()
                if "相关" in result and "不相关" not in result:
                    refined_docs.append(doc)
                    print(f"[CRAG] ✓ 保留: {title[:50]}...")
                else:
                    print(f"[CRAG] ✗ 过滤: {title[:50]}...")

            except Exception as e:
                print(f"[CRAG] 相关性评估失败: {e}")
                # 评估失败时保留结果
                refined_docs.append(doc)

        print(f"[CRAG] 精炼后保留 {len(refined_docs)}/{len(web_docs)} 个结果")
        return refined_docs

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

