"""智能检索器实现 - 2025 最佳实践版

支持：
1. 语义检索（Dense）
2. 关键词检索（BM25）
3. 真正的混合检索（BM25 + Dense + RRF 融合）
4. Cross-encoder 重排序
"""
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.agentic_rag.evaluation_config import RetrievalQualityConfig
from src.agentic_rag.evaluators import RetrievalQualityEvaluator
from src.agentic_rag.threshold_config import ThresholdConfig, RetrieverThresholds
from src.intent.models import PipelineOption
from src.agentic_rag.retriever.bm25_retriever import BM25Retriever, HAS_BM25
from src.agentic_rag.retriever.reranker import CrossEncoderReranker, HAS_CROSS_ENCODER
from src.agentic_rag.retriever.fusion import reciprocal_rank_fusion


class IntelligentRetriever:
    """智能检索器 - 2025 最佳实践版

    支持：
    1. 语义检索（Dense）
    2. 关键词检索（BM25）
    3. 真正的混合检索（BM25 + Dense + RRF 融合）
    4. Cross-encoder 重排序
    """

    def __init__(
        self,
        vectorstore: Chroma,
        llm: Optional[ChatOpenAI] = None,
        default_strategy: str = "semantic",
        evaluation_config: Optional[RetrievalQualityConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None,
        enable_bm25: bool = True,
        enable_reranker: bool = True
    ):
        """
        初始化检索器

        Args:
            vectorstore: 向量数据库
            llm: 用于查询改写的 LLM
            default_strategy: 默认检索策略
            evaluation_config: 评估配置
            threshold_config: 阈值配置
            enable_bm25: 是否启用 BM25 检索
            enable_reranker: 是否启用重排序
        """
        self.vectorstore = vectorstore
        self.threshold_config = threshold_config

        # 获取检索器阈值配置
        retriever_thresholds = threshold_config.retriever if threshold_config else RetrieverThresholds()

        # 使用配置的 LLM 温度
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=retriever_thresholds.llm_temperature)
        self.default_strategy = default_strategy
        self.embeddings = vectorstore._embedding_function

        # 初始化评估器
        self.evaluation_config = evaluation_config or RetrievalQualityConfig()
        self.evaluator = RetrievalQualityEvaluator(
            embeddings=self.embeddings,
            config=self.evaluation_config,
            threshold_config=threshold_config
        )

        # 初始化 BM25 检索器
        self.enable_bm25 = enable_bm25 and HAS_BM25
        self.bm25_retriever = None
        if self.enable_bm25:
            self._init_bm25()

        # 初始化 Cross-encoder 重排序器
        self.enable_reranker = enable_reranker and HAS_CROSS_ENCODER
        self.reranker = None
        if self.enable_reranker:
            self.reranker = CrossEncoderReranker()

    def _init_bm25(self):
        """初始化 BM25 索引"""
        try:
            # 从向量数据库获取所有文档
            all_docs = self.vectorstore.get()
            if all_docs and all_docs.get('documents'):
                documents = [
                    Document(
                        page_content=content,
                        metadata=meta if meta else {}
                    )
                    for content, meta in zip(
                        all_docs['documents'],
                        all_docs.get('metadatas', [{}] * len(all_docs['documents']))
                    )
                ]
                self.bm25_retriever = BM25Retriever(documents)
                print(f"[检索] BM25 索引已构建，共 {len(documents)} 个文档")
        except Exception as e:
            print(f"[检索] BM25 初始化失败: {e}")
            self.enable_bm25 = False

    def update_bm25_index(self, documents: List[Document]):
        """更新 BM25 索引"""
        if self.bm25_retriever:
            self.bm25_retriever.add_documents(documents)
        else:
            self.bm25_retriever = BM25Retriever(documents)

    def retrieve(
        self,
        query: str,
        strategy: Optional[str] = None,
        strategies: Optional[List[PipelineOption]] = None,
        context: Optional[str] = None,
        k: int = 3,
        rewrite_query: bool = False,
        split_queries: Optional[List] = None
    ) -> List[Document]:
        """
        检索文档

        Args:
            query: 查询文本
            strategy: 单个检索策略（向后兼容）
            strategies: 检索策略列表
            context: 上下文信息
            k: 返回文档数量
            rewrite_query: 是否改写查询
            split_queries: 拆分后的查询列表（支持两种格式）
                - List[str]: 向后兼容的字符串列表
                - List[Dict]: 包含 query, strategy, k 的字典列表（2025 最佳实践）

        Returns:
            检索到的文档列表
        """
        # 如果是对比查询且有拆分查询，使用拆分查询进行检索
        if split_queries and len(split_queries) >= 1:
            return self._retrieve_with_split_queries(
                split_queries,
                strategies or [strategy or self.default_strategy],
                k
            )

        # 确定使用的策略
        if strategies:
            strategies_to_use = strategies
        else:
            strategies_to_use = [strategy or self.default_strategy]

        # 查询改写
        if rewrite_query and context:
            query = self.rewrite_query(query, context)
            print(f"[检索] 改写后的查询: {query}")

        # 使用组合策略检索
        all_docs = []

        for strategy_item in strategies_to_use:
            docs = self._retrieve_with_strategy(query, strategy_item, context, k)
            all_docs.extend(docs)

        # 后处理（去重等）
        all_docs = self._post_process(all_docs, query)

        print(f"[检索] 使用策略 {strategies_to_use}，检索到 {len(all_docs)} 个文档")
        return all_docs

    def _retrieve_with_split_queries(
        self,
        split_queries: List,
        default_strategies: List[PipelineOption],
        default_k: int
    ) -> List[Document]:
        """
        使用拆分查询进行检索

        2025 最佳实践：每个子查询使用独立的检索策略

        Args:
            split_queries: 拆分后的查询列表，支持两种格式：
                - List[str]: 向后兼容，所有查询使用相同策略
                - List[Dict]: 每个包含 {query, strategy, k}，使用独立策略
            default_strategies: 默认检索策略（当子查询没有指定策略时使用）
            default_k: 默认检索数量

        Returns:
            检索到的文档列表
        """
        all_docs = []

        print(f"[检索] 对比查询：使用 {len(split_queries)} 个拆分查询")

        for item in split_queries:
            # 解析子查询信息
            if isinstance(item, dict):
                # 新格式：包含独立策略
                query_text = item.get("query", "")
                query_strategies = item.get("strategy", default_strategies)
                query_k = item.get("k", default_k)
                # 确保 strategies 是列表
                if not isinstance(query_strategies, list):
                    query_strategies = [query_strategies] if query_strategies else default_strategies
            else:
                # 向后兼容：字符串格式
                query_text = str(item)
                query_strategies = default_strategies
                query_k = default_k

            print(f"[检索] 子查询: '{query_text}' | 策略: {query_strategies} | k={query_k}")

            for strategy in query_strategies:
                docs = self._retrieve_with_strategy(query_text, strategy, None, query_k)
                all_docs.extend(docs)

        # 获取第一个查询文本用于后处理
        first_query = ""
        if split_queries:
            first_item = split_queries[0]
            first_query = first_item.get("query", "") if isinstance(first_item, dict) else str(first_item)

        all_docs = self._post_process(all_docs, first_query)
        print(f"[检索] 对比查询检索完成，共 {len(all_docs)} 个文档")
        return all_docs

    def _retrieve_with_strategy(
        self,
        query: str,
        strategy: str,
        context: Optional[str],
        k: int
    ) -> List[Document]:
        """
        使用指定策略检索

        Args:
            query: 查询文本
            strategy: 检索策略
            context: 上下文信息
            k: 检索数量

        Returns:
            检索到的文档列表
        """
        if strategy == "hybrid":
            # 2025 最佳实践：真正的混合检索
            return self._true_hybrid_search(query, k)
        elif strategy == "rerank":
            # Cross-encoder 重排序
            return self._rerank_search(query, context, k)
        elif strategy == "bm25":
            # 纯 BM25 检索
            return self._bm25_search(query, k)
        else:
            # 默认使用语义检索
            return self._semantic_search(query, k)

    def _semantic_search(self, query: str, k: int) -> List[Document]:
        """语义检索（Dense）"""
        return self.vectorstore.similarity_search(query, k=k)

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """BM25 关键词检索"""
        if not self.enable_bm25 or not self.bm25_retriever:
            print("[检索] BM25 不可用，回退到语义检索")
            return self._semantic_search(query, k)

        results = self.bm25_retriever.search(query, k=k)
        return [doc for doc, _ in results]

    def _true_hybrid_search(self, query: str, k: int) -> List[Document]:
        """
        真正的混合检索 - 2025 最佳实践

        结合 BM25 + Dense + RRF 融合 + 可选重排序
        """
        results_list = []

        # 1. 语义检索（Dense）
        try:
            dense_results = self.vectorstore.similarity_search_with_score(query, k=k * 2)
            dense_docs = [(doc, score) for doc, score in dense_results]
            results_list.append(dense_docs)
            print(f"[检索] Dense 检索: {len(dense_docs)} 个文档")
        except Exception as e:
            print(f"[检索] Dense 检索失败: {e}")

        # 2. BM25 检索（如果可用）
        if self.enable_bm25 and self.bm25_retriever:
            bm25_results = self.bm25_retriever.search(query, k=k * 2)
            if bm25_results:
                results_list.append(bm25_results)
                print(f"[检索] BM25 检索: {len(bm25_results)} 个文档")

        # 3. MMR 检索（增加多样性）
        try:
            mmr_docs = self.vectorstore.max_marginal_relevance_search(query, k=k)
            mmr_results = [(doc, 1.0) for doc in mmr_docs]
            results_list.append(mmr_results)
            print(f"[检索] MMR 检索: {len(mmr_results)} 个文档")
        except Exception:
            pass

        # 4. RRF 融合
        if len(results_list) > 1:
            fused_results = reciprocal_rank_fusion(results_list)
            docs = [doc for doc, _ in fused_results[:k * 2]]
            print(f"[检索] RRF 融合后: {len(docs)} 个文档")
        elif results_list:
            docs = [doc for doc, _ in results_list[0][:k * 2]]
        else:
            return []

        # 5. 可选：Cross-encoder 重排序
        if self.enable_reranker and self.reranker and self.reranker.model:
            reranked = self.reranker.rerank(query, docs, top_k=k)
            print(f"[检索] Cross-encoder 重排序完成")
            return [doc for doc, _ in reranked]

        return docs[:k]

    def _rerank_search(
        self,
        query: str,
        context: Optional[str],
        k: int
    ) -> List[Document]:
        """重排序检索"""
        # 先用语义检索获取候选
        candidates = self._semantic_search(query, k=k * 3)

        # 使用 Cross-encoder 重排序
        if self.enable_reranker and self.reranker and self.reranker.model:
            reranked = self.reranker.rerank(query, candidates, top_k=k)
            return [doc for doc, _ in reranked]

        # 如果 reranker 不可用，回退到 MMR
        return self.vectorstore.max_marginal_relevance_search(query, k=k)

    def rewrite_query(self, query: str, context: str = None) -> str:
        """
        改写查询以提高检索质量

        Args:
            query: 原始查询
            context: 上下文信息

        Returns:
            改写后的查询
        """
        retriever_config = self.threshold_config.retriever if self.threshold_config else None
        max_words = retriever_config.max_query_words if retriever_config else 20
        context_limit = retriever_config.context_length_limit if retriever_config else 500
        min_keywords = retriever_config.min_query_keywords_for_validation if retriever_config else 2

        if not context:
            template = f"""你是一个查询优化助手。请将以下搜索查询改写为更准确、更具体的查询，以提高检索效果。

原始查询：{{query}}

要求：
1. 保持原始查询的核心意图不变
2. 使用更具体、更准确的关键词
3. 只返回改写后的查询，不要包含任何解释或其他内容
4. 保持查询简洁，不超过{max_words}个词

改写后的查询："""
        else:
            template = f"""你是一个查询优化助手。基于以下信息，将搜索查询改写为更准确、更具体的查询。

原始查询：{{query}}

之前的检索结果（仅供参考，不要完全依赖）：
{{context}}

要求：
1. 保持原始查询的核心意图和主题不变
2. 如果原始查询已经很明确，可以稍作优化但不要偏离原意
3. 不要因为上下文信息而改变查询的核心问题
4. 只返回改写后的查询，不要包含任何解释或其他内容
5. 保持查询简洁，不超过{max_words}个词

改写后的查询："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        if context:
            response = chain.invoke({
                "query": query,
                "context": context[:context_limit]
            })
        else:
            response = chain.invoke({"query": query})

        rewritten = response.content.strip()

        # 企业级最佳实践：使用语义相似度验证改写后的查询，而不是简单的关键词匹配
        if not self._validate_rewritten_query(query, rewritten, min_keywords):
            print(f"[检索] ⚠️ 查询改写可能偏离原意，使用原查询")
            return query

        return rewritten
    
    def _validate_rewritten_query(
        self,
        original_query: str,
        rewritten_query: str,
        min_keywords: int = 2
    ) -> bool:
        """
        验证改写后的查询是否合理（企业级最佳实践）
        
        企业级最佳实践：使用embedding语义相似度验证，而不是关键词匹配。
        理解语义相关性，支持多语言。
        
        Args:
            original_query: 原始查询
            rewritten_query: 改写后的查询
            min_keywords: 最少关键词数（用于回退判断）
            
        Returns:
            如果改写合理，返回True
        """
        # 企业级最佳实践：使用统一的语义相似度工具
        if self.embeddings:
            try:
                from src.agentic_rag.utils.semantic_similarity import SemanticSimilarityCalculator
                
                calculator = SemanticSimilarityCalculator(self.embeddings)
                similarity = calculator.calculate_similarity(
                    text1=original_query,
                    text2=rewritten_query,
                    normalize=True
                )
                
                # 如果语义相似度足够高（>0.5），认为改写合理
                if similarity > 0.5:
                    return True
                # 如果相似度太低（<0.3），认为可能偏离原意
                elif similarity < 0.3:
                    return False
                # 在0.3-0.5之间，使用LLM进一步判断
                
            except Exception as e:
                # embedding计算失败，使用LLM判断
                pass
        
        # 如果没有embedding或相似度在中间范围，使用LLM判断
        if self.llm:
            return self._llm_validate_rewritten_query(original_query, rewritten_query)
        
        # 最后回退：简单的长度检查（非常不可靠，应该避免）
        if len(original_query.split()) < min_keywords:
            return True
        
        # 简单的关键词重叠检查（仅作为最后手段）
        original_words = set(original_query.lower().split())
        rewritten_words = set(rewritten_query.lower().split())
        overlap = len(original_words & rewritten_words)
        overlap_ratio = overlap / len(original_words) if original_words else 0
        
        return overlap_ratio > 0.2  # 至少20%的关键词重叠
    
    def _llm_validate_rewritten_query(
        self,
        original_query: str,
        rewritten_query: str
    ) -> bool:
        """
        使用LLM验证改写后的查询是否合理（企业级最佳实践）
        
        使用统一的LLM判断工具，避免代码重复。
        
        Args:
            original_query: 原始查询
            rewritten_query: 改写后的查询
            
        Returns:
            如果改写合理，返回True
        """
        if not self.llm:
            return True  # 如果没有LLM，默认认为合理
        
        try:
            from src.agentic_rag.utils.llm_judge import LLMJudge
            
            judge = LLMJudge(self.llm)
            return judge.validate_rewritten_query(
                original_query=original_query,
                rewritten_query=rewritten_query,
                confidence_threshold=0.5
            )
            
        except Exception as e:
            # LLM判断失败，返回True（保守策略，允许改写）
            return True
    
    def _extract_meaningful_words(self, text: str) -> List[str]:
        """
        提取有意义的词（企业级最佳实践）
        
        去除停用词和标点，提取核心词汇。
        支持多语言。
        
        Args:
            text: 文本
            
        Returns:
            有意义的词列表
        """
        import re
        
        # 转换为小写
        text_lower = text.lower()
        
        # 提取单词（支持中英文）
        words = re.findall(r'\b\w+\b|[\u4e00-\u9fff]+', text_lower)
        
        # 过滤停用词（通用停用词）
        common_stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'of', 'to', 'in', 'on', 'at', 'for', 'with', 'by', 'from', 'as',
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个'
        }
        
        meaningful_words = [
            word for word in words 
            if len(word) > 1 and word not in common_stopwords
        ]
        
        return meaningful_words

    def _post_process(
        self,
        docs: List[Document],
        query: str
    ) -> List[Document]:
        """
        后处理：去重、过滤等

        Args:
            docs: 文档列表
            query: 查询文本

        Returns:
            处理后的文档列表
        """
        # 去重（基于页面内容）
        seen = set()
        unique_docs = []
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

    def evaluate_retrieval_quality(
        self,
        query: str,
        retrieved_docs: List[Document],
        threshold: Optional[float] = None,
        include_details: bool = False
    ) -> Tuple[float, bool]:
        """
        评估检索质量

        Args:
            query: 查询文本
            retrieved_docs: 检索到的文档
            threshold: 质量阈值
            include_details: 是否返回详细信息

        Returns:
            (质量分数, 是否满足阈值)
        """
        result = self.evaluator.evaluate(
            query=query,
            retrieved_docs=retrieved_docs,
            threshold=threshold,
            include_details=include_details
        )

        # result.final_score 在计算时已经是 Python float 类型
        return result.final_score, result.meets_threshold

