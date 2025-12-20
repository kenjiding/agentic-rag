"""智能检索器实现"""
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.agentic_rag.evaluation_config import RetrievalQualityConfig
from src.agentic_rag.evaluators import RetrievalQualityEvaluator
from src.agentic_rag.threshold_config import ThresholdConfig, RetrieverThresholds
from src.agentic_rag.intent_analyse import PipelineOption

class IntelligentRetriever:
    """智能检索器"""
    
    def __init__(
        self,
        vectorstore: Chroma,
        llm: Optional[ChatOpenAI] = None,
        default_strategy: str = "semantic",
        evaluation_config: Optional[RetrievalQualityConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None
    ):
        """
        初始化检索器
        
        Args:
            vectorstore: 向量数据库
            llm: 用于查询改写的 LLM
            default_strategy: 默认检索策略
            evaluation_config: 评估配置
            threshold_config: 阈值配置
        """
        self.vectorstore = vectorstore
        self.threshold_config = threshold_config
        
        # 获取检索器阈值配置
        retriever_thresholds = threshold_config.retriever if threshold_config else RetrieverThresholds()
        
        # 使用配置的 LLM 温度
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=retriever_thresholds.llm_temperature)
        self.default_strategy = default_strategy
        self.embeddings = vectorstore._embedding_function
        
        # 初始化评估器（传递 threshold_config 以便使用评估阈值）
        self.evaluation_config = evaluation_config or RetrievalQualityConfig()
        self.evaluator = RetrievalQualityEvaluator(
            embeddings=self.embeddings,
            config=self.evaluation_config,
            threshold_config=threshold_config
        )
        
    def retrieve(
        self,
        query: str,
        strategy: Optional[str] = None,
        strategies: Optional[List[PipelineOption]] = None,
        context: Optional[str] = None,
        k: int = 3,
        rewrite_query: bool = False,
        split_queries: Optional[List[str]] = None
    ) -> List[Document]:
        """
        检索文档
        
        Args:
            query: 查询文本
            strategy: 单个检索策略（向后兼容，如果strategies不为None则忽略此参数）
            strategies: 检索策略列表（支持组合策略）
            context: 上下文信息（用于查询改写或重排序）
            k: 返回文档数量
            rewrite_query: 是否改写查询
            split_queries: 拆分后的查询列表（对于对比查询）
            
        Returns:
            检索到的文档列表
        """
        # 如果是对比查询且有拆分查询，使用拆分查询进行检索
        if split_queries and len(split_queries) > 1:
            return self._retrieve_with_split_queries(split_queries, strategies or [strategy or self.default_strategy], k)
        
        # 确定使用的策略
        if strategies:
            # 使用策略列表（组合策略）
            strategies_to_use = strategies
        else:
            # 使用单个策略（向后兼容）
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
        split_queries: List[str], 
        strategies: List[PipelineOption], 
        k: int
    ) -> List[Document]:
        """
        使用拆分查询进行检索（用于对比查询）
        
        Args:
            split_queries: 拆分后的查询列表
            strategies: 检索策略列表
            k: 每个查询的检索数量
            
        Returns:
            合并后的文档列表
        """
        all_docs = []
        
        print(f"[检索] 对比查询：使用 {len(split_queries)} 个拆分查询，策略: {strategies}")
        
        for split_query in split_queries:
            print(f"[检索] 检索拆分查询: {split_query}")
            # 对每个拆分查询使用组合策略
            for strategy in strategies:
                docs = self._retrieve_with_strategy(split_query, strategy, None, k)
                all_docs.extend(docs)
        
        # 后处理（去重等）
        all_docs = self._post_process(all_docs, split_queries[0] if split_queries else "")
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
            return self._hybrid_search(query, k)
        elif strategy == "rerank":
            return self._rerank_search(query, context, k)
        else:
            # 默认使用语义检索（包括 "semantic" 和其他未知策略）
            return self._semantic_search(query, k)
    
    def _semantic_search(self, query: str, k: int) -> List[Document]:
        """语义检索"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """混合检索（使用 MMR 增加多样性）"""
        return self.vectorstore.max_marginal_relevance_search(query, k=k)
    
    def _rerank_search(
        self,
        query: str,
        context: Optional[str],
        k: int
    ) -> List[Document]:
        """重排序检索（使用上下文改写查询后执行语义检索）"""
        if context:
            query = self.rewrite_query(query, context)
        return self._semantic_search(query, k)
    
    def rewrite_query(self, query: str, context: str = None) -> str:
        """
        改写查询以提高检索质量
        
        Args:
            query: 原始查询
            context: 上下文信息（可选，之前的检索结果或对话历史）
            
        Returns:
            改写后的查询
        """
        # 获取配置参数
        retriever_config = self.threshold_config.retriever if self.threshold_config else None
        max_words = retriever_config.max_query_words if retriever_config else 20
        context_limit = retriever_config.context_length_limit if retriever_config else 500
        min_keywords = retriever_config.min_query_keywords_for_validation if retriever_config else 2
        
        # 如果没有上下文或上下文为空，使用简化版改写
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
                "context": context[:context_limit]  # 限制上下文长度，避免误导
            })
        else:
            response = chain.invoke({
                "query": query
            })
        
        rewritten = response.content.strip()
        
        # 验证改写后的查询是否合理（至少包含原查询的一些关键词）
        query_keywords = set(query.lower().split())
        rewritten_keywords = set(rewritten.lower().split())
        
        # 如果改写后的查询完全偏离，使用原查询
        if len(query_keywords & rewritten_keywords) == 0 and len(query_keywords) > min_keywords:
            print(f"[检索] ⚠️ 查询改写可能偏离原意，使用原查询")
            return query
        
        return rewritten
    
    def _post_process(
        self,
        docs: List[Document],
        query: str
    ) -> List[Document]:
        """
        后处理：去重、过滤等
        
        Args:
            docs: 文档列表
            query: 查询文本（保留用于将来可能的过滤功能）
            
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
        评估检索质量（使用评估器）
        
        Args:
            query: 查询文本
            retrieved_docs: 检索到的文档
            threshold: 质量阈值（如果为None，使用配置中的动态阈值）
            include_details: 是否返回详细信息
            
        Returns:
            (质量分数, 是否满足阈值)
            如果 include_details=True，可以通过 evaluator 获取详细信息
        """
        result = self.evaluator.evaluate(
            query=query,
            retrieved_docs=retrieved_docs,
            threshold=threshold,
            include_details=include_details
        )
        
        return result.final_score, result.meets_threshold
