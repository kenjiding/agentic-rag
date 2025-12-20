"""查询优化模块

基于2025-2026年最佳实践：
1. Query2Doc: 使用LLM生成伪文档扩展查询
2. HyDE (Hypothetical Document Embeddings): 生成假设性答案
3. Step-back Prompting: 生成更抽象的查询
4. Multi-Query: 生成多个不同角度的查询
5. Query Decomposition: 复杂查询分解
6. Entity/Keyword Enhancement: 实体和关键词增强

该模块根据意图识别结果，对原始查询进行优化，提升检索效率和准确性。
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .intent_classifier import QueryIntent

class OptimizedQuery(BaseModel):
    """优化后的查询结构"""
    
    # 原始查询
    original_query: str = Field(description="原始用户查询")
    
    # 主查询（优化后的核心查询）
    primary_query: str = Field(
        description="优化后的主查询，添加了关键信息、消除了歧义、更适合检索"
    )
    
    # 扩展查询（多角度查询）
    expanded_queries: List[str] = Field(
        default_factory=list,
        description="从不同角度表述的查询，用于提升召回率"
    )
    
    # 关键词列表
    keywords: List[str] = Field(
        default_factory=list,
        description="提取的关键词，用于混合检索或过滤"
    )
    
    # 假设性答案（HyDE）
    hypothetical_answer: Optional[str] = Field(
        default=None,
        description="假设性的理想答案，用于基于答案的检索"
    )
    
    # 抽象查询（Step-back）
    abstract_query: Optional[str] = Field(
        default=None,
        description="更抽象、更高层次的查询，用于获取背景知识"
    )
    
    # 优化说明
    optimization_notes: str = Field(
        description="优化过程的说明"
    )


class QueryOptimizer:
    """查询优化器
    
    根据意图识别结果，使用多种策略优化查询，提升检索质量。
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        enable_hyde: bool = False,
        enable_stepback: bool = False,
        max_expanded_queries: int = 3
    ):
        """
        初始化查询优化器
        
        Args:
            llm: LLM实例
            enable_hyde: 是否启用HyDE（假设性文档嵌入）
            enable_stepback: 是否启用Step-back（抽象查询）
            max_expanded_queries: 最大扩展查询数量
        """
        if llm is None:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        self.llm = llm
        self.enable_hyde = enable_hyde
        self.enable_stepback = enable_stepback
        self.max_expanded_queries = max_expanded_queries
    
    def optimize(
        self, 
        query: str, 
        intent: QueryIntent,
        context: Optional[Dict[str, Any]] = None
    ) -> OptimizedQuery:
        """
        优化查询
        
        Args:
            query: 原始查询
            intent: 意图识别结果
            context: 额外上下文（如之前的对话、用户偏好等）
            
        Returns:
            优化后的查询结构
        """
        # 根据意图类型选择优化策略
        if intent["is_comparison"]:
            return self._optimize_comparison_query(query, intent, context)
        elif intent["intent_type"] == "multi_hop":
            return self._optimize_multihop_query(query, intent, context)
        elif intent["intent_type"] in ["analytical", "causal"]:
            return self._optimize_analytical_query(query, intent, context)
        elif intent["intent_type"] == "complex":
            return self._optimize_complex_query(query, intent, context)
        else:
            return self._optimize_simple_query(query, intent, context)
    
    def _optimize_simple_query(
        self, 
        query: str, 
        intent: QueryIntent,
        context: Optional[Dict[str, Any]]
    ) -> OptimizedQuery:
        """优化简单查询"""
        
        template = """你是一个专业的查询优化专家。请优化以下查询，使其更适合文档检索。

**原始查询**: {query}

**意图信息**:
- 类型: {intent_type}
- 复杂度: {complexity}
- 关键实体: {entities}
- 时间点: {time_points}

**优化目标**:
1. 明确化：消除歧义，添加必要的限定词
2. 丰富化：添加同义词、相关概念
3. 精确化：提取核心关键词
4. 多样化：生成不同角度的查询表述

**优化策略**:
- 添加领域上下文（如果查询过于简短）
- 展开缩写和简称
- 添加时间、地点等关键限定
- 保持查询的核心意图不变

请输出:
1. primary_query: 一个优化后的主查询（更明确、更完整）
2. expanded_queries: {max_queries}个不同角度的查询（同义表达、不同粒度）
3. keywords: 5-10个核心关键词
4. optimization_notes: 简要说明优化思路

以JSON格式输出。"""

        prompt = ChatPromptTemplate.from_template(template)
        
        structured_llm = self.llm.with_structured_output(
            OptimizedQuery,
            method="json_schema"
        )
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({
                "query": query,
                "intent_type": intent["intent_type"],
                "complexity": intent["complexity"],
                "entities": ", ".join(intent["entities"]) if intent["entities"] else "无",
                "time_points": ", ".join(intent["time_points"]) if intent["time_points"] else "无",
                "max_queries": self.max_expanded_queries
            })
            
            # 可选：添加HyDE
            if self.enable_hyde:
                result.hypothetical_answer = self._generate_hypothetical_answer(query, intent)
            
            # 可选：添加Step-back
            if self.enable_stepback:
                result.abstract_query = self._generate_abstract_query(query, intent)
            
            result.original_query = query
            return result
            
        except Exception as e:
            print(f"[查询优化] 错误: {e}")
            return self._fallback_optimization(query, intent)
    
    def _optimize_comparison_query(
        self, 
        query: str, 
        intent: QueryIntent,
        context: Optional[Dict[str, Any]]
    ) -> OptimizedQuery:
        """优化对比查询"""
        
        template = """你是一个专业的查询优化专家。请优化以下对比查询，使其更适合文档检索。

**原始查询**: {query}

**对比信息**:
- 对比项: {comparison_items}
- 关键实体: {entities}
- 时间点: {time_points}

**对比查询优化策略**:
1. **主查询设计**: 
   - 不要直接问对比问题（如"哪个更大"）
   - 而是转换为获取对比所需的事实信息
   - 例如："2019年和2020年的GDP对比" → "2019年GDP数据, 2020年GDP数据"

2. **扩展查询设计**:
   - 为每个对比项生成独立的事实查询
   - 添加必要的上下文和限定词
   - 确保查询能够检索到可比较的信息

3. **关键词提取**:
   - 提取对比的核心维度（价格、性能、规模等）
   - 提取所有对比对象的名称
   - 提取时间、地点等限定词

**示例**:
原查询: "2019和2020年苹果营收对比"
primary_query: "苹果公司2019年和2020年的年度营收数据"
expanded_queries: [
    "苹果公司2019年财报营收",
    "苹果公司2020年财报营收",
    "苹果公司2019-2020年收入变化"
]
keywords: ["苹果", "营收", "2019", "2020", "财报", "收入"]

请输出:
1. primary_query: 一个优化后的主查询（聚焦获取对比所需的事实）
2. expanded_queries: {max_queries}个查询（最好包含针对每个对比项的独立查询）
3. keywords: 核心关键词（包含所有对比维度和对象）
4. optimization_notes: 优化说明

以JSON格式输出。"""

        prompt = ChatPromptTemplate.from_template(template)
        
        structured_llm = self.llm.with_structured_output(
            OptimizedQuery,
            method="json_schema"
        )
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({
                "query": query,
                "comparison_items": ", ".join(intent["comparison_items"]) if intent["comparison_items"] else "未明确拆分",
                "entities": ", ".join(intent["entities"]) if intent["entities"] else "无",
                "time_points": ", ".join(intent["time_points"]) if intent["time_points"] else "无",
                "max_queries": self.max_expanded_queries
            })
            
            result.original_query = query
            return result
            
        except Exception as e:
            print(f"[查询优化-对比] 错误: {e}")
            return self._fallback_optimization(query, intent)
    
    def _optimize_multihop_query(
        self, 
        query: str, 
        intent: QueryIntent,
        context: Optional[Dict[str, Any]]
    ) -> OptimizedQuery:
        """优化多跳查询"""
        
        template = """你是一个专业的查询优化专家。请优化以下多跳查询，使其更适合文档检索。

**原始查询**: {query}

**多跳查询特点**:
- 需要多个步骤的推理
- 需要先获取中间信息，再进行下一步查询
- 信息之间有依赖关系

**优化策略**:
1. **主查询**: 设计一个能够覆盖查询主题的综合性查询
2. **查询分解**: 将多跳查询分解为多个子查询
3. **关键路径**: 识别推理的关键步骤和所需信息

**示例**:
原查询: "特斯拉CEO的母校在哪个城市？"
primary_query: "特斯拉CEO教育背景和毕业院校"
expanded_queries: [
    "特斯拉现任CEO是谁",
    "埃隆·马斯克教育背景",
    "宾夕法尼亚大学所在城市"
]

请输出:
1. primary_query: 覆盖整个查询主题的综合查询
2. expanded_queries: 分解后的子查询（按推理顺序）
3. keywords: 核心关键词
4. optimization_notes: 分解思路说明

以JSON格式输出。"""

        prompt = ChatPromptTemplate.from_template(template)
        
        structured_llm = self.llm.with_structured_output(
            OptimizedQuery,
            method="json_schema"
        )
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({
                "query": query,
                "max_queries": self.max_expanded_queries
            })
            
            # 多跳查询特别适合Step-back
            if self.enable_stepback:
                result.abstract_query = self._generate_abstract_query(query, intent)
            
            result.original_query = query
            return result
            
        except Exception as e:
            print(f"[查询优化-多跳] 错误: {e}")
            return self._fallback_optimization(query, intent)
    
    def _optimize_analytical_query(
        self, 
        query: str, 
        intent: QueryIntent,
        context: Optional[Dict[str, Any]]
    ) -> OptimizedQuery:
        """优化分析性查询"""
        
        template = """你是一个专业的查询优化专家。请优化以下分析性查询，使其更适合文档检索。

**原始查询**: {query}

**分析性查询特点**:
- 需要综合多个信息源
- 需要推理、总结、分析
- 通常涉及因果关系、趋势、影响等

**优化策略**:
1. **主查询**: 聚焦核心分析对象和维度
2. **扩展查询**: 从不同角度获取分析所需的背景信息
   - 事实数据
   - 专家观点
   - 相关案例
   - 理论框架
3. **Step-back**: 生成更抽象的背景查询

**示例**:
原查询: "人工智能对就业市场的影响是什么？"
primary_query: "人工智能对就业市场的影响 研究报告 数据"
expanded_queries: [
    "人工智能导致的工作岗位变化",
    "人工智能创造的新职业类型",
    "各行业受人工智能影响的程度"
]
abstract_query: "技术进步对劳动力市场的历史影响"

请输出优化后的查询结构，以JSON格式输出。"""

        prompt = ChatPromptTemplate.from_template(template)
        
        structured_llm = self.llm.with_structured_output(
            OptimizedQuery,
            method="json_schema"
        )
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({
                "query": query,
                "max_queries": self.max_expanded_queries
            })
            
            # 分析性查询特别适合HyDE和Step-back
            if self.enable_hyde:
                result.hypothetical_answer = self._generate_hypothetical_answer(query, intent)
            
            if self.enable_stepback:
                result.abstract_query = self._generate_abstract_query(query, intent)
            
            result.original_query = query
            return result
            
        except Exception as e:
            print(f"[查询优化-分析] 错误: {e}")
            return self._fallback_optimization(query, intent)
    
    def _optimize_complex_query(
        self, 
        query: str, 
        intent: QueryIntent,
        context: Optional[Dict[str, Any]]
    ) -> OptimizedQuery:
        """优化复杂查询"""
        
        # 复杂查询使用综合策略
        template = """你是一个专业的查询优化专家。请优化以下复杂查询。

**原始查询**: {query}

**意图信息**:
- 类型: {intent_type}
- 实体: {entities}
- 时间: {time_points}

**复杂查询优化策略**:
1. 简化和聚焦：提取核心问题
2. 分解和扩展：从多个角度切入
3. 添加上下文：补充必要的背景信息
4. 关键词提取：识别检索的关键术语

请输出:
1. primary_query: 简化后的核心查询
2. expanded_queries: {max_queries}个扩展查询（不同角度和粒度）
3. keywords: 核心关键词列表
4. optimization_notes: 优化说明

以JSON格式输出。"""

        prompt = ChatPromptTemplate.from_template(template)
        
        structured_llm = self.llm.with_structured_output(
            OptimizedQuery,
            method="json_schema"
        )
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({
                "query": query,
                "intent_type": intent["intent_type"],
                "entities": ", ".join(intent["entities"]) if intent["entities"] else "无",
                "time_points": ", ".join(intent["time_points"]) if intent["time_points"] else "无",
                "max_queries": self.max_expanded_queries
            })
            
            if self.enable_hyde:
                result.hypothetical_answer = self._generate_hypothetical_answer(query, intent)
            
            if self.enable_stepback:
                result.abstract_query = self._generate_abstract_query(query, intent)
            
            result.original_query = query
            return result
            
        except Exception as e:
            print(f"[查询优化-复杂] 错误: {e}")
            return self._fallback_optimization(query, intent)
    
    def _generate_hypothetical_answer(
        self, 
        query: str, 
        intent: QueryIntent
    ) -> str:
        """生成假设性答案（HyDE）"""
        
        template = """假设你是一个专家，请为以下问题生成一个理想的、详细的答案。
这个答案将用于检索相似的文档，所以请包含相关的术语、概念和细节。

问题: {query}

请生成一个假设性的理想答案（2-3句话）："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"query": query})
            return response.content
        except:
            return ""
    
    def _generate_abstract_query(
        self, 
        query: str, 
        intent: QueryIntent
    ) -> str:
        """生成抽象查询（Step-back）"""
        
        template = """请将以下具体查询转换为一个更抽象、更高层次的查询，
以便检索相关的背景知识和理论框架。

具体查询: {query}

抽象查询（一句话）："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"query": query})
            return response.content.strip()
        except:
            return ""
    
    def _fallback_optimization(
        self, 
        query: str, 
        intent: QueryIntent
    ) -> OptimizedQuery:
        """回退优化（当LLM失败时使用）"""
        
        # 简单的启发式优化
        import re
        
        # 提取关键词
        keywords = []
        if intent["entities"]:
            keywords.extend(intent["entities"])
        if intent["time_points"]:
            keywords.extend(intent["time_points"])
        
        # 移除停用词，提取其他关键词
        stopwords = {'的', '了', '是', '在', '有', '和', '与', 'the', 'a', 'an', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\w+', query.lower())
        keywords.extend([w for w in words if w not in stopwords and len(w) > 1])
        keywords = list(set(keywords))[:10]
        
        # 生成扩展查询
        expanded = [query]
        if intent["entities"]:
            for entity in intent["entities"][:2]:
                expanded.append(f"{entity} {query.replace(entity, '').strip()}")
        
        return OptimizedQuery(
            original_query=query,
            primary_query=query,
            expanded_queries=expanded[:self.max_expanded_queries],
            keywords=keywords,
            optimization_notes="回退模式：使用简单的启发式规则"
        )

