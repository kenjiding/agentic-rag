"""意图识别模块

基于2025-2026年最佳实践：
1. 联合多意图检测与槽位填充（AGIF框架思想）
2. 统一结构生成（UIE框架思想）
3. 知识增强的提示调优
4. 使用LLM进行结构化输出

该模块在接收到用户问题后，首先进行意图识别，以明确用户需求，
从而选择最适合的处理策略和响应方式。
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agentic_rag.threshold_config import ThresholdConfig

PipelineOption = Literal["semantic", "hybrid", "rerank"]

class QueryIntent(BaseModel):
    """查询意图结构（统一信息抽取框架）"""
    
    # 主要意图类型
    intent_type: Literal[
        "factual",      # 事实性查询：询问具体事实、数据、定义
        "comparison",   # 对比查询：比较两个或多个对象/时间点
        "analytical",   # 分析性查询：需要推理、分析、总结
        "procedural",  # 程序性查询：询问如何做某事
        "causal",      # 因果查询：询问原因、结果、影响
        "temporal",    # 时间序列查询：询问变化趋势、历史
        "multi_hop",   # 多跳查询：需要多个步骤推理
        "other"        # 其他类型
    ] = Field(description="查询的主要意图类型")
    
    # 查询复杂度
    complexity: Literal["simple", "moderate", "complex"] = Field(
        description="查询复杂度：simple(<5词), moderate(5-15词), complex(>15词或需要多步推理)"
    )
    
    # 是否需要对比（槽位填充）
    is_comparison: bool = Field(
        description="是否需要对比多个对象、时间点或状态"
    )
    
    # 提取的关键信息（槽位）
    entities: List[str] = Field(
        default_factory=list,
        description="查询中提到的关键实体（人名、地名、时间、组织等）"
    )
    
    # 时间信息
    time_points: List[str] = Field(
        default_factory=list,
        description="查询中提到的具体时间点（年份、日期等）"
    )
    
    # 对比查询拆分后的完整查询列表（如果是对比查询）
    comparison_items: List[str] = Field(
        default_factory=list,
        description="""如果is_comparison=True，此字段应包含拆分后的完整查询列表。

**核心原则**：
对比查询需要多个独立的事实作为输入，因此必须拆分为多个事实性查询。

**通用拆分方法**：
1. 识别对比的维度（时间/对象/属性/状态等）
2. 识别查询的核心信息需求（不是对比本身，而是用于对比的基础事实）
3. 为每个对比项生成独立的事实性查询
4. 移除所有对比性表述（因为对比是在获取事实后进行的）

**关键转换**：
"X和Y的Z【对比性表述】" → ["X的Z是【事实询问】", "Y的Z是【事实询问】"]

对比性表述包括但不限于：
- 变化类：上升/下降/增加/减少/变化/波动/趋势
- 比较类：更大/更小/更好/更差/更快/更慢/差异/区别
- 关系类：相比/对比/比较/versus/vs/哪个/哪种

事实询问包括但不限于：
- 数值查询：是多少/有多少/数量/金额/价格/规模
- 状态查询：是什么/如何/怎样/情况/特点/属性
- 存在查询：有没有/是否/存在/包含

**多样化场景示例**：

1. 时间对比（趋势分析）：
   原："2019-2021年贝索斯财富是上升还是下降？"
   拆分：["2019年贝索斯的财富是多少？", "2020年贝索斯的财富是多少？", "2021年贝索斯的财富是多少？"]

2. 对象对比（横向比较）：
   原："苹果和微软哪个市值更高？"
   拆分：["苹果的市值是多少？", "微软的市值是多少？"]

3. 属性对比（多维比较）：
   原："Python和Java在性能和易用性上有什么区别？"
   拆分：["Python的性能特点是什么？", "Python的易用性特点是什么？", "Java的性能特点是什么？", "Java的易用性特点是什么？"]

4. 状态对比（前后变化）：
   原："疫情前后中国旅游业变化如何？"
   拆分：["疫情前中国旅游业的情况是怎样的？", "疫情后中国旅游业的情况是怎样的？"]

5. 多对象多属性：
   原："特斯拉Model 3和Model Y在价格和续航上的差异？"
   拆分：["特斯拉Model 3的价格是多少？", "特斯拉Model 3的续航是多少？", "特斯拉Model Y的价格是多少？", "特斯拉Model Y的续航是多少？"]

6. 隐含对比（需要推理）：
   原："哪个国家的GDP增长最快？"
   拆分：["各主要国家的GDP增长率分别是多少？"] （单一查询，因为需要先获取多个数据再比较）

7. 复杂嵌套对比：
   原："相比美国，中国和印度在AI领域的投资增长更快吗？"
   拆分：["美国在AI领域的投资增长率是多少？", "中国在AI领域的投资增长率是多少？", "印度在AI领域的投资增长率是多少？"]

8. 非数值对比：
   原："民主党和共和党在气候政策上的立场有何不同？"
   拆分：["民主党在气候政策上的立场是什么？", "共和党在气候政策上的立场是什么？"]

**特殊情况处理**：
- 如果对比项不明确或需要先查询才能确定，可以生成单一的开放性查询
- 如果对比维度过多（>5个），考虑合并相关维度或生成更概括的查询
- 保持查询的自然语言流畅性，避免过于机械的拆分
"""
    )
    
    # 检索策略建议
    recommended_retrieval_strategy: List[PipelineOption] = Field(
        default_factory=list,
        description="推荐的检索策略, 你需要认真详细的分析, 然后给出单个策略, 或者多个策略组合的方式, 如: semantic+hybrid, semantic+rerank, hybrid+rerank, semantic+hybrid+rerank等"
    )
    
    # 检索数量建议
    recommended_k: int = Field(
        description="推荐的检索文档数量（根据意图类型和复杂度调整, 当intent_type为comparison时, 会拆分成多个单独的查询, k值需要根据对比项的数量来调整）"
    )
    
    # 是否需要多轮检索
    needs_multi_round_retrieval: bool = Field(
        description="是否需要多轮检索（对于复杂查询或多跳查询）"
    )
    
    # 置信度
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="意图识别的置信度（0-1）"
    )
    
    # 说明
    reasoning: str = Field(
        description="意图识别的推理过程（使用与查询相同的语言）"
    )


class IntentClassifier:
    """意图分类器
    
    使用LLM进行意图识别，支持多语言和复杂查询。
    结合联合多意图检测与槽位填充的思想，同时识别意图和提取关键信息。
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        threshold_config: Optional[ThresholdConfig] = None
    ):
        """
        初始化意图分类器
        
        Args:
            llm: LLM实例（如果为None，使用默认配置创建）
            threshold_config: 阈值配置
        """
        self.threshold_config = threshold_config
        
        # 使用配置的LLM温度（如果未提供）
        if llm is None:
            # 意图识别需要较低温度以保证稳定性
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        
        self.llm = llm
    
    def improve_query(self, query: str) -> str:
        """
        对查询进行优化
        
        Args:
            query: 用户查询
            
        Returns:
           优化后的查询
        """
        template = """你是一个专业的查询意图分析器。请分析以下查询，识别其意图类型、复杂度和关键信息。
        
        Args:
            query: 用户查询
            
        Returns:
           优化后的查询
        """
        pass
    def classify(self, query: str) -> QueryIntent:
        """
        对查询进行意图分类
        
        基于2025-2026年最佳实践：
        1. 联合多意图检测与槽位填充：同时识别意图和提取关键信息
        2. 统一结构生成：使用结构化输出确保一致性
        3. 通用性设计：使用通用的方法识别意图，不依赖特定领域知识
        
        Args:
            query: 用户查询
            
        Returns:
            查询意图结构
        """
        template = """你是一个专业的查询意图分析器。请分析以下查询，识别其意图类型、复杂度和关键信息。

**分析要求**：
1. **联合意图检测与槽位填充**：同时识别意图类型和提取关键信息（实体、时间、对比对象等）
2. **统一结构生成**：严格按照指定的JSON结构输出，确保一致性
3. **通用性**：使用通用的方法识别意图，适用于任何领域和场景

**意图类型说明**：
- factual: 事实性查询（询问具体事实、数据、定义）
- comparison: 对比查询（比较两个或多个对象/时间点/状态）
- analytical: 分析性查询（需要推理、分析、总结）
- procedural: 程序性查询（询问如何做某事）
- causal: 因果查询（询问原因、结果、影响）
- temporal: 时间序列查询（询问变化趋势、历史）
- multi_hop: 多跳查询（需要多个步骤推理）
- other: 其他类型

**对比查询的通用处理方法**：

对比查询的本质是：用户想要基于多个独立的事实进行比较分析。
因此，你的任务是将对比查询拆分为获取这些基础事实的查询。

**拆分的通用步骤**：
1. **识别对比维度**：时间（2019 vs 2020）、对象（苹果 vs 微软）、属性（价格 vs 性能）、状态（前 vs 后）等
2. **识别核心信息需求**：用户实际需要什么信息？（财富数据、市值、性能指标等）
3. **移除对比性表述**：删除"上升/下降/更大/更小/差异/对比"等词汇
4. **转换为事实性询问**：将"X和Y谁更Z"转换为"X的Z是多少？Y的Z是多少？"
5. **确保查询独立性**：每个拆分查询应该能够独立回答一个具体事实

**关键原则**：
- 拆分后的查询应该是事实性的，而不是对比性的
- 每个查询应该针对一个具体的对比项（时间点/对象/属性等）
- 保留查询的核心主题和关键信息
- 确保查询的自然语言流畅性

**通用转换模式**：
- "X和Y的Z【对比词】" → ["X的Z是【事实询问】？", "Y的Z是【事实询问】？"]
- "【时间1】和【时间2】的X如何变化" → ["【时间1】的X是多少？", "【时间2】的X是多少？"]
- "哪个/哪种X更【比较词】" → ["各X的【相关属性】分别是什么？"]
- "X相比Y【对比表述】" → ["X的【属性】是什么？", "Y的【属性】是什么？"]

**检索策略建议**：
- semantic: 标准语义检索（适合大多数查询）
- hybrid: 混合检索，使用MMR增加多样性（适合对比查询、需要多个不同信息片段的查询）
- rerank: 重排序检索（适合需要精确匹配的查询）
- 也可以根据实际情况深思熟虑后选择最合适的策略, 可以单个策略, 也可以多个策略组合的方式进行检索

**检索数量建议**：
- simple: 3-5
- moderate: 5-7
- complex/comparison: 8-12
- multi_hop: 10-15

查询：{query}

请严格按照QueryIntent结构输出JSON，确保：
1. 准确识别意图类型（基于查询的实际意图，不依赖特定领域知识）
2. 正确评估复杂度（基于查询长度和推理步骤）
3. 提取所有关键信息（实体、时间等，使用通用的识别方法）
4. **对于对比查询，必须将对比性问题转换为事实性问题**（适用于任何对比场景）
5. 提供合理的检索策略和数量建议（基于意图类型和复杂度）
6. 给出置信度和推理过程

输出JSON："""

        prompt = ChatPromptTemplate.from_template(template)
        
        # 使用结构化输出确保一致性
        structured_llm = self.llm.with_structured_output(QueryIntent, method="json_schema")
        chain = prompt | structured_llm
        
        try:
            intent = chain.invoke({"query": query})
            return intent
        except Exception as e:
            print(f"[意图识别] 错误: {e}")
            # 回退到默认意图
            return self._fallback_intent(query)
    
    def _fallback_intent(self, query: str) -> QueryIntent:
        """
        回退意图（当LLM识别失败时使用）
        
        使用通用的启发式规则，不依赖特定领域知识
        
        Args:
            query: 用户查询
            
        Returns:
            默认意图结构
        """
        import re
        
        # 复杂度检测：基于查询长度
        words = len(query.split())
        complexity = "simple" if words < 5 else ("moderate" if words < 15 else "complex")
        
        # 对比检测：通用的对比模式（多语言）
        comparison_patterns = [
            # 中文对比词
            r'\b(对比|比较|相比|变化|上升|下降|增加|减少|差异|区别|哪个|哪种|更|较)\b',
            # 英文对比词
            r'\b(compare|comparison|versus|vs|compared to|difference|change|increase|decrease|which|better|worse|more|less)\b',
            # 符号
            r'\b(vs\.?|versus)\b'
        ]
        is_comparison = any(re.search(pattern, query, re.IGNORECASE) for pattern in comparison_patterns)
        
        # 时间点检测：通用的时间格式
        time_patterns = [
            r'\b\d{4}\b',  # 年份
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 日期
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # 英文日期
            r'\b(一月|二月|三月|四月|五月|六月|七月|八月|九月|十月|十一月|十二月)\b',  # 中文月份
        ]
        time_points = []
        for pattern in time_patterns:
            time_points.extend(re.findall(pattern, query, re.IGNORECASE))
        time_points = list(set([str(tp).strip() for tp in time_points]))
        
        # 实体检测：通用的实体模式
        entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 大写开头的词（英文）
            r'"[^"]+"',  # 双引号内的内容
            r"'[^']+'",  # 单引号内的内容
        ]
        entities = []
        for pattern in entity_patterns:
            entities.extend(re.findall(pattern, query))
        entities = list(set([e.strip('"\'') for e in entities]))[:10]
        
        # 如果有多个时间点，更可能是对比查询
        if len(time_points) >= 2:
            is_comparison = True
        
        # 尝试生成简单的拆分查询（如果是对比查询）
        comparison_items = []
        if is_comparison:
            # 通用的对比词移除
            comparison_words = [
                # 中文
                '对比', '比较', '相比', '变化', '上升', '下降', '增加', '减少', 
                '差异', '区别', '和', '与', '还是', '哪个', '哪种', '更', '较',
                # 英文
                'compare', 'comparison', 'versus', 'vs', 'compared to', 'difference',
                'change', 'increase', 'decrease', 'and', 'or', 'which', 'better',
                'worse', 'more', 'less', 'than'
            ]
            
            clean_query = query
            for word in comparison_words:
                clean_query = re.sub(rf'\b{re.escape(word)}\b', ' ', clean_query, flags=re.IGNORECASE)
            clean_query = ' '.join(clean_query.split())  # 清理多余空格
            
            # 如果检测到时间点，为每个时间点生成查询
            if time_points:
                base_query = clean_query
                for tp in time_points:
                    base_query = base_query.replace(tp, '').strip()
                
                for tp in time_points:
                    if base_query:
                        comparison_items.append(f"{tp}{base_query}是多少？")
            # 如果没有明显的时间点，但检测到对比词汇，生成通用查询
            elif clean_query:
                comparison_items.append(f"{clean_query}的具体情况是什么？")
        
        return QueryIntent(
            intent_type="comparison" if is_comparison else "factual",
            complexity=complexity,
            is_comparison=is_comparison,
            entities=entities,
            time_points=time_points,
            comparison_items=comparison_items,
            recommended_retrieval_strategy=["hybrid"] if is_comparison else ["semantic"],
            recommended_k=10 if is_comparison else (5 if complexity == "simple" else 7),
            needs_multi_round_retrieval=complexity == "complex",
            confidence=0.5,
            reasoning="回退模式：使用通用启发式规则。对于对比查询，已尝试将对比性问题转换为事实性问题。"
        )