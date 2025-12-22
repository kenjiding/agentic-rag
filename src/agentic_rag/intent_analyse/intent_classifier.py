"""意图识别模块

基于2025-2026年最佳实践：
1. 联合多意图检测与槽位填充（AGIF框架思想）
2. 统一结构生成（UIE框架思想）
3. 知识增强的提示调优
4. 使用LLM进行结构化输出
5. 通用查询分解机制（自动判断是否需要分解）

该模块在接收到用户问题后，首先进行意图识别，以明确用户需求，
从而选择最适合的处理策略和响应方式。对于复杂查询，自动判断
是否需要分解为多个子查询以提高检索效果。
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agentic_rag.threshold_config import ThresholdConfig

PipelineOption = Literal["semantic", "hybrid", "rerank"]

# 查询分解类型
DecompositionType = Literal[
    "comparison",     # 对比分解：按对比项分解（A vs B → 查A + 查B）
    "multi_hop",      # 多跳分解：按推理步骤分解，有顺序依赖
    "dimensional",    # 多维分解：按分析维度分解（原因分析 → 技术+商业+市场）
    "temporal",       # 时间分解：按时间段分解（10年发展 → 多个时间段）
    "causal_chain",   # 因果链分解：按因果关系分解
    "information_needs"  # 信息需求分解：按独立信息需求点分解
]


class SubQuery(BaseModel):
    """子查询结构 - 包含查询文本、检索策略和执行依赖

    2025 最佳实践：
    1. 每个子查询有独立的检索策略，避免复杂策略应用于简单查询
    2. 支持执行顺序和依赖关系，用于多跳查询等场景
    """

    query: str = Field(description="子查询文本")

    purpose: str = Field(
        default="",
        description="该子查询的目的说明（如：获取基础事实、验证假设、补充细节等）"
    )

    recommended_strategy: List[PipelineOption] = Field(
        default_factory=lambda: ["semantic"],
        description="该子查询的推荐检索策略（根据子查询的特点独立选择）"
    )

    recommended_k: int = Field(
        default=5,
        description="该子查询的推荐检索数量"
    )

    order: int = Field(
        default=0,
        description="执行顺序：0表示可并行执行，>0表示需按顺序执行（数字越小越先执行）"
    )

    depends_on: List[int] = Field(
        default_factory=list,
        description="依赖的子查询索引列表（用于多跳查询，表示需要先执行哪些查询）"
    )


class QueryIntent(BaseModel):
    """查询意图结构（统一信息抽取框架）

    支持通用的查询分解机制，能够根据查询复杂度自动判断是否需要分解，
    并根据不同的分解类型生成相应的子查询。
    """

    # 主要意图类型
    intent_type: Literal[
        "factual",      # 事实性查询：询问具体事实、数据、定义
        "comparison",   # 对比查询：比较两个或多个对象/时间点
        "analytical",   # 分析性查询：需要推理、分析、总结
        "procedural",   # 程序性查询：询问如何做某事
        "causal",       # 因果查询：询问原因、结果、影响
        "temporal",     # 时间序列查询：询问变化趋势、历史
        "multi_hop",    # 多跳查询：需要多个步骤推理
        "other"         # 其他类型
    ] = Field(description="查询的主要意图类型")

    # 查询复杂度
    complexity: Literal["simple", "moderate", "complex"] = Field(
        description="查询复杂度：simple(单一信息点), moderate(2-3个信息点), complex(多信息点或需要多步推理)"
    )

    # ==================== 通用查询分解机制 ====================

    # 是否需要查询分解（核心判断字段）
    needs_decomposition: bool = Field(
        description="""是否需要将原查询分解为多个子查询。

**需要分解的信号**（满足任一条件即可）：
1. 复杂度为 complex
2. 包含多个独立的信息需求点（如"介绍X的原理、应用和前景"）
3. 需要对比多个对象/时间点（comparison）
4. 需要多步推理（multi_hop）
5. 需要从多个维度分析（analytical）
6. 需要按时间段查询（temporal跨度大）
7. 涉及因果链条（causal有多个层次）

**不需要分解的信号**：
1. 简单的单一事实查询
2. 查询已经足够具体明确
3. procedural类型查询（步骤是内容本身，不是检索单位）
4. 分解会导致上下文信息丢失
"""
    )

    # 分解类型（当 needs_decomposition=True 时必填）
    decomposition_type: Optional[Literal[
        "comparison",        # 对比分解：按对比项分解（A vs B → 查A + 查B）
        "multi_hop",         # 多跳分解：按推理步骤分解，有顺序依赖
        "dimensional",       # 多维分解：按分析维度分解（分析原因 → 技术+商业+市场）
        "temporal",          # 时间分解：按时间段分解（10年发展 → 多个时间段）
        "causal_chain",      # 因果链分解：按因果关系分解（为什么 → 直接+间接+根本原因）
        "information_needs"  # 信息需求分解：按独立信息需求点分解
    ]] = Field(
        default=None,
        description="分解类型，当 needs_decomposition=True 时必须指定"
    )

    # 分解原因说明
    decomposition_reason: str = Field(
        default="",
        description="为什么需要分解的简要说明（如果 needs_decomposition=True）"
    )

    # 通用子查询列表
    sub_queries: List[SubQuery] = Field(
        default_factory=list,
        description="""分解后的子查询列表。每个子查询包含：
- query: 子查询文本
- purpose: 该子查询的目的
- recommended_strategy: 推荐的检索策略
- recommended_k: 推荐的检索数量
- order: 执行顺序（0=可并行，>0=按顺序）
- depends_on: 依赖的子查询索引

**分解类型与子查询特点**：

1. comparison（对比分解）- order=0，可并行执行
   - 将对比查询拆分为独立的事实查询
   - 移除对比性表述，转换为事实询问

2. multi_hop（多跳分解）- order>0，有顺序依赖
   - 按推理步骤分解，后续步骤依赖前序结果
   - 使用 depends_on 指明依赖关系

3. dimensional（多维分解）- order=0，可并行执行
   - 按分析维度拆分（技术/商业/市场等）
   - 每个维度独立查询

4. temporal（时间分解）- order=0，可并行执行
   - 按时间段拆分
   - 每个时间段独立查询

5. causal_chain（因果链分解）- 可能有顺序依赖
   - 直接原因、间接原因、根本原因
   - 可能需要按因果层次查询

6. information_needs（信息需求分解）- order=0，可并行执行
   - 按独立信息需求点拆分
   - 每个需求点独立查询
"""
    )

    # ==================== 槽位填充信息 ====================

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

    # ==================== 检索策略建议 ====================

    # 检索策略建议（针对原查询或不分解时使用）
    recommended_retrieval_strategy: List[PipelineOption] = Field(
        default_factory=list,
        description="推荐的检索策略（当不分解时使用，分解时参考各子查询的策略）"
    )

    # 检索数量建议
    recommended_k: int = Field(
        default=5,
        description="推荐的检索文档数量（当不分解时使用）"
    )

    # 是否需要多轮检索
    needs_multi_round_retrieval: bool = Field(
        description="是否需要多轮检索（对于多跳查询或需要迭代细化的查询）"
    )

    # ==================== 元信息 ====================

    # 置信度
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="意图识别的置信度（0-1）"
    )

    # 推理过程说明
    reasoning: str = Field(
        description="意图识别和分解决策的推理过程（使用与查询相同的语言）"
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
        template = """你是一个专业的查询意图分析器和查询分解专家。请分析以下查询，识别其意图类型、复杂度，并**自主判断是否需要将查询分解为多个子查询**以提高检索效果。

# 分析要求

1. **联合意图检测与槽位填充**：同时识别意图类型和提取关键信息
2. **自主分解判断**：根据查询复杂度和特点，自动判断是否需要分解
3. **通用性**：使用通用的方法识别意图，适用于任何领域和场景

# 意图类型说明

- factual: 事实性查询（询问具体事实、数据、定义）
- comparison: 对比查询（比较两个或多个对象/时间点/状态）
- analytical: 分析性查询（需要推理、分析、总结）
- procedural: 程序性查询（询问如何做某事）
- causal: 因果查询（询问原因、结果、影响）
- temporal: 时间序列查询（询问变化趋势、历史）
- multi_hop: 多跳查询（需要多个步骤推理）
- other: 其他类型

# 查询分解机制（核心功能）

## 是否需要分解的判断标准

**需要分解的信号**（满足任一条件）：
1. 包含多个独立的信息需求点（如"介绍X的原理、应用和前景"）
2. 需要对比多个对象/时间点（comparison）
3. 需要多步推理，后续步骤依赖前序结果（multi_hop）
4. 需要从多个维度分析（analytical）
5. 时间跨度大，需要按时间段查询（temporal）
6. 涉及因果链条，有多个层次（causal）
7. 单次检索难以覆盖所有信息需求

**不需要分解的信号**：
1. 简单的单一事实查询（如"北京的人口是多少？"）
2. 查询已经足够具体明确
3. procedural类型查询（步骤是内容本身，不是检索单位）
4. 分解会导致上下文信息丢失

## 分解类型

- comparison: 对比分解 - 按对比项分解（A vs B → 查A + 查B），可并行执行
- multi_hop: 多跳分解 - 按推理步骤分解，有顺序依赖（先查X，再根据X查Y）
- dimensional: 多维分解 - 按分析维度分解（分析原因 → 技术+商业+市场），可并行执行
- temporal: 时间分解 - 按时间段分解（10年发展 → 多个时间段），可并行执行
- causal_chain: 因果链分解 - 按因果关系分解（为什么 → 直接+间接+根本原因）
- information_needs: 信息需求分解 - 按独立信息需求点分解，可并行执行

## 子查询结构 SubQuery

每个子查询包含：
- query: 子查询文本
- purpose: 该子查询的目的说明
- recommended_strategy: 推荐的检索策略 ["semantic"] / ["hybrid"] / ["rerank"] / ["semantic", "rerank"]
- recommended_k: 推荐的检索数量 (3-10)
- order: 执行顺序（0=可并行，1/2/3...=按顺序执行）
- depends_on: 依赖的子查询索引列表（用于multi_hop）

## 检索策略选择指南

- semantic: 简单事实查询，明确单一信息点
- hybrid: 需要多角度信息，多样化信息片段
- rerank: 专业术语，需要高精度匹配
- 组合策略: 复杂专业查询可用 ["semantic", "rerank"]

## 分解示例

### 示例1 - comparison（对比分解，可并行）
原查询："2019和2020年苹果营收对比"
分析：对比查询，需要分解为独立的事实查询
sub_queries: [
  {{"query": "2019年苹果的营收是多少？", "purpose": "获取2019年数据", "recommended_strategy": ["semantic"], "recommended_k": 3, "order": 0}},
  {{"query": "2020年苹果的营收是多少？", "purpose": "获取2020年数据", "recommended_strategy": ["semantic"], "recommended_k": 3, "order": 0}}
]

### 示例2 - multi_hop（多跳分解，有顺序依赖）
原查询："谁是马云的大学同学中最成功的企业家？"
分析：需要多步推理，先查同学，再查成就
sub_queries: [
  {{"query": "马云的大学同学有哪些人？", "purpose": "获取同学名单", "recommended_strategy": ["hybrid"], "recommended_k": 5, "order": 1, "depends_on": []}},
  {{"query": "马云大学同学中有哪些人成为了企业家？", "purpose": "筛选企业家", "recommended_strategy": ["hybrid"], "recommended_k": 5, "order": 2, "depends_on": [0]}},
  {{"query": "这些企业家同学各自的成就和影响力如何？", "purpose": "比较成就", "recommended_strategy": ["hybrid"], "recommended_k": 8, "order": 3, "depends_on": [1]}}
]

### 示例3 - dimensional（多维分解，可并行）
原查询："分析特斯拉成功的原因"
分析：需要从多个维度分析
sub_queries: [
  {{"query": "特斯拉的技术创新有哪些？", "purpose": "技术维度", "recommended_strategy": ["hybrid"], "recommended_k": 5, "order": 0}},
  {{"query": "特斯拉的商业模式是什么？", "purpose": "商业维度", "recommended_strategy": ["hybrid"], "recommended_k": 5, "order": 0}},
  {{"query": "特斯拉的市场营销策略是什么？", "purpose": "营销维度", "recommended_strategy": ["hybrid"], "recommended_k": 5, "order": 0}},
  {{"query": "特斯拉的领导力和企业文化如何？", "purpose": "管理维度", "recommended_strategy": ["hybrid"], "recommended_k": 5, "order": 0}}
]

### 示例4 - causal_chain（因果链分解）
原查询："为什么2008年金融危机会发生？"
分析：涉及因果链条
sub_queries: [
  {{"query": "2008年金融危机的直接导火索是什么？", "purpose": "直接原因", "recommended_strategy": ["semantic"], "recommended_k": 5, "order": 0}},
  {{"query": "次贷危机是如何引发的？", "purpose": "间接原因", "recommended_strategy": ["hybrid"], "recommended_k": 5, "order": 0}},
  {{"query": "2008年金融危机的深层制度性原因是什么？", "purpose": "根本原因", "recommended_strategy": ["hybrid"], "recommended_k": 5, "order": 0}}
]

### 示例5 - information_needs（信息需求分解，可并行）
原查询："介绍量子计算的原理、应用和发展前景"
分析：包含3个独立信息需求点
sub_queries: [
  {{"query": "量子计算的基本原理是什么？", "purpose": "原理介绍", "recommended_strategy": ["semantic"], "recommended_k": 5, "order": 0}},
  {{"query": "量子计算目前有哪些应用场景？", "purpose": "应用场景", "recommended_strategy": ["hybrid"], "recommended_k": 5, "order": 0}},
  {{"query": "量子计算的发展前景如何？", "purpose": "发展前景", "recommended_strategy": ["hybrid"], "recommended_k": 5, "order": 0}}
]

### 示例6 - 不需要分解
原查询："北京的人口是多少？"
分析：简单事实查询，无需分解
needs_decomposition: false
sub_queries: []

# 查询

{query}

# 输出要求

请严格按照QueryIntent结构输出JSON：
1. 准确识别意图类型
2. 正确评估复杂度（simple/moderate/complex）
3. **自主判断是否需要分解**（needs_decomposition）
4. 如需分解，指定分解类型（decomposition_type）和原因（decomposition_reason）
5. 生成子查询列表（sub_queries），每个子查询包含完整信息
6. 提取关键实体和时间点
7. 给出置信度和推理过程（reasoning使用与查询相同的语言）

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
        has_comparison_pattern = any(re.search(pattern, query, re.IGNORECASE) for pattern in comparison_patterns)
        
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
        
        # ==================== 通用查询分解判断 ====================
        sub_queries: List[SubQuery] = []
        needs_decomposition = False
        decomposition_type: Optional[str] = None
        decomposition_reason = ""

        # 判断是否需要分解：检测对比查询
        if has_comparison_pattern or len(time_points) >= 2:
            needs_decomposition = True
            decomposition_type = "comparison"
            decomposition_reason = "检测到对比查询，需要拆分为独立的事实查询"
        elif complexity == "complex":
            # 复杂查询可能需要分解
            needs_decomposition = True
            decomposition_type = "information_needs"
            decomposition_reason = "复杂查询包含多个信息需求点"

        # 如果需要分解，生成子查询
        if needs_decomposition:
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

            # 情况1：时间对比/时间分解 - 为每个时间点生成查询
            if len(time_points) >= 2:
                # 如果之前没有设置 decomposition_type，根据是否有对比模式决定
                if not decomposition_type:
                    decomposition_type = "comparison" if has_comparison_pattern else "temporal"
                base_query = clean_query
                for tp in time_points:
                    base_query = base_query.replace(tp, '').strip()

                for tp in time_points:
                    if base_query:
                        sub_queries.append(SubQuery(
                            query=f"{tp}{base_query}是多少？",
                            purpose=f"获取{tp}的具体数据",
                            recommended_strategy=["semantic"],
                            recommended_k=3,
                            order=0  # 可并行执行
                        ))

            # 情况2：对象对比 - 为每个检测到的实体生成查询
            elif len(entities) >= 2 and has_comparison_pattern:
                decomposition_type = "comparison"
                # 尝试从查询中提取属性关键词
                attribute_keywords = []
                attribute_patterns = [
                    r'(价格|市值|营收|收入|利润|规模|性能|速度|效率)',
                    r'(price|value|revenue|profit|performance|speed|efficiency)',
                ]
                for pattern in attribute_patterns:
                    matches = re.findall(pattern, clean_query, re.IGNORECASE)
                    attribute_keywords.extend(matches)

                attribute = attribute_keywords[0] if attribute_keywords else "情况"

                for entity in entities[:4]:  # 最多处理4个实体
                    sub_queries.append(SubQuery(
                        query=f"{entity}的{attribute}是什么？",
                        purpose=f"获取{entity}的{attribute}信息",
                        recommended_strategy=["semantic"],
                        recommended_k=3,
                        order=0
                    ))

            # 情况3：复杂查询但无法明确拆分 - 生成开放性查询
            elif clean_query:
                sub_queries.append(SubQuery(
                    query=f"{clean_query}的具体情况是什么？",
                    purpose="获取综合信息",
                    recommended_strategy=["hybrid"],
                    recommended_k=8,
                    order=0
                ))

        # 确定意图类型
        if decomposition_type == "comparison":
            intent_type = "comparison"
        elif complexity == "complex":
            intent_type = "analytical"
        else:
            intent_type = "factual"

        return QueryIntent(
            intent_type=intent_type,
            complexity=complexity,
            needs_decomposition=needs_decomposition,
            decomposition_type=decomposition_type,
            decomposition_reason=decomposition_reason,
            sub_queries=sub_queries,
            entities=entities,
            time_points=time_points,
            recommended_retrieval_strategy=["hybrid"] if needs_decomposition else ["semantic"],
            recommended_k=10 if needs_decomposition else (5 if complexity == "simple" else 7),
            needs_multi_round_retrieval=complexity == "complex",
            confidence=0.5,
            reasoning=f"回退模式：使用通用启发式规则。{decomposition_reason if needs_decomposition else '简单查询，无需分解。'}"
        )