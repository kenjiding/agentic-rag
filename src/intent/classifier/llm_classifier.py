"""LLM-based intent classifier.

Core implementation of intent classification using LLM with structured output.
Based on 2025-2026 best practices for unified information extraction and query decomposition.
"""
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.intent.classifier.base import BaseIntentClassifier
from src.intent.models.query_intent import QueryIntent, SubQuery, Entities
from src.intent.models.types import PipelineOption, DecompositionType, IntentType, ComplexityLevel
from src.intent.config.settings import IntentConfig


class IntentClassifier(BaseIntentClassifier):
    """LLM-based intent classifier.

    Uses LLM with structured output for intent recognition and query decomposition.
    Supports multi-language queries and automatic decomposition decision making.

    Features:
    - Joint intent detection and slot filling
    - Automatic query decomposition decision
    - Domain-independent (general-purpose)
    - Fallback mechanism when LLM fails
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        config: Optional[IntentConfig] = None
    ):
        """
        Initialize the intent classifier.

        Args:
            llm: LLM instance (if None, creates default with config settings)
            config: Configuration (if None, uses default)
        """
        self.config = config or IntentConfig.default()

        if llm is None:
            llm = ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature
            )

        self.llm = llm
        # 使用 with_structured_output 规范 schema 输出
        self._structured_llm = llm.with_structured_output(QueryIntent)

    def classify(self, query: str) -> QueryIntent:
        """
        Classify query intent using LLM.

        Based on 2025-2026 best practices:
        1. Joint intent detection and slot filling
        2. Unified structure generation with structured output
        3. Domain-independent approach

        Args:
            query: User query

        Returns:
            QueryIntent object with complete classification
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
6. **提取所有实体**，统一存放在 entities 字典中：
   - general_entities: List[str] - 通用实体（人名、地名、组织等）
   - time_points: List[str] - 时间点（年份、日期等）
   - user_phone: Optional[str] - 用户手机号（11位，1开头）
   - quantity: Optional[int] - 购买数量
   - search_keyword: Optional[str] - 搜索关键词（商品名称）
7. 给出置信度和推理过程（reasoning使用与查询相同的语言）

输出JSON："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self._structured_llm

        try:
            result = chain.invoke({"query": query})
            # with_structured_output 直接返回 QueryIntent 对象
            if isinstance(result, QueryIntent):
                return result
            elif isinstance(result, dict):
                return QueryIntent(**result)
            else:
                # 容错：使用 model_validate
                return QueryIntent.model_validate(result)
        except Exception as e:
            print(f"[意图识别] 错误: {e}")
            return self._fallback_intent(query)

    def _fallback_intent(self, query: str) -> QueryIntent:
        """
        Fallback intent when LLM classification fails.

        Uses general heuristic rules that work across domains.

        Args:
            query: User query

        Returns:
            Default intent structure
        """
        import re

        # Complexity detection based on query length
        words = len(query.split())
        complexity: ComplexityLevel = "simple" if words < 5 else ("moderate" if words < 15 else "complex")

        # Comparison detection with multi-language patterns
        comparison_patterns = [
            # Chinese comparison words
            r'\b(对比|比较|相比|变化|上升|下降|增加|减少|差异|区别|哪个|哪种|更|较)\b',
            # English comparison words
            r'\b(compare|comparison|versus|vs|compared to|difference|change|increase|decrease|which|better|worse|more|less)\b',
            # Symbols
            r'\b(vs\.?|versus)\b'
        ]
        has_comparison_pattern = any(
            re.search(pattern, query, re.IGNORECASE) for pattern in comparison_patterns
        )

        # Time point detection with general time formats
        time_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # English dates
            r'\b(一月|二月|三月|四月|五月|六月|七月|八月|九月|十月|十一月|十二月)\b',  # Chinese months
        ]
        time_points: List[str] = []
        for pattern in time_patterns:
            time_points.extend(re.findall(pattern, query, re.IGNORECASE))
        time_points = list(set([str(tp).strip() for tp in time_points]))

        # ==================== Entity Extraction ====================
        # 统一实体模型，包含通用实体和业务实体
        # Extract general entities (通用实体)
        entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized words (English)
            r'"[^"]+"',  # Content in double quotes
            r"'[^']+'",  # Content in single quotes
        ]
        general_entities: List[str] = []
        for pattern in entity_patterns:
            general_entities.extend(re.findall(pattern, query))
        general_entities = list(set([e.strip('"\'') for e in general_entities]))[:10]

        # Extract business entities (业务实体)
        # Extract phone number (11 digits, starting with 1)
        phone_pattern = re.compile(r'1[3-9]\d{9}')
        phone_match = phone_pattern.search(query)
        user_phone: Optional[str] = phone_match.group(0) if phone_match else None

        # Extract quantity (numbers followed by 件/个)
        quantity_pattern = re.compile(r'(\d+)\s*[件个]')
        quantity_match = quantity_pattern.search(query)
        quantity: Optional[int] = None
        if quantity_match:
            try:
                quantity = int(quantity_match.group(1))
            except ValueError:
                pass

        # Extract search keyword (simplified: remove common words)
        keywords_to_remove = ["下单", "购买", "买", "我要", "件", "个", "商品", "产品", "手机号", "是", "的"]
        search_keyword = query
        for keyword in keywords_to_remove:
            search_keyword = search_keyword.replace(keyword, "")
        search_keyword = phone_pattern.sub("", search_keyword)  # Remove phone number
        search_keyword = re.sub(r'\d+', "", search_keyword)  # Remove numbers
        search_keyword = re.sub(r'[，。、；：？！,.;:?!]', "", search_keyword).strip()
        search_keyword_value: Optional[str] = search_keyword if search_keyword and len(search_keyword) >= 2 else None

        # Create Entities model instance
        entities = Entities(
            general_entities=general_entities,
            time_points=time_points,
            user_phone=user_phone,
            quantity=quantity,
            search_keyword=search_keyword_value
        )

        # ==================== Universal Query Decomposition ====================
        sub_queries: List[SubQuery] = []
        needs_decomposition = False
        decomposition_type: Optional[DecompositionType] = None
        decomposition_reason = ""

        # Determine if decomposition is needed: comparison query detection
        if has_comparison_pattern or len(time_points) >= 2:
            needs_decomposition = True
            decomposition_type = "comparison"
            decomposition_reason = "检测到对比查询，需要拆分为独立的事实查询"
        elif complexity == "complex":
            # Complex queries may need decomposition
            needs_decomposition = True
            decomposition_type = "information_needs"
            decomposition_reason = "复杂查询包含多个信息需求点"

        # Generate sub-queries if decomposition is needed
        if needs_decomposition:
            # General comparison word removal
            comparison_words = [
                # Chinese
                '对比', '比较', '相比', '变化', '上升', '下降', '增加', '减少',
                '差异', '区别', '和', '与', '还是', '哪个', '哪种', '更', '较',
                # English
                'compare', 'comparison', 'versus', 'vs', 'compared to', 'difference',
                'change', 'increase', 'decrease', 'and', 'or', 'which', 'better',
                'worse', 'more', 'less', 'than'
            ]

            clean_query = query
            for word in comparison_words:
                clean_query = re.sub(rf'\b{re.escape(word)}\b', ' ', clean_query, flags=re.IGNORECASE)
            clean_query = ' '.join(clean_query.split())  # Clean extra spaces

            # Case 1: Time comparison/temporal decomposition - generate query for each time point
            if len(time_points) >= 2:
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
                            order=0  # Can execute in parallel
                        ))

            # Case 2: Object comparison - generate query for each detected entity
            elif len(general_entities) >= 2 and has_comparison_pattern:
                decomposition_type = "comparison"
                # Try to extract attribute keywords from query
                attribute_keywords = []
                attribute_patterns = [
                    r'(价格|市值|营收|收入|利润|规模|性能|速度|效率)',
                    r'(price|value|revenue|profit|performance|speed|efficiency)',
                ]
                for pattern in attribute_patterns:
                    matches = re.findall(pattern, clean_query, re.IGNORECASE)
                    attribute_keywords.extend(matches)

                attribute = attribute_keywords[0] if attribute_keywords else "情况"

                for entity in general_entities[:4]:  # Handle at most 4 entities
                    sub_queries.append(SubQuery(
                        query=f"{entity}的{attribute}是什么？",
                        purpose=f"获取{entity}的{attribute}信息",
                        recommended_strategy=["semantic"],
                        recommended_k=3,
                        order=0
                    ))

            # Case 3: Complex query but unclear split - generate open query
            elif clean_query:
                sub_queries.append(SubQuery(
                    query=f"{clean_query}的具体情况是什么？",
                    purpose="获取综合信息",
                    recommended_strategy=["hybrid"],
                    recommended_k=8,
                    order=0
                ))

        # Determine intent type
        if decomposition_type == "comparison":
            intent_type: IntentType = "comparison"
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
            entities=entities,  # 统一实体模型，包含通用实体和业务实体
            recommended_retrieval_strategy=["hybrid"] if needs_decomposition else ["semantic"],
            recommended_k=10 if needs_decomposition else (5 if complexity == "simple" else 7),
            needs_multi_round_retrieval=complexity == "complex",
            confidence=0.5,
            reasoning=f"回退模式：使用通用启发式规则。{decomposition_reason if needs_decomposition else '简单查询，无需分解。'}"
        )
