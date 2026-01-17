"""LLM-based intent classifier.

Core implementation of intent classification using LLM with structured output.
Based on 2025-2026 best practices for unified information extraction and query decomposition.
"""
from typing import Optional, List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
import logging
import re

from src.intent.classifier.base import BaseIntentClassifier
from src.intent.models.query_intent import QueryIntent, SubQuery, Entities
from src.intent.models.types import PipelineOption, DecompositionType, IntentType, ComplexityLevel
from src.intent.config.settings import IntentConfig
from src.utils.llm_factory import create_llm_for_intent_classification

logger = logging.getLogger(__name__)


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
        llm: Optional[BaseChatModel] = None,
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
            # 使用工厂函数创建 LLM，支持配置中的模型名称
            # 如果配置中的模型名称不包含 provider，默认为 openai
            model_name = self.config.llm_model
            if ":" not in model_name:
                model_name = f"openai:{model_name}"

            llm = create_llm_for_intent_classification(
                model_name=model_name,
                temperature=self.config.llm_temperature
            )

        self.llm = llm
        # 使用 with_structured_output 规范 schema 输出
        self._structured_llm = llm.with_structured_output(QueryIntent)

    @staticmethod
    def _get_classification_prompt_template() -> str:
        """获取意图分类的 prompt template（公共模板，避免重复）"""
        return """你是查询意图分析专家，请分析以下查询并输出结构化结果。

# 核心任务（按优先级排序）

**1. 业务意图分类（最高优先级）**
根据用户表达的真实意图，选择最合适的business_intent_type：
- "social_chat": 社交互动（感谢、问候、告别、闲聊等）
- "order_management": 订单管理（查询/取消/修改订单、**创建订单/购买**）
- "product_comparison": 产品对比（比较多个产品）
- "product_search": 产品搜索（查找产品、了解产品信息、产品推荐）
- "general_chat": 通用对话（无法明确归类的普通对话）

**关键判断规则**：
- **order_management**：用户已经选定产品（有明确product_id）且表达购买/下单意图 → 这是创建订单操作
- **product_search**：用户还在查找/搜索产品（没有明确product_id，或只是询问产品信息）

**判断示例**：
- "谢谢"、"谢谢你的帮助"、"你好"、"再见" → social_chat
- "查订单ORD123"、"取消订单"、"查询我的订单" → order_management
- **"购买产品ID:6"、"我要买6号产品"、"下单产品ID:1"、"购买1号"** → **order_management**（有明确product_id且表达购买意图）
- "iPhone和华为哪个好"、"对比这几个产品" → product_comparison
- "买iPhone"、"搜索产品"、"华为手机怎么样"、"我想了解iPhone 15 Pro" → product_search（没有明确product_id，只是搜索/了解）
- "今天天气如何"、"介绍下量子计算" → general_chat

**2. 实体提取**
提取关键实体（按优先级）：
- product_id: 明确的产品ID数字（如"产品ID:1"、"1号产品"、"买3号"）
- order_id: 订单号字符串（如"ORD123456"或"123"），**拒绝通用词汇**（"我的订单"、"这个订单"等不算）
- quantity: 购买数量（如"买3个"）
- search_keyword: 核心搜索词（不含"产品"、"商品"等通用词）
- general_entities: 人名、地名、组织等
- time_points: 时间点（年份、日期）

**3. 查询分解判断**
仅在以下情况分解查询：
- 包含多个独立信息点（如"X的原理、应用和前景"）
- 需要对比多个对象/时间点
- 需要多步推理（multi_hop）
- 需要多维度分析

分解类型：comparison（对比）、multi_hop（多跳）、information_needs（信息需求）、dimensional（多维）

# 意图类型（intent_type）

factual、comparison、analytical、procedural、causal、temporal、multi_hop、other

# 分解示例

**示例1 - 对比分解**：
查询："2019和2020年苹果营收对比"
→ sub_queries: [{{"query": "2019年苹果营收", "purpose": "获取2019年数据"}}, {{"query": "2020年苹果营收", "purpose": "获取2020年数据"}}]

**示例2 - 信息需求分解**：
查询："介绍量子计算的原理、应用和前景"
→ sub_queries: [{{"query": "量子计算原理"}}, {{"query": "量子计算应用"}}, {{"query": "量子计算前景"}}]

**示例3 - 多跳分解**：
查询："马斯克是什么学历？他创业经历如何？"
→ sub_queries: [{{"query": "马斯克学历", "order": 1}}, {{"query": "马斯克创业经历", "order": 2, "depends_on": [0]}}]

**示例4 - 无需分解**：
查询："北京人口是多少？"
→ needs_decomposition: false

# 实体提取示例

"产品ID:1" → product_id=1
"查订单ORD123456" → order_id="ORD123456"
"买3台华为手机" → quantity=3, search_keyword="华为手机"（无明确product_id，属于搜索）
"购买产品ID:6" → product_id=6（有明确product_id，属于订单管理）
"iPhone 15 Pro Max" → search_keyword="iPhone 15 Pro Max"（无明确product_id，属于搜索）
"查询我的订单" → order_id=null（无具体订单号）

# 输出要求

严格按照QueryIntent结构输出，确保：
1. business_intent_type准确反映用户真实意图
2. entities按优先级提取，避免错误识别
3. 合理判断是否需要查询分解
4. reasoning用查询相同语言，明确说明业务意图

# 查询

{query}

输出JSON："""

    def _process_classification_result(self, result: Any) -> QueryIntent:
        """处理分类结果（公共逻辑，避免重复）"""
        if isinstance(result, QueryIntent):
            intent = result
        elif isinstance(result, dict):
            intent = QueryIntent(**result)
        else:
            # 容错：使用 model_validate
            intent = QueryIntent.model_validate(result)

        # 后处理验证：确保order_id格式正确（Entities模型的validator会自动处理）
        # 如果order_id是无效的（如"订单"），validator会将其设置为None
        if intent.entities and intent.entities.order_id:
            # validator已经验证过，这里只是记录日志
            logger.debug(f"提取到order_id: {intent.entities.order_id}")

        return intent

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
        template = self._get_classification_prompt_template()
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self._structured_llm

        try:
            result = chain.invoke({"query": query})
            return self._process_classification_result(result)
        except Exception as e:
            logger.error(f"[意图识别] 错误: {e}", exc_info=True)
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
        # Extract quantity (numbers followed by 件/个/台)
        quantity_pattern = re.compile(r'(\d+)\s*[件个台]')
        quantity_match = quantity_pattern.search(query)
        quantity: Optional[int] = None
        if quantity_match:
            try:
                quantity = int(quantity_match.group(1))
            except ValueError:
                pass

        # Extract search keyword (remove common/generic words, keep core keywords)
        # 通用词汇列表：这些词不应该出现在 search_keyword 中
        generic_words_to_remove = [
            "下单", "购买", "买", "我要", "我想", "想要", "想", "要",
            "件", "个", "台", "款",
            "商品", "产品", "东西", "货", "物品",
            "订单", "订单号", "是", "的", "有没有", "有", "没有",
            "一下", "看看", "查", "查询", "搜索", "找", "帮我"
        ]
        search_keyword = query
        for keyword in generic_words_to_remove:
            search_keyword = search_keyword.replace(keyword, "")
        # Remove phone numbers and long numbers (likely order IDs)
        search_keyword = re.sub(r'1[3-9]\d{9}', "", search_keyword)  # Remove phone numbers
        search_keyword = re.sub(r'\d{6,}', "", search_keyword)  # Remove long numbers (order IDs)
        search_keyword = re.sub(r'\d+', "", search_keyword)  # Remove remaining numbers
        search_keyword = re.sub(r'[，。、；：？！,.;:?!\s]+', "", search_keyword).strip()
        # Only set search_keyword if there's meaningful content left
        search_keyword_value: Optional[str] = search_keyword if search_keyword and len(search_keyword) >= 2 else None

        # Extract product_id (优先级高于 search_keyword)
        # 匹配模式：产品ID:1, 产品ID：1, ID:1, 1号产品, 第1个, 第一个, 我要买3号 等
        product_id_patterns = [
            r'(?:产品(?:ID|编号)|产品|ID)[：:]\s*(\d+)',  # "产品ID: 1", "产品编号：1"
            r'(\d+)号产品',  # "1号产品"
            r'(?:买|购买|要)?(?:我要|我想|想要)?(?:买|购买|要)?(?:第?)([一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]|10|\d+)号(?:产品)?$',  # "我要买3号", "买3号", "3号"
            r'第([一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]|10|\d+)个',  # "第1个", "第一个"
            r'就选第?([一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]|10|\d+)个?',  # "就选1", "就选第一个"
        ]
        product_id: Optional[int] = None
        for pattern in product_id_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    pid_str = match.group(1)
                    # 处理中文数字
                    chinese_nums = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                                   '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                                   '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5,
                                   '陆': 6, '柒': 7, '捌': 8, '玖': 9, '拾': 10}
                    if pid_str in chinese_nums:
                        product_id = chinese_nums[pid_str]
                    else:
                        product_id = int(pid_str)
                    break
                except ValueError:
                    pass

        # 如果提取到 product_id，清空 search_keyword（避免冲突）
        if product_id is not None:
            search_keyword_value = None

        # Extract order_id (订单ID，字符串格式，支持字母+数字组合如ORD123456或纯数字如"123")
        # 匹配模式：订单号ORD123、ORD123、订单ORD123、查询ORD1242343订单信息、订单ID:123等
        order_id_patterns = [
            r'(?:订单号|订单)[：:：\s]*(ORD[A-Z0-9]+|\d{6,})',  # "订单号ORD123"、"订单:ORD123"、"订单 13455556500"
            r'(ORD[A-Z0-9]+)',  # 独立的"ORD123456"格式
            r'(?:查询|查|想查)(?:.*?)(ORD[A-Z0-9]+|\d{6,})(?:订单|信息)',  # "查询ORD1242343订单信息"
            r'(?:订单)(?:.*?)(ORD[A-Z0-9]+|\d{6,})',  # "我现在想查询ORD1242343订单信息"
            r'(?:订单(?:ID|编号))[：:：\s]*(\d+)',  # "订单ID:123"、"订单编号456"（纯数字）
            r'(?:ID|编号)[：:：\s]*(\d+)(?:\s*订单|$)',  # "ID:123"、"编号456"
        ]
        order_id: Optional[str] = None
        for pattern in order_id_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted_value = match.group(1) if match.group(1) else None
                if extracted_value:
                    # 如果是包含字母的订单号（如ORD123456），转换为大写
                    # 如果是纯数字，保持原样（但作为字符串）
                    if 'ORD' in extracted_value.upper() or any(c.isalpha() for c in extracted_value):
                        order_id = extracted_value.upper()
                    else:
                        order_id = extracted_value  # 纯数字，作为字符串
                    if order_id:
                        break

        # Create Entities model instance
        entities = Entities(
            general_entities=general_entities,
            time_points=time_points,
            quantity=quantity,
            search_keyword=search_keyword_value,
            product_id=product_id,
            order_id=order_id
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
            reasoning=f"回退模式：使用通用启发式规则。{decomposition_reason if needs_decomposition else '简单查询，无需分解。'}",
            business_intent_type="general_chat",  # Fallback模式默认为通用对话
            suggested_next_action="chat"
        )

    async def aclassify(self, query: str) -> QueryIntent:
        """
        Asynchronously classify query intent using LLM.

        2025-2026 最佳实践：
        1. 使用异步LLM调用提高并发性能
        2. 保持与同步方法相同的功能

        Args:
            query: User query

        Returns:
            QueryIntent object with complete classification
        """
        template = self._get_classification_prompt_template()
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self._structured_llm

        try:
            # 使用异步调用
            result = await chain.ainvoke({"query": query})
            return self._process_classification_result(result)
        except Exception as e:
            logger.error(f"[意图识别] 异步调用错误: {e}", exc_info=True)
            # 降级到同步方法
            return self.classify(query)
