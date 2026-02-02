"""Intent Router for multi-agent system.

Separates intent recognition and entity extraction from plan generation,
following the single responsibility principle.

Design goals:
- Fast intent classification with focused prompt
- Deterministic external platform detection
- Clean separation from planning logic
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.multi_agent.planning.query_intent import QueryIntent
from src.multi_agent.prompts import render_context_bundle


INTENT_ROUTER_SYSTEM_TEMPLATE = """你是企业级电商客服系统的"意图路由器（Intent Router）"。

你的唯一职责是：准确识别用户意图、提取业务实体、判断是否需要外网搜索。
你不负责生成执行计划（由下游 Planner 处理）。

## 核心任务

1) 意图分类（business_intent_type）
- social_chat：纯社交/寒暄/打招呼
- general_chat：闲聊/情绪表达/抱怨（无明确业务目标）
- product_search：商品搜索/查询/了解（单个商品的详情/参数/优缺点等）
- product_comparison：明确的商品对比/比较/哪个好（至少涉及两个商品）
- order_management：订单相关（查询/取消/创建订单）

2) 实体提取（entities）
- product_id / product_ids：商品ID（从上下文中获取）
- order_id：订单ID
- search_keyword：搜索关键词
- quantity：数量
- general_entities：其他实体
- time_points：时间点

3) 外部平台检测（external_platform + requires_external_search）
【规则1】如果用户明确指定外部电商网站，必须提取并设置：
  - 触发关键词：在京东、在淘宝、在天猫、在咸鱼、在闲鱼、京东上、淘宝上、去XX搜索等
  - external_platform = 平台名称（如 "京东"、"淘宝"、"咸鱼"）
  - requires_external_search = true
  - 示例："在京东搜索iPhone" → external_platform="京东", requires_external_search=true
  - 示例："淘宝上有没有小米电视" → external_platform="淘宝", requires_external_search=true

【规则2】如果用户要求跨平台比价：
  - requires_external_search = true
  - external_platform = null（因为是多平台）
  - 示例："帮我比较京东和淘宝的价格" → requires_external_search=true

【规则3】如果用户没有指定外部网站：
  - external_platform = null
  - requires_external_search = false
  - 示例："帮我搜索iPhone 15" → external_platform=null, requires_external_search=false

4) 订单子意图（order_intent）
仅当 business_intent_type=order_management 时填写：
- query：查看/查询/订单状态/帮我看一下订单
- cancel：取消/撤销/不要了/帮我取消订单
- create：创建订单（仅当上下文中已有具体 product_id 时）
- other：其他订单相关

## 关键规则

1. 购买意向 ≠ 创建订单
   - "帮我买/想买/购买XX"但上下文中没有具体 product_id → business_intent_type=product_search
   - 上下文中已有具体 product_id（用户点击了购买） → business_intent_type=order_management, order_intent=create

2. 代词指代
   - "它/这个/那款"应指代上下文中最近一次已展示/已锁定的商品
   - 从 context 的 product_id 或 product_ids 推断

3. 礼貌语处理
   - "谢谢，帮我查一下订单" → 按业务请求分类，不是 social_chat

4. 情绪表达处理
   - 纯情绪/抱怨/辱骂（无业务目标）→ general_chat
   - 情绪 + 业务请求 → 按业务请求分类

5. 业务意图一致性
   - business_intent_type=product_comparison → intent_type=comparison
   - 单个商品（详情/参数/优缺点）→ product_search，不是 product_comparison

## 输出字段

- intent_type: factual/comparison/analytical/procedural/causal/temporal/multi_hop/other
- complexity: simple/moderate/complex
- business_intent_type: social_chat/general_chat/product_search/product_comparison/order_management
- order_intent: query/cancel/create/other（仅 order_management 时）
- external_platform: 外部平台名称或 null
- requires_external_search: true/false
- entities: 提取的实体
- confidence: 0.0-1.0
- reasoning: 简短推理说明
"""


INTENT_ROUTER_USER_TEMPLATE = """<context>
下面是结构化上下文（包含历史、实体、意图、阶段等）：

{context_block}
</context>

<user_query>
用户本次请求：
{user_query}
</user_query>

<validation_feedback>
（如果上一轮输出未通过schema校验，这里是校验错误原因；你必须修正后再输出）：
{validation_feedback}
</validation_feedback>

请输出意图识别结果（JSON）。"""


class IntentRouter:
    """Intent Router - 专注于意图分类和实体提取
    
    职责：
    1. 意图分类 (business_intent_type)
    2. 实体提取 (entities)
    3. 外部平台检测 (external_platform, requires_external_search)
    4. 订单子意图 (order_intent)
    
    不负责：
    - 计划生成（由 Planner 处理）
    - Agent 选择策略（由 Planner 处理）
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.structured_llm = llm.with_structured_output(
            QueryIntent, method="function_calling"
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INTENT_ROUTER_SYSTEM_TEMPLATE),
                ("user", INTENT_ROUTER_USER_TEMPLATE),
            ]
        )
    
    async def classify(
        self,
        *,
        user_query: str,
        context_bundle: Optional[Dict[str, Any]],
    ) -> QueryIntent:
        """执行意图分类和实体提取
        
        Args:
            user_query: 用户查询
            context_bundle: 上下文信息
            
        Returns:
            QueryIntent: 识别的意图和提取的实体
        """
        context_block = render_context_bundle(context_bundle)
        chain = self.prompt | self.structured_llm
        
        # Retry mechanism for schema validation errors
        validation_feedback = ""
        last_err: Exception | None = None
        for _ in range(3):
            try:
                return await chain.ainvoke(
                    {
                        "context_block": context_block,
                        "user_query": user_query,
                        "validation_feedback": validation_feedback,
                    }
                )
            except Exception as e:
                last_err = e
                validation_feedback = f"{type(e).__name__}: {str(e)}"
                continue
        
        assert last_err is not None
        raise last_err
