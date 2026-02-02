"""LLM planner for multi-step customer support tasks.

After Intent-Planner separation refactoring:
- IntentRouter handles intent classification and entity extraction
- Planner focuses only on generating executable plans based on the classified intent

The planner converts a user request + QueryIntent into a Plan
composed of reusable coarse steps. Steps are executed by plan_executor in the graph.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.multi_agent.planning.models import PlanningOutput
from src.multi_agent.planning.query_intent import QueryIntent
from src.multi_agent.prompts import render_context_bundle


PLANNER_SYSTEM_TEMPLATE = """你是企业级电商客服系统的"规划器（Planner）"。

你的唯一职责是：根据已识别的意图（query_intent）生成可执行的多步骤计划。
意图识别和实体提取已由上游 IntentRouter 完成，你不需要重复识别。

## 输入信息

你会收到：
1. 用户查询（user_query）
2. 已识别的意图（query_intent），包含：
   - business_intent_type: 业务意图类型
   - external_platform: 用户指定的外部平台（如有）
   - requires_external_search: 是否需要外网搜索
   - order_intent: 订单子意图（如有）
   - entities: 已提取的实体
3. 上下文信息（context）

## 计划生成规则

执行方式（step_type）：
- agent_call: 调用一个专职Agent完成该步骤
- ask_user: 向用户提问以补齐缺失信息
- finish: 结束（只能作为最后一步）

可用 agent 与 next_action 映射：
- rag_agent -> rag_search
- chat_agent -> chat
- product_agent -> product_search
- order_agent -> order_management
- consultation_agent -> consultation
- browser_agent -> browser_search

### Agent 选择策略（基于 query_intent 确定性路由）

【规则1】requires_external_search=true 且 external_platform 有值
→ 直接使用 browser_agent，无条件执行
示例计划：
  1) agent_call(browser_agent -> browser_search)
  2) finish

【规则2】requires_external_search=true 且 external_platform 为空（跨平台比价）
→ 直接使用 browser_agent
示例计划：
  1) agent_call(browser_agent -> browser_search)
  2) finish

【规则3】requires_external_search=false 且 business_intent_type=product_search
→ 先用 product_agent，设置 browser_agent 为条件 fallback
示例计划：
  1) agent_call(product_agent -> product_search)（无条件执行）
  2) agent_call(browser_agent -> browser_search)（设置 execution_condition: type=if_previous_empty, reference_agent=product_agent, result_key=products）
  3) finish

【规则4】business_intent_type=product_comparison
→ 根据 context 中是否已有商品ID决定
A) 已有两个商品ID：
  1) agent_call(consultation_agent -> consultation)
  2) finish
B) 只有商品名称，无ID：
  1) agent_call(product_agent -> product_search)
  2) agent_call(consultation_agent -> consultation)
  3) finish

【规则5】business_intent_type=order_management
→ 根据 order_intent 路由
  1) agent_call(order_agent -> order_management)
  2) finish

【规则6】business_intent_type=social_chat 或 general_chat
→ 使用 chat_agent
  1) agent_call(chat_agent -> chat)
  2) finish

### 声明式条件执行机制

每个步骤可以设置 execution_condition 字段：
- type="always"：无条件执行（默认）
- type="if_previous_empty"：仅当引用 agent 返回空结果时执行
  - 必须设置 reference_agent 和 result_key

### 通用规则

1. 禁止输出"只有finish一步"的计划
2. finish 只能作为最后一步，之前必须至少有一个 agent_call 或 ask_user
3. 高风险动作（退款/取消订单/创建订单等）必须标注 risk_level=high 或 medium
4. 若信息不足，使用 ask_user 步骤提出澄清问题
5. plan.steps 至少 1 步且 step_id 唯一

## 输出

只输出 plan（不需要输出 query_intent，因为已由上游提供）。
"""


PLANNER_USER_TEMPLATE = """<query_intent>
已识别的意图（由 IntentRouter 提供）：

business_intent_type: {business_intent_type}
external_platform: {external_platform}
requires_external_search: {requires_external_search}
order_intent: {order_intent}
intent_type: {intent_type}
entities: {entities}
confidence: {confidence}
reasoning: {reasoning}
</query_intent>

<context>
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

请输出 Plan（JSON）。"""


class Planner:
    """LLM Planner - 专注于计划生成
    
    职责：
    - 根据已识别的 QueryIntent 生成可执行计划
    - Agent 选择策略
    - 条件执行配置
    
    不负责：
    - 意图识别（由 IntentRouter 处理）
    - 实体提取（由 IntentRouter 处理）
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        # Use function-calling mode for structured output
        self.structured_llm = llm.with_structured_output(
            PlanningOutput, method="function_calling"
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PLANNER_SYSTEM_TEMPLATE),
                ("user", PLANNER_USER_TEMPLATE),
            ]
        )

    async def plan(
        self,
        *,
        user_query: str,
        query_intent: QueryIntent,
        context_bundle: Optional[Dict[str, Any]],
    ) -> PlanningOutput:
        """生成执行计划
        
        Args:
            user_query: 用户查询
            query_intent: 已识别的意图（由 IntentRouter 提供）
            context_bundle: 上下文信息
            
        Returns:
            PlanningOutput: 包含可执行计划
        """
        context_block = render_context_bundle(context_bundle)
        chain = self.prompt | self.structured_llm

        # Prepare intent fields for template
        entities_str = ""
        if query_intent.entities:
            entities_dict = query_intent.entities.model_dump(exclude_none=True)
            entities_str = str(entities_dict) if entities_dict else "无"
        
        # Hard-schema + retry (enterprise robustness)
        validation_feedback = ""
        last_err: Exception | None = None
        for _ in range(3):
            try:
                return await chain.ainvoke(
                    {
                        "context_block": context_block,
                        "user_query": user_query,
                        "validation_feedback": validation_feedback,
                        # Intent fields
                        "business_intent_type": query_intent.business_intent_type,
                        "external_platform": query_intent.external_platform or "无",
                        "requires_external_search": str(query_intent.requires_external_search),
                        "order_intent": query_intent.order_intent or "无",
                        "intent_type": query_intent.intent_type,
                        "entities": entities_str,
                        "confidence": query_intent.confidence,
                        "reasoning": query_intent.reasoning or "无",
                    }
                )
            except Exception as e:
                last_err = e
                validation_feedback = f"{type(e).__name__}: {str(e)}"
                continue
        
        assert last_err is not None
        raise last_err
