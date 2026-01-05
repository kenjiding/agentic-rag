"""Supervisor Agent - 监督者Agent，负责路由和协调

本模块实现了Supervisor Agent，它是多Agent系统的核心协调者。
Supervisor负责分析用户意图，决定调用哪个Agent或工具。

2025-2026 最佳实践（一步一步智能模式）：
- 基于当前状态（entities）进行智能路由决策
- 每次请求都重新进行意图识别和路由
- 支持动态Agent注册
- 提供路由决策的可解释性
- 错误处理和降级策略
"""
import re
from typing import Dict, Any, Optional, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator
from src.multi_agent.state import MultiAgentState, ConversationPhase
from src.multi_agent.agents.base_agent import BaseAgent
from src.multi_agent.config import get_keywords_config
import logging

logger = logging.getLogger(__name__)


class RoutingDecision(BaseModel):
    """路由决策结构定义

    使用Pydantic模型定义路由决策的输出结构，确保LLM输出符合预期格式。

    关键约束（双向逻辑一致性）：
    1. 当next_action为"finish"时，selected_agent必须为None
    2. 当selected_agent不为None时，next_action不能为"finish"
    """
    next_action: Literal["rag_search", "chat", "product_search", "order_management", "finish"] = Field(
        ...,
        description="下一步行动：rag_search表示需要RAG搜索，chat表示一般对话，product_search表示商品搜索，order_management表示订单管理，finish表示结束。注意：如果设置为finish，则selected_agent必须为null。"
    )
    selected_agent: Optional[Literal["rag_agent", "chat_agent", "product_agent", "order_agent"]] = Field(
        None,
        description="选中的Agent名称。CRITICAL CONSTRAINT: 如果next_action为finish，则selected_agent必须为null（None）。否则，必须指定一个有效的Agent名称。"
    )
    routing_reason: str = Field(
        ...,
        description="路由决策的原因说明，解释为什么选择这个Agent或行动"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="决策置信度，0.0-1.0之间的数值，表示对决策的把握程度"
    )

    @model_validator(mode='after')
    def validate_action_agent_consistency(self):
        """验证：next_action和selected_agent之间的逻辑一致性"""
        if self.next_action == "finish" and self.selected_agent is not None:
            raise ValueError(
                f"逻辑错误：next_action和selected_agent不一致。"
                f"next_action为'finish'时，selected_agent必须为None（表示任务结束，不需要路由到任何agent）。"
                f"但收到selected_agent={self.selected_agent}。"
                f"解决方案：如果selected_agent有值，请将next_action设置为对应的动作（如'order_management'、'product_search'等）；"
                f"如果确实要结束，请将selected_agent设置为null。"
            )
        return self


class SupervisorAgent:
    """Supervisor Agent - 多Agent系统的协调者（一步一步智能模式）

    职责：
    1. 分析用户意图和需求
    2. 根据当前状态（entities）进行智能路由决策
    3. 管理执行流程
    4. 处理错误和重试

    2025-2026 最佳实践：
    - 基于LLM的智能路由
    - 支持Agent能力描述
    - 可解释的决策过程
    - 灵活的扩展机制
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        agents: Optional[List[BaseAgent]] = None,
        fallback_llm: Optional[ChatOpenAI] = None
    ):
        """
        初始化Supervisor

        Args:
            llm: 语言模型实例，用于路由决策
            agents: 可用的Agent列表
            fallback_llm: 降级策略使用的LLM（可选，如果为None则使用更便宜的模型）
        """
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.agents: Dict[str, BaseAgent] = {}

        # 创建结构化输出的LLM（使用with_structured_output）
        self.structured_llm = self.llm.with_structured_output(RoutingDecision)

        # 降级策略使用的LLM（使用更便宜的模型，降低成本）
        self.fallback_llm = fallback_llm or ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1
        )
        self.fallback_structured_llm = self.fallback_llm.with_structured_output(RoutingDecision)

        # 注册Agents
        if agents:
            for agent in agents:
                self.register_agent(agent)

    def register_agent(self, agent: BaseAgent):
        """注册Agent"""
        self.agents[agent.get_name()] = agent
        logger.info(f"Supervisor注册Agent: {agent.get_name()}")

    def get_available_agents(self) -> List[Dict[str, str]]:
        """获取可用Agent列表及其描述"""
        return [
            {
                "name": agent.get_name(),
                "description": agent.get_description()
            }
            for agent in self.agents.values()
        ]

    async def route(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        路由决策 - 决定调用哪个Agent（一步一步智能模式）

        使用LLM分析用户意图和当前状态（entities），选择最合适的Agent。
        每次请求都重新进行路由决策，不依赖预先定义的任务链。

        Args:
            state: 当前的多Agent系统状态

        Returns:
            包含以下字段的字典：
            - next_action: 下一步行动（"rag_search", "chat", "product_search", "order_management", "finish"）
            - selected_agent: 选中的Agent名称（��果有）
            - routing_reason: 路由决策的原因说明
            - confidence: 决策置信度（0-1）
        """
        try:
            user_message = self._extract_user_message(state)

            if not user_message:
                return {
                    "next_action": "finish",
                    "selected_agent": None,
                    "routing_reason": "未找到用户消息",
                    "confidence": 0.0
                }

            # 【任务完成检测】检查是否需要清理状态
            state_updates = self._check_and_cleanup_completed_task(state, user_message)
            if state_updates:
                # 需要清理状态，先返回清理后的状态，不进行路由
                # 下一次请求会使用清理后的状态进行路由
                logger.info(f"【任务完成】检测到任务完成，清理状态: {state_updates}")
                return {
                    "next_action": "chat",
                    "selected_agent": "chat_agent",
                    "routing_reason": "任务已完成，清理临时状态",
                    "confidence": 1.0,
                    **state_updates  # 包含 entities 清理、conversation_phase 重置等
                }

            # 执行 LLM 路由决策
            llm_result = await self._do_llm_routing(state, user_message)
            return llm_result

        except Exception as e:
            logger.error(f"Supervisor路由决策错误: {str(e)}", exc_info=True)
            return await self._fallback_routing_with_llm(self._extract_user_message(state) or "")

    def _check_and_cleanup_completed_task(self, state: MultiAgentState, user_message: str) -> Dict[str, Any]:
        """
        检查任务是否完成，并清理相关状态

        任务完成的信号：
        1. conversation_phase 为 order_completed
        2. 没有 confirmation_pending（订单已确认完成）
        3. 用户消息表示对话结束（感谢、再见等）或是一般性对话

        Args:
            state: 当前的多Agent系统状态
            user_message: 用户消息

        Returns:
            如果需要清理状态，返回包含清理后字段的字典；否则返回空字典
        """
        # 对话结束关键词（感谢、再见等）
        closing_keywords = [
            "谢谢", "感谢", "多谢", "好的", "知道了", "了解了",
            "再见", "拜拜", "没事了", "可以了", "行了",
            "thank", "thanks", "ok", "okay", "bye", "goodbye"
        ]

        user_message_lower = user_message.lower().strip()
        is_closing = any(keyword in user_message_lower for keyword in closing_keywords)

        current_phase = state.conversation_phase
        has_pending_confirmation = state.confirmation_pending is not None

        # 检测条件：
        # 1. 当前阶段是 order_completed（订单刚完成）
        # 2. 没有待确认操作
        # 3. 用户消息表示对话结束 OR 用户说"谢谢你的帮助"这类确认完成的话
        if (current_phase == "order_completed" and
            not has_pending_confirmation and
            is_closing):
            # 清理临时 entities
            entities_to_clear = ["product_id", "user_phone", "search_keyword", "quantity"]
            cleaned_entities = {k: v for k, v in state.entities.items() if k not in entities_to_clear}

            logger.info(f"【任务完成清理】清理前 entities: {state.entities}")
            logger.info(f"【任务完成清理】清理后 entities: {cleaned_entities}")

            return {
                "entities": cleaned_entities,
                "conversation_phase": "idle"
            }

        # 如果当前阶段是 order_completed，但用户说了新的业务相关的话
        # 也应该重置阶段（让新任务能正常开始）
        if current_phase == "order_completed" and not has_pending_confirmation:
            # 检查是否是新的业务意图（包含购买、查询、订单等关键词）
            business_keywords = ["买", "购买", "下单", "订单", "查", "搜索", "产品", "商品"]
            is_new_business = any(keyword in user_message for keyword in business_keywords)

            if is_new_business:
                logger.info(f"【新任务】检测到新的业务意图，重置阶段到 idle")
                return {
                    "conversation_phase": "idle"
                }

        return {}

    def _build_intent_context(self, query_intent: Optional[Dict[str, Any]]) -> str:
        """构建意图识别上下文信息"""
        if not query_intent:
            return "（无意图识别信息）"

        context_parts = []

        intent_type = query_intent.get("intent_type", "unknown")
        complexity = query_intent.get("complexity", "unknown")
        context_parts.append(f"意图类型: {intent_type}")
        context_parts.append(f"复杂度: {complexity}")

        return "\n".join(context_parts)

    def _build_entity_context(self, state: MultiAgentState) -> str:
        """
        构建实体状态上下文信息

        一步一步智能模式：让 LLM 能够看到累积的实体状态，
        根据当前状态智能路由到合适的 Agent。

        Args:
            state: 多 Agent 系统状态

        Returns:
            格式化的实体上下文字符串
        """
        all_entities = state.entities

        if not all_entities and not state.last_product_search_context:
            return "（无累积实体信息）"

        context_parts = []

        # 【对话阶段】帮助 LLM 快速判断当前对话状态
        phase_descriptions = {
            "idle": "空闲状态，没有正在进行的任务",
            "product_selecting": "正在选择产品",
            "order_creating": "正在创建订单（等待确认）",
            "order_completed": "订单已完成，等待用户确认或结束对话",
        }
        context_parts.append(f"【对话阶段】: {phase_descriptions.get(state.conversation_phase, state.conversation_phase)}")

        # 【关键状态指示】帮助 LLM 快速判断当前进度
        context_parts.append("\n【当前进度状态】")

        # 检查是否有 product_id（用户已选定产品）
        if all_entities.get("product_id"):
            context_parts.append("  ✓ 用户已选定产品 (product_id存在)")
        else:
            context_parts.append("  ✗ 用户未选定产品 (product_id不存在)")

        # 检查是否有 user_phone（用户已提供手机号）
        if all_entities.get("user_phone"):
            context_parts.append("  ✓ 用户已提供手机号")
        else:
            context_parts.append("  ✗ 用户未提供手机号")

        # 累积实体信息（详细）
        if all_entities:
            context_parts.append("\n【累积实体详细信息】")
            for key, value in all_entities.items():
                if value is not None:
                    context_parts.append(f"  - {key}: {value}")

        # 最近产品搜索上下文（用于用户取消后重新发起请求的场景）
        if state.last_product_search_context:
            search_ctx = state.last_product_search_context
            context_parts.append("\n【最近产品搜索记录】")
            context_parts.append(f"  - 搜索关键词: {search_ctx.get('search_keyword')}")
            context_parts.append(f"  - 数量: {search_ctx.get('quantity', 1)}")
            products = search_ctx.get("products", [])
            if products:
                context_parts.append(f"  - 搜索到 {len(products)} 个产品:")
                for p in products[:5]:  # 最多显示5个产品
                    name = p.get("name", "N/A")
                    pid = p.get("id") or p.get("product_id", "N/A")
                    price = p.get("price", "")
                    price_str = f" ¥{price}" if price else ""
                    context_parts.append(f"    * ID:{pid} {name}{price_str}")
                if len(products) > 5:
                    context_parts.append(f"    ... 还有 {len(products) - 5} 个产品")

        return "\n".join(context_parts)

    def _collect_all_entities(self, state: MultiAgentState) -> Dict[str, Any]:
        """收集所有可用的实体信息"""
        all_entities = state.entities.copy()

        query_intent = state.query_intent
        if query_intent and query_intent.get("entities"):
            intent_entities = query_intent["entities"]

            if hasattr(intent_entities, "model_dump"):
                entities_dict = intent_entities.model_dump(exclude_none=True)
            elif isinstance(intent_entities, dict):
                entities_dict = intent_entities
            else:
                entities_dict = {}

            for key, value in entities_dict.items():
                if value is not None:
                    if isinstance(value, list):
                        if len(value) > 0:
                            all_entities[key] = value
                    else:
                        all_entities[key] = value

        return all_entities

    def _get_agents_description(self) -> str:
        """构建可用 Agent 的描述文本"""
        return "\n".join([
            f"- {agent['name']}: {agent['description']}"
            for agent in self.get_available_agents()
        ])

    def _validate_selected_agent(self, agent_name: Optional[str]) -> Optional[str]:
        """验证并返回有效的 Agent 名称"""
        if not agent_name:
            return None
        if agent_name not in self.agents:
            logger.warning(f"选中的 Agent {agent_name} 不存在，使用 chat_agent")
            return "chat_agent" if "chat_agent" in self.agents else None
        return agent_name

    def _extract_user_message(self, state: MultiAgentState) -> Optional[str]:
        """从状态中提取最新的用户消息"""
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None

    async def _do_llm_routing(self, state: MultiAgentState, user_message: str) -> Dict[str, Any]:
        """
        执行 LLM 单步路由（一步一步智能模式）

        设计说明：
        - user_message 只包含当前查询（最后一条 HumanMessage）
        - 历史上下文通过 entity_context 提供（累积的实体信息）
        - entity_context 已包含累积实体信息，足以判断多轮补充场景
        """
        query_intent = state.query_intent
        intent_context = self._build_intent_context(query_intent)
        entity_context = self._build_entity_context(state)
        agents_description = self._get_agents_description()

        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能路由系统，负责根据用户问题和上下文信息决定调用哪个Agent。

可用Agent列表：
{agents}

路由规则（基于用户问题和上下文信息）：

【核心规则 - 购买流程】（最高优先级）：
- **用户想购买但没有选定具体产品（entities中无product_id）**：必须先路由到 product_agent 搜索产品！
  - 例如："帮我购买3个西门子产品"、"买2台华为手机"、"我要下单买冰箱"
  - next_action设为"product_search"，selected_agent设为"product_agent"
- **用户已选定产品（entities中有product_id）且提供了手机号**：路由到 order_agent 创建订单
  - next_action设为"order_management"，selected_agent设为"order_agent"

【商品查询规则】：
- 用户询问商品信息、价格、参数等（但没有购买意图）：选择 product_agent
- 例如："西门子产品有哪些"、"这款冰箱多少钱"、"华为手机有什么型号"

【订单管理规则】：
- **查询/取消订单**：选择 order_agent
- 例如："查一下我的订单"、"取消刚才的订单"、"我的订单状态"

【对话阶段路由规则】（重要）：
- **如果对话阶段为"正在选择产品"(product_selecting)**：
  - 用户提供了手机号 → 路由到 order_agent（继续订单流程）
  - 用户说"选择ID:1"、"买这个"、"我要这个"等 → 路由到 order_agent（用户已选定产品）
  - 用户说"不要了"、"换个"、"重新搜索"等 → 路由到 product_agent（重新搜索）
- **如果对话阶段为"正在创建订单"(order_creating)**：
  - 用户说"确认"、"好的"、"可以"等 → 路由到 order_agent（确认订单）
  - 用户说"取消"、"不要了"等 → 路由到 chat_agent（取消订单）
- **如果对话阶段为"订单已完成"(order_completed)**：
  - 用户说"谢谢"、"再见"等 → 路由到 chat_agent（结束对话）
  - 用户提出新的业务需求 → 路由到对应的 agent（开始新任务）

【普通交流/闲聊规则】（重要）：
- **用户表达感谢、问候、礼貌用语**：必须路由到 chat_agent，返回友好温暖的回复！
  - 例如："谢谢"、"感谢"、"你好"、"��见"、"好的"、"知道了"、"哈哈"等
  - next_action设为"chat"，selected_agent设为"chat_agent"
  - **严禁将普通交流设为finish！**

【知识检索规则】：
- 用户询问产品使用方法、功能介绍、技术问题等 → rag_agent
- 例如："怎么使用"、"如何操作"、"有什么功能"等

【finish使用规则】（极少使用）：
- 只有在完全无法理解用户意图，且没有任何agent能处理时才使用finish
- finish会导致直接结束对话返回空白，所以**优先选择chat_agent处理**
- 宁可路由到chat_agent让LLM尝试理解，也不要直接finish

【字段一致性规则】：
- 如果next_action不是"finish"，则必须指定一个有效的selected_agent
- 如果next_action是"finish"，则selected_agent必须为null（None）

【多轮对话处理】：
- 根据"累积实体信息"判断用户进度
- 如果 entities 中有 product_id，说明用户已选定产品，可以进入订单流程
- 如果 entities 中有 search_keyword 但没有 product_id，说明还在搜索阶段

【重新开始购买场景】：
- 如果用户说"我还是想买"、"重新开始"、"还是那个"、"再试试"等，且"最近产品搜索记录"中有产品信息
- 应该重新展示之前的产品列表，路由到 product_agent

**意图识别结果**（已由前置节点完成，仅供参考）：
{intent_context}

**累积实体信息**（包含用户已提供的所有信息）：
{entity_context}"""),
            ("user", "用户问题: {question}")
        ])

        try:
            routing_decision = await self.structured_llm.ainvoke(
                routing_prompt.format_messages(
                    agents=agents_description,
                    question=user_message,
                    intent_context=intent_context,
                    entity_context=entity_context
                )
            )

            selected_agent = self._validate_selected_agent(routing_decision.selected_agent)

            result = {
                "next_action": routing_decision.next_action,
                "selected_agent": selected_agent,
                "routing_reason": routing_decision.routing_reason,
                "confidence": routing_decision.confidence
            }

            logger.info(f"Supervisor路由决策: {result}")
            return result

        except Exception as e:
            logger.error(f"结构化输出解析失败: {e}, 使用降级策略", exc_info=True)
            return await self._fallback_routing_with_llm(user_message)

    async def _fallback_routing_with_llm(self, user_message: str) -> Dict[str, Any]:
        """
        降级路由策略（企业级最佳实践）- 使用更便宜的LLM进行快速路由
        """
        try:
            agents_description = self._get_agents_description()

            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个路由系统。快速分析用户问题，决定调用哪个Agent。

可用Agent：
{agents}

规则：
- 商品搜索 → product_agent (next_action: "product_search", selected_agent: "product_agent")
- 订单管理 → order_agent (next_action: "order_management", selected_agent: "order_agent")
- 知识检索 → rag_agent (next_action: "rag_search", selected_agent: "rag_agent")
- **普通交流/闲聊（谢谢、你好、再见等）** → chat_agent (next_action: "chat", selected_agent: "chat_agent")
- 其他无法处理的 → 优先选择chat_agent，极少数情况才用finish

CRITICAL: 当next_action为"finish"时，selected_agent必须为null（None），不能指定任何Agent。
IMPORTANT: 普通交流（如"谢谢"）必须路由到chat_agent，不要用finish！

快速决策。"""),
                ("user", "问题: {question}")
            ])

            routing_decision = await self.fallback_structured_llm.ainvoke(
                simple_prompt.format_messages(
                    agents=agents_description,
                    question=user_message
                )
            )

            selected_agent = self._validate_selected_agent(routing_decision.selected_agent)

            # 防御性检查
            if routing_decision.next_action == "finish" and selected_agent is not None:
                logger.warning(
                    f"降级策略检测到逻辑不一致：next_action='finish'但selected_agent={selected_agent}，"
                    f"强制将selected_agent设置为None"
                )
                selected_agent = None

            result = {
                "next_action": routing_decision.next_action,
                "selected_agent": selected_agent,
                "routing_reason": f"降级策略（LLM）: {routing_decision.routing_reason}",
                "confidence": routing_decision.confidence * 0.8
            }

            logger.info(f"降级策略路由决策: {result}")
            return result

        except Exception as e:
            logger.error(f"降级策略LLM路由失败: {e}, 使用最终降级方案", exc_info=True)
            return self._final_fallback_routing(user_message)

    def _final_fallback_routing(self, user_message: str) -> Dict[str, Any]:
        """
        最终降级策略 - 仅在LLM完全失败时使用
        """
        # 通用的问题模式检测
        has_question_mark = "?" in user_message or "？" in user_message

        # 如果包含问题特征，倾向于使用RAG搜索
        if has_question_mark:
            return {
                "next_action": "rag_search",
                "selected_agent": "rag_agent" if "rag_agent" in self.agents else None,
                "routing_reason": "最终降级策略：基于通用问题模式检测",
                "confidence": 0.4
            }

        # 默认使用chat_agent
        return {
            "next_action": "chat",
            "selected_agent": "chat_agent" if "chat_agent" in self.agents else None,
            "routing_reason": "最终降级策略：默认使用chat_agent",
            "confidence": 0.3
        }
