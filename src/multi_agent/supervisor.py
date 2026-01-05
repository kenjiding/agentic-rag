"""Supervisor Agent - 监督者Agent，负责路由和协调

本模块实现了Supervisor Agent，它是多Agent系统的核心协调者。
Supervisor负责分析用户意图，决定调用哪个Agent或工具。

2025-2026 最佳实践：
- 使用LLM进行智能路由决策
- 支持动态Agent注册
- 提供路由决策的可解释性
- 错误处理和降级策略
- 使用with_structured_output确保输出格式正确
"""
import re
from typing import Dict, Any, Optional, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator
from src.multi_agent.state import MultiAgentState
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
    next_action: Literal["rag_search", "chat", "product_search", "order_management", "tool_call", "execute_task_chain", "finish"] = Field(
        ...,
        description="下一步行动：rag_search表示需要RAG搜索，chat表示一般对话，product_search表示商品搜索，order_management表示订单管理，tool_call表示工具调用，execute_task_chain表示执行任务链，finish表示结束。注意：如果设置为finish，则selected_agent必须为null。"
    )
    selected_agent: Optional[Literal["rag_agent", "chat_agent", "product_agent", "order_agent", "task_orchestrator"]] = Field(
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
        """验证：next_action和selected_agent之间的逻辑一致性
        
        双向约束规则（这两个约束是等价的，表示同一个逻辑关系）：
        1. 如果next_action为"finish"，则selected_agent必须为None
        2. 如果selected_agent不为None，则next_action不能为"finish"
        
        业务逻辑：如果指定了selected_agent，说明需要路由到某个agent执行任务，
        此时next_action应该是具体的动作（如"order_management"、"product_search"等），
        而不是"finish"。反之，如果next_action为"finish"，表示任务结束，
        不需要路由到任何agent，所以selected_agent必须为None。
        
        这是企业级最佳实践：使用Pydantic模型验证器强制字段间的逻辑一致性，
        确保LLM输出的结构化数据符合业务规则。
        """
        # 检查逻辑不一致：next_action为"finish"但selected_agent有值
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
    """Supervisor Agent - 多Agent系统的协调者
    
    职责：
    1. 分析用户意图和需求
    2. 决定调用哪个Agent或工具
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
        # 这样可以直接获得符合RoutingDecision结构的输出，无需手动解析JSON
        self.structured_llm = self.llm.with_structured_output(RoutingDecision)
        
        # 降级策略使用的LLM（使用更便宜的模型，降低成本）
        # 企业级最佳实践：降级时也使用LLM，但用更简单的prompt和更便宜的模型
        self.fallback_llm = fallback_llm or ChatOpenAI(
            model="gpt-3.5-turbo",  # 使用更便宜的模型
            temperature=0.1
        )
        self.fallback_structured_llm = self.fallback_llm.with_structured_output(RoutingDecision)
        
        # 注册Agents
        if agents:
            for agent in agents:
                self.register_agent(agent)
    
    def register_agent(self, agent: BaseAgent):
        """
        注册Agent
        
        Args:
            agent: 要注册的Agent实例
        """
        self.agents[agent.get_name()] = agent
        logger.info(f"Supervisor注册Agent: {agent.get_name()}")
    
    def get_available_agents(self) -> List[Dict[str, str]]:
        """
        获取可用Agent列表及其描述
        
        Returns:
            Agent信息列表，每个元素包含name和description
        """
        return [
            {
                "name": agent.get_name(),
                "description": agent.get_description()
            }
            for agent in self.agents.values()
        ]
    
    async def route(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        路由决策 - 决定调用哪个Agent

        使用LLM分析用户意图，选择最合适的Agent。
        如果状态中包含意图识别结果，会利用这些信息做更智能的路由。

        新增多步骤任务编排支持：
        - 检测任务链：如果状态中有活跃的任务链，路由到任务编排器
        - 创建任务链：检测是否需要创建多步骤任务链
        - 单步路由：原有的单步路由逻辑

        Args:
            state: 当前的多Agent系统状态

        Returns:
            包含以下字段的字典：
            - next_action: 下一步行动（"rag_search", "chat", "tool_call", "execute_task_chain", "finish"）
            - selected_agent: 选中的Agent名称（如果有）
            - routing_reason: 路由决策的原因说明
            - confidence: 决策置信度（0-1）
            - task_chain: 任务链（如果创建）
        """
        try:
            user_message = self._extract_user_message(state)

            # 1. 处理现有任务链状态
            chain_result, needs_cleanup = await self._handle_existing_task_chain(state, user_message)
            if chain_result:
                return chain_result

            # 2. 尝试创建新任务链
            create_result = await self._try_create_task_chain(state)
            if create_result:
                return create_result

            # 3. LLM 单步路由（降级路径）
            llm_result = await self._do_llm_routing(state, user_message)
            if needs_cleanup:
                llm_result["task_chain"] = None
                llm_result["routing_reason"] = f"[清理旧任务链后] {llm_result['routing_reason']}"
            return llm_result

        except Exception as e:
            logger.error(f"Supervisor路由决策错误: {str(e)}", exc_info=True)
            return await self._fallback_routing_with_llm(self._extract_user_message(state) or "")

    async def _handle_existing_task_chain(self, state: MultiAgentState, user_message: Optional[str]) -> tuple[Optional[Dict[str, Any]], bool]:
        """处理现有任务链状态

        Returns:
            (路由结果字典, 是否需要清理标记)
        """
        task_chain = state.task_chain
        query_intent = state.query_intent

        # 【调试日志】详细记录状态信息
        logger.info(
            f"[任务链处理] 检查现有任务链状态: "
            f"task_chain={task_chain is not None}, "
            f"user_message={user_message}, "
            f"query_intent={query_intent is not None}"
        )

        if task_chain:
            logger.info(f"[任务链处理] task_chain 详情: chain_type={task_chain.chain_type}, current_index={task_chain.current_step_index}, steps_count={len(task_chain.steps)}")

        if not task_chain:
            logger.info("[任务链处理] 没有活跃的任务链，返回 None")
            return None, False

        logger.info(f"检测到活跃任务链: task_chain={task_chain is not None}, user_message={user_message}")

        # 如果没有用户消息但有活跃的任务链，继续执行任务链
        if not user_message and task_chain:
            current_index = task_chain.current_step_index
            steps = task_chain.steps

            logger.info(f"检测到活跃任务链，无用户消息: current_index={current_index}, steps_count={len(steps)}")

            if current_index < len(steps):
                current_step = steps[current_index]
                step_type = current_step.step_type

                logger.info(f"任务链当前步骤: step_type={step_type}, index={current_index}")
                
                # 【关键修复】无论当前步骤是什么类型，只要有活跃的 task_chain，都应该路由到 task_orchestrator
                # task_orchestrator 会根据当前步骤类型执行相应的逻辑
                logger.info(f"无新用户消息但有活跃任务链，路由到 task_orchestrator: step_type={step_type}, index={current_index}")
                return {
                    "next_action": "execute_task_chain",
                    "selected_agent": None,
                    "routing_reason": f"恢复任务链执行，当前步骤: {step_type}",
                    "confidence": 1.0
                }, False
            else:
                # 任务链已完成
                logger.info(f"任务链已完成: current_index={current_index}, steps_count={len(steps)}")
                return None, True

        # 核心逻辑：检测用户新输入是否与任务链/选择相关
        should_clear_task_chain = False
        clear_reason = ""

        # 检查用户消息是否是选择/确认操作
        is_selection_response = False
        if user_message:
            keywords_config = get_keywords_config()
            is_pure_number = bool(re.match(r'^\d+$', user_message.strip()))
            is_selection_response = is_pure_number or any(kw in user_message for kw in keywords_config.selection_keywords)

        # 如果不是选择响应，检查用户输入是否是补充信息
        if not is_selection_response:
            if task_chain and not should_clear_task_chain:
                from src.multi_agent.task_orchestrator import TaskChainOrchestrator
                orchestrator = TaskChainOrchestrator()

                current_step_index = task_chain.current_step_index
                steps = task_chain.steps
                if current_step_index < len(steps):
                    current_step = steps[current_step_index]
                    step_type = current_step.step_type

                    step_def = orchestrator.AVAILABLE_STEP_TYPES.get(step_type)
                    if step_def:
                        required_fields = step_def.get("requires", [])
                        all_entities = self._collect_all_entities(state, include_task_chain=True)

                        missing_fields = []
                        for field in required_fields:
                            field_aliases = [field, f"selected_{field}"]
                            if not any(all_entities.get(alias) for alias in field_aliases):
                                missing_fields.append(field)

                        if missing_fields and user_message:
                            is_supplementing = self._check_if_supplementing_info(user_message, missing_fields, all_entities)
                            if is_supplementing:
                                logger.info(f"用户提供了补充信息，继续执行任务链: step_type={step_type}, missing_fields={missing_fields}")
                                return {"next_action": "execute_task_chain", "selected_agent": None,
                                        "routing_reason": f"用户提供了任务链所需的信息（{', '.join(missing_fields)}），继续执行",
                                        "confidence": 0.9}, False

            # 检查用户意图是否与任务链匹配
            if query_intent and task_chain:
                intent_type = query_intent.get("intent_type", "")
                chain_type = task_chain.chain_type
                is_order_query_intent = any(keyword in intent_type.lower() for keyword in ["order", "订单", "factual"])
                is_purchase_task_chain = chain_type == "order_with_search"

                if is_order_query_intent and is_purchase_task_chain:
                    should_clear_task_chain = True
                    clear_reason = f"用户意图变化（{intent_type}），与购买流程不匹配"

            # 检查任务链当前步骤
            if task_chain and not should_clear_task_chain:
                current_step_index = task_chain.current_step_index
                steps = task_chain.steps
                if current_step_index < len(steps):
                    current_step = steps[current_step_index]
                    step_type = current_step.step_type
                    step_status = current_step.status

        # 执行清理
        if should_clear_task_chain:
            logger.info(f"🧹 自动清理任务链: {clear_reason}")

            logger.info("任务链已清理，继续执行正常路由流程")
            return None, True

        # 意图匹配，继续执行任务链
        logger.info("用户输入与任务链匹配，继续执行任务链")
        return {
            "next_action": "execute_task_chain",
            "selected_agent": None,
            "routing_reason": "继续执行活跃的任务链",
            "confidence": 1.0
        }, False

    def _build_intent_context(self, query_intent: Optional[Dict[str, Any]]) -> str:
        """
        构建意图识别上下文信息

        Args:
            query_intent: 意图识别结果字典

        Returns:
            格式化的意图上下文字符串
        """
        if not query_intent:
            return "（无意图识别信息）"

        context_parts = []

        intent_type = query_intent.get("intent_type", "unknown")
        complexity = query_intent.get("complexity", "unknown")
        context_parts.append(f"意图类型: {intent_type}")
        context_parts.append(f"复杂度: {complexity}")

        needs_decomposition = query_intent.get("needs_decomposition", False)
        if needs_decomposition:
            decomposition_type = query_intent.get("decomposition_type")
            context_parts.append(f"需要分解: 是 ({decomposition_type})")

            sub_queries = query_intent.get("sub_queries", [])
            if sub_queries:
                context_parts.append(f"子查询数量: {len(sub_queries)}")
                context_parts.append("子查询:")
                for i, sq in enumerate(sub_queries[:3], 1):
                    sq_query = sq.get("query", str(sq)) if isinstance(sq, dict) else str(sq)
                    context_parts.append(f"  {i}. {sq_query[:60]}...")

        recommended_strategy = query_intent.get("recommended_retrieval_strategy", [])
        if recommended_strategy:
            context_parts.append(f"推荐检索策略: {', '.join(recommended_strategy)}")

        return "\n".join(context_parts)

    def _build_entity_context(self, state: MultiAgentState) -> str:
        """
        构建实体状态上下文信息

        根源解决方案：让 LLM 能够看到累积的实体状态，
        而不仅仅是当前用户消息。这样用户分多轮提供信息时，
        LLM 能够正确理解上下文，不会把补充信息当作一般对话。

        Args:
            state: 多 Agent 系统状态

        Returns:
            格式化的实体上下文字符串
        """
        all_entities = self._collect_all_entities(state)

        if not all_entities and not state.last_product_search_context:
            return "（无累积实体信息）"

        context_parts = []

        # 累积实体信息
        if all_entities:
            context_parts.append("累积实体信息:")
            for key, value in all_entities.items():
                if value is not None:
                    context_parts.append(f"  - {key}: {value}")

        # 最近产品搜索上下文（用于用户取消后重新发起请求的场景）
        if state.last_product_search_context:
            search_ctx = state.last_product_search_context
            context_parts.append("\n最近产品搜索记录:")
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

    async def _check_if_supplementing_info(self, user_message: str, missing_fields: List[str], current_entities: Dict[str, Any]) -> bool:
        """
        使用 LLM 判断用户是否在补充缺失的信息

        通用解决方案：不硬编码每种字段类型的检测模式，而是让 LLM 理解语义。

        Args:
            user_message: 用户输入消息
            missing_fields: 缺失的字段列表
            current_entities: 当前已收集的实体信息

        Returns:
            True 如果用户在补充信息，False 否则
        """
        try:
            from pydantic import BaseModel

            class SupplementCheck(BaseModel):
                is_supplementing: bool = Field(description="是否在补充信息")
                provided_field: str = Field(description="提供的字段名（如 user_phone、quantity 等）")

            structured_llm = self.llm.with_structured_output(SupplementCheck)

            prompt = f"""判断用户是否在补充任务所需的信息。

缺失字段: {', '.join(missing_fields)}
当前已收集信息: {current_entities}

用户输入: {user_message}

如果用户输入提供了缺失字段的值（如手机号、数量、地址等），返回 True。
注意：用户可能用各种方式表达，如"手机号是138..."、"就买2个"、"送到XXX"等。"""

            result = await structured_llm.ainvoke(prompt)
            if result.is_supplementing:
                logger.info(f"LLM 检测到用户补充了字段: {result.provided_field}")
            return result.is_supplementing
        except Exception as e:
            logger.warning(f"LLM 补充信息检测失败: {e}，保守返回 False")
            return False

    def _collect_all_entities(self, state: MultiAgentState, include_task_chain: bool = False) -> Dict[str, Any]:
        """
        收集所有可用的实体信息

        统一的实体收集逻辑，避免重复代码。

        Args:
            state: 多 Agent 系统状态
            include_task_chain: 是否包含任务链上下文（保留参数用于向后兼容）

        Returns:
            合并后的实体字典
        """
        all_entities = state.entities.copy()

        query_intent = state.query_intent
        if query_intent and query_intent.get("entities"):
            intent_entities = query_intent["entities"]
            
            # 处理 entities 可能是 Pydantic 模型或字典的情况
            if hasattr(intent_entities, "model_dump"):
                # 如果是 Pydantic 模型，使用 model_dump 并排除 None 值
                entities_dict = intent_entities.model_dump(exclude_none=True)
            elif isinstance(intent_entities, dict):
                # 如果是字典，直接使用
                entities_dict = intent_entities
            else:
                entities_dict = {}
            
            # 只更新有值的字段，避免空值覆盖已有值
            for key, value in entities_dict.items():
                if value is not None:
                    # 对于列表类型（如 general_entities, time_points），只有非空列表才覆盖
                    if isinstance(value, list):
                        if len(value) > 0:
                            all_entities[key] = value
                    else:
                        # 对于其他类型（str, int 等），直接更新
                        all_entities[key] = value

        return all_entities

    def _get_agents_description(self) -> str:
        """
        构建可用 Agent 的描述文本

        统一的 Agent 描述构建逻辑，避免重复代码。

        Returns:
            格式化的 Agent 描述字符串
        """
        return "\n".join([
            f"- {agent['name']}: {agent['description']}"
            for agent in self.get_available_agents()
        ])

    def _validate_selected_agent(self, agent_name: Optional[str]) -> Optional[str]:
        """
        验证并返回有效的 Agent 名称

        如果指定的 Agent 不存在，返回默认的 chat_agent。

        Args:
            agent_name: 要验证的 Agent 名称

        Returns:
            有效的 Agent 名称
        """
        if not agent_name:
            return None
        if agent_name not in self.agents:
            logger.warning(f"选中的 Agent {agent_name} 不存在，使用 chat_agent")
            return "chat_agent" if "chat_agent" in self.agents else None
        return agent_name

    def _extract_user_message(self, state: MultiAgentState) -> Optional[str]:
        """
        从状态中提取最新的用户消息

        Args:
            state: 多 Agent 系统状态

        Returns:
            最新的用户消息内容，如果没有则返回 None
        """
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None

    async def _fallback_routing_with_llm(self, user_message: str) -> Dict[str, Any]:
        """
        降级路由策略（企业级最佳实践）- 使用更便宜的LLM进行快速路由
        
        企业级最佳实践：
        1. 即使降级也使用LLM，确保决策质量
        2. 使用更便宜的模型（如gpt-3.5-turbo）降低成本
        3. 使用更简单的prompt，提高响应速度
        4. 仍然使用结构化输出，确保格式正确
        
        Args:
            user_message: 用户消息
            
        Returns:
            路由决策字典
        """
        try:
            agents_description = self._get_agents_description()

            # 简化的prompt，提高响应速度
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个路由系统。快速分析用户问题，决定调用哪个Agent。

可用Agent：
{agents}

规则：
- 商品搜索 → product_agent (next_action: "product_search", selected_agent: "product_agent")
- 订单管理 → order_agent (next_action: "order_management", selected_agent: "order_agent")
- 知识检索 → rag_agent (next_action: "rag_search", selected_agent: "rag_agent")
- 一般对话 → chat_agent (next_action: "chat", selected_agent: "chat_agent")
- 无法处理 → finish (next_action: "finish", selected_agent: null)

CRITICAL: 当next_action为"finish"时，selected_agent必须为null（None），不能指定任何Agent。

快速决策。"""),
                ("user", "问题: {question}")
            ])

            # 使用更便宜的模型进行降级路由（异步调用）
            routing_decision = await self.fallback_structured_llm.ainvoke(
                simple_prompt.format_messages(
                    agents=agents_description,
                    question=user_message
                )
            )

            # 验证选中的Agent是否存在
            selected_agent = self._validate_selected_agent(routing_decision.selected_agent)
            
            # 防御性检查：确保next_action为finish时，selected_agent为None
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
                "confidence": routing_decision.confidence * 0.8  # 降级策略的置信度稍低
            }
            
            logger.info(f"降级策略路由决策: {result}")
            return result
            
        except Exception as e:
            logger.error(f"降���策略LLM路由失败: {e}, 使用最终降级方案", exc_info=True)
            # 最终降级：如果LLM也失败，使用简单的启发式规则
            return self._final_fallback_routing(user_message)

    async def _try_create_task_chain(self, state: MultiAgentState) -> Optional[Dict[str, Any]]:
        """尝试创建新的任务链"""
        from src.multi_agent.task_orchestrator import get_task_orchestrator
        orchestrator = get_task_orchestrator()

        # 提取用户消息用于日志
        user_message = None
        for msg in reversed(state.messages):
            if hasattr(msg, 'content') and not hasattr(msg, 'name'):  # HumanMessage 没有 name 属性
                user_message = msg.content
                break

        logger.info(f"[任务链检测] 开始检测多步骤任务，用户消息: {user_message}")

        task_type = await orchestrator.detect_multi_step_task(state)

        if task_type:
            logger.info(f"[任务链检测] ✓ 检测到多步骤任务: {task_type}")
            new_task_chain = await orchestrator.create_task_chain(task_type, state)
            return {
                "next_action": "execute_task_chain",
                "selected_agent": None,
                "routing_reason": f"创建多步骤任务链: {task_type}",
                "confidence": 0.9,
                "task_chain": new_task_chain
            }

        logger.info(f"[任务链检测] ✗ 未检测到多步骤任务，将使用普通 LLM 路由")
        return None

    async def _do_llm_routing(self, state: MultiAgentState, user_message: Optional[str]) -> Dict[str, Any]:
        """执行 LLM 单步路由
        
        设计说明：
        - user_message 只包含当前查询（最后一条 HumanMessage），不包含历史消息
        - 历史上下文通过 entity_context 提供（累积的实体信息）
        - 这样设计的原因：
          1. Supervisor 职责是路由决策，不是理解对话细节
          2. entity_context 已包含累积实体信息，足以判断多轮补充场景
          3. 保持 Supervisor 轻量级，避免增加 token 消耗
          4. 指代消解等复杂理解由具体 Agent 处理（如 Order Agent 会从对话历史中提取信息）
        """
        if not user_message:
            return {
                "next_action": "finish",
                "selected_agent": None,
                "routing_reason": "未找到用户消息",
                "confidence": 0.0
            }

        query_intent = state.query_intent
        intent_context = self._build_intent_context(query_intent)
        entity_context = self._build_entity_context(state)
        agents_description = self._get_agents_description()

        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能路由系统，负责根据用户问题和上下文信息决定调用哪个Agent。

可用Agent列表：
{agents}

路由规则（基于用户问题和上下文信息）：
1. 商品相关：用户询问商品、搜索产品、比价等，选择 product_agent，next_action设为"product_search"，selected_agent设为"product_agent"

2. 订单相关：
   - **查询/取消订单**：选择 order_agent，next_action设为"order_management"，selected_agent设为"order_agent"
   - **创建订单**：如果用户提供了明确的 product_id（或累积状态中有），选择 order_agent，next_action设为"order_management"，selected_agent设为"order_agent"

3. 知识检索：如果用户问题需要从知识库中检索信息，选择 rag_agent，next_action设为"rag_search"，selected_agent设为"rag_agent"

4. 一般对话：如果是一般性对话或简单问题，选择 chat_agent，next_action设为"chat"，selected_agent设为"chat_agent"

5. **结束处理**：如果问题无法由现有Agent处理，next_action设为"finish"，selected_agent设为null（None）
   - **CRITICAL CONSTRAINT**: 当next_action为"finish"时，selected_agent必须为null（None），绝对不能指定任何Agent名称
   - 这是一个硬性约束，违反此规则会导致系统错误和验证失败

**字段一致性规则（必须严格遵守，这是系统级别的双向约束）**：
- 如果next_action不是"finish"（即"rag_search"、"chat"、"product_search"、"order_management"、"tool_call"、"execute_task_chain"），则必须指定一个有效的selected_agent（不能为null）
- 如果next_action是"finish"，则selected_agent必须为null（None），不能指定任何Agent名称
- **反向约束（同样重要）**：如果selected_agent不为null（有具体的agent名称），则next_action不能为"finish"，必须设置为对应的动作（例如：selected_agent="order_agent"时，next_action应为"order_management"）
- 这两个字段的值必须保持逻辑一致：不能出现next_action="finish"但selected_agent有值的情况，也不能出现selected_agent有值但next_action="finish"的情况

**重要**：用户可能分多轮提供信息或重新发起购买请求。
- 根据"累积实体信息"判断用户是否正在补充之前任务所需的信息。
- 例如：用户之前选择了商品（有 selected_product_id），现在只说了手机号，这应该路由到 order_agent 而不是 chat_agent。

**重新开始购买场景**：
- 如果用户说"我还是想买"、"重新开始"、"刚才的产品"、"还是那个"、"再试试"等表达，且"最近产品搜索记录"中有产品信息，说明用户想基于之前的搜索结果继续购买流程。
- 此时应该路由到 product_agent，next_action 设为 "product_search"，selected_agent 设为 "product_agent"，让产品代理重新展示产品列表。
- 例如：用户之前搜索了"西门子"产品并点击取消，现在说"不好意思，我还是想买"，应该重新展示之前搜索到的西门子产品列表。

**意图识别结果**（已由前置节点完成，仅供参考）：
{intent_context}

**累积实体信息**（包含用户已提供的所有信息）：
{entity_context}"""),
            ("user", "用户问题: {question}")
        ])

        try:
            # 使用异步LLM调用提高性能
            # 注意：只传递当前 user_message，历史上下文通过 entity_context 提供
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

    def _final_fallback_routing(self, user_message: str) -> Dict[str, Any]:
        """
        最终降级策略 - 仅在LLM完全失败时使用
        
        这是一个非常简单的启发式规则，仅在极端情况下使用。
        企业级最佳实践：应该尽量避免走到这一步。
        
        使用通用的模式检测，不依赖特定语言的关键词。
        
        Args:
            user_message: 用户消息
            
        Returns:
            路由决策字典
        """
        import re
        
        # 通用的问题模式检测（不依赖特定语言）
        # 1. 问号检测（通用符号）
        has_question_mark = "?" in user_message or "？" in user_message
        
        # 2. 疑问词模式检测（使用正则表达式，支持多语言）
        # 匹配常见的疑问词模式，不硬编码具体词汇
        question_patterns = [
            r'\b(what|who|when|where|why|how|which|whom|whose)\b',  # 英文疑问词
            r'\b(什么|谁|何时|哪里|为什么|如何|哪个|哪些)\b',  # 中文疑问词
            r'\b(quoi|qui|quand|où|pourquoi|comment)\b',  # 法语疑问词
            r'\b(was|wer|wann|wo|warum|wie)\b',  # 德语疑问词
        ]
        has_question_word = any(
            re.search(pattern, user_message, re.IGNORECASE) for pattern in question_patterns
        )
        
        # 3. 问题长度检测（短问题更可能是查询类问题）
        is_short_query = len(user_message.split()) <= 10
        
        # 如果包含问题特征，倾向于使用RAG搜索
        if has_question_mark or (has_question_word and is_short_query):
            return {
                "next_action": "rag_search",
                "selected_agent": "rag_agent" if "rag_agent" in self.agents else None,
                "routing_reason": "最终降级策略：基于通用问题模式检测",
                "confidence": 0.4  # 置信度很低
            }
        
        # 默认使用chat_agent
        return {
            "next_action": "chat",
            "selected_agent": "chat_agent" if "chat_agent" in self.agents else None,
            "routing_reason": "最终降级策略：默认使用chat_agent",
            "confidence": 0.3  # 置信度很低
        }

