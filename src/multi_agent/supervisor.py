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
from pydantic import BaseModel, Field
from src.multi_agent.state import MultiAgentState
from src.multi_agent.agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class RoutingDecision(BaseModel):
    """路由决策结构定义

    使用Pydantic模型定义路由决策的输出结构，确保LLM输出符合预期格式。
    """
    next_action: Literal["rag_search", "chat", "product_search", "order_management", "tool_call", "execute_task_chain", "finish"] = Field(
        ...,
        description="下一步行动：rag_search表示需要RAG搜索，chat表示一般对话，product_search表示商品搜索，order_management表示订单管理，tool_call表示工具调用，execute_task_chain表示执行任务链，finish表示结束"
    )
    selected_agent: Optional[str] = Field(
        None,
        description="选中的Agent名称，如果next_action为finish则可以为null"
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
            # === 新增：多步骤任务编排支持 ===

            # 提取用户消息（提前提取，用于检查特殊消息）
            user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break

            # 0. 检查是否是任务链继续消息（优先级最高）
            if user_message and "__TASK_CHAIN_CONTINUE__" in user_message:
                logger.info("检测到任务链继续消息，直接路由到任务编排器")
                return {
                    "next_action": "execute_task_chain",
                    "selected_agent": None,
                    "routing_reason": "任务链继续执行",
                    "confidence": 1.0
                }

            # 1. 检查是否有活跃的任务链或待选择状态
            task_chain = state.get("task_chain")
            pending_selection = state.get("pending_selection")
            # 标记是否需要清理（用于在最终返回时携带清理状态）
            needs_cleanup = False

            if task_chain or pending_selection:
                logger.info(f"检测到活跃任务链或待选择状态: task_chain={task_chain is not None}, pending_selection={pending_selection is not None}")

                # === 核心逻辑：检测用户新输入是否与任务链/选择相关 ===
                should_clear_task_chain = False
                clear_reason = ""

                # 检查用户消息是否是选择/确认操作
                is_selection_response = False
                if user_message:
                    # 选择操作通常包含：数字选择、关键词确认、纯数字（如"1"、"2"）
                    selection_keywords = ["选择", "确认", "第", "1.", "2.", "3.", "4.", "5.", "1、", "2、", "3、", "4、", "5、"]
                    # 检查是否是纯数字（如 "1"、"2" 等）
                    is_pure_number = bool(re.match(r'^\d+$', user_message.strip()))
                    is_selection_response = is_pure_number or any(kw in user_message for kw in selection_keywords)

                    # 【关键修复】只有存在 pending_selection 且用户单独说"取消"（不是"取消订单"等）时，
                    # 才视为取消选择操作；否则"取消"可能是业务操作（如取消订单）
                    if pending_selection and not is_selection_response:
                        # 检查是否是单独的"取消"操作（取消选择）
                        cancel_selection_patterns = [
                            r'^取消$',           # 只说"取消"
                            r'^取消选择',        # 取消选择
                            r'^不选了',          # 不选了
                            r'^不要了',          # 不要了
                            r'^算了$',           # 算了
                        ]
                        is_selection_response = any(re.search(p, user_message.strip()) for p in cancel_selection_patterns)

                # 如果不是选择响应，检查是否是完全不相关的新问题
                if not is_selection_response:
                    # 【关键改进】检查用户意图是否与任务链匹配
                    query_intent = state.get("query_intent")
                    if query_intent:
                        intent_type = query_intent.get("intent_type", "")

                        # 如果用户意图是明确的订单查询/管理，且任务链是购买流程，清除任务链
                        is_order_query_intent = any(keyword in intent_type.lower() for keyword in ["order", "订单", "factual"])
                        is_purchase_task_chain = task_chain and task_chain.get("chain_type") == "order_with_search"

                        if is_order_query_intent and is_purchase_task_chain:
                            should_clear_task_chain = True
                            clear_reason = f"用户意图变化（{intent_type}），与购买流程不匹配"

                    # 检查任务链当前步骤
                    if task_chain and not should_clear_task_chain:
                        current_step_index = task_chain.get("current_step_index", 0)
                        steps = task_chain.get("steps", [])
                        if current_step_index < len(steps):
                            current_step = steps[current_step_index]
                            step_type = current_step.get("step_type")
                            step_status = current_step.get("status")

                            # 【关键修复】如果任务链在等待用户选择（pending 或 in_progress），
                            # 但用户的新输入不是选择操作，清除任务链
                            if step_type == "user_selection" and step_status in ["pending", "in_progress"]:
                                should_clear_task_chain = True
                                clear_reason = "用户跳过商品选择，发起新问题"

                    # 如果有 pending_selection 但用户不是在选择，也清除
                    if pending_selection and not should_clear_task_chain:
                        should_clear_task_chain = True
                        clear_reason = "用户跳过选择，发起新问题"

                # 执行清理
                if should_clear_task_chain:
                    logger.info(f"🧹 自动清理任务链和待选择状态: {clear_reason}")

                    # 清理 pending_selection（如果存在）
                    if pending_selection:
                        from src.confirmation.selection_manager import get_selection_manager
                        selection_manager = get_selection_manager()
                        try:
                            await selection_manager.cancel_selection(pending_selection.get("selection_id", ""))
                        except Exception as e:
                            logger.warning(f"清理 pending_selection 失败: {e}")

                    # 【关键改进】清理后不要直接返回 finish，而是继续执行后续路由逻辑
                    # 将 task_chain 和 pending_selection 设置为 None，然后让代码继续执行下去
                    # 这样用户的新问题会被正常路由到合适的 agent
                    logger.info("任务链已清理，继续执行正常路由流程")
                    # 标记需要清理，以便在最终返回时携带
                    needs_cleanup = True
                    task_chain = None  # 标记为清理
                    pending_selection = None  # 标记为清理

                else:
                    # 意图匹配，继续执行任务链
                    logger.info("用户输入与任务链匹配，继续执行任务链")
                    return {
                        "next_action": "execute_task_chain",
                        "selected_agent": None,
                        "routing_reason": "继续执行活跃的任务链",
                        "confidence": 1.0
                    }

            # 2. 检测是否需要创建新的任务链
            from src.multi_agent.task_orchestrator import get_task_orchestrator
            orchestrator = get_task_orchestrator()
            task_type = orchestrator.detect_multi_step_task(state)

            if task_type:
                logger.info(f"检测到多步骤任务: {task_type}")
                # 创建任务链
                new_task_chain = orchestrator.create_task_chain(task_type, state)
                return {
                    "next_action": "execute_task_chain",
                    "selected_agent": None,
                    "routing_reason": f"创建多步骤任务链: {task_type}",
                    "confidence": 0.9,
                    "task_chain": new_task_chain
                }

            # === 原有单步路由逻辑 ===

            # 检查是否有用户消息（user_message 已在前面提取）
            if not user_message:
                return {
                    "next_action": "finish",
                    "selected_agent": None,
                    "routing_reason": "未找到用户消息",
                    "confidence": 0.0
                }

            # 获取意图识别结果
            query_intent = state.get("query_intent")
            intent_context = self._build_intent_context(query_intent)

            # 构建路由提示词
            available_agents = self.get_available_agents()
            agents_description = "\n".join([
                f"- {agent['name']}: {agent['description']}"
                for agent in available_agents
            ])

            routing_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个智能路由系统，负责分析用户意图并决定调用哪个Agent。

**重要提示**：如果用户想要购买商品但没有提供具体的 product_id（如"我要下单，购买XX商品"），这应该由任务链系统处理，不要直接路由到 order_agent。

可用Agent列表：
{agents}

路由规则：
1. 商品相关：用户询问商品、搜索产品、比价等，选择 product_agent，next_action设为"product_search"
   - 关键词：商品、产品、手机、电脑、价格、多少钱、推荐、品牌
   - 示例："2000元以下的手机"、"华为笔记本有哪些"、"推荐一款性价比高的手机"

2. 订单相关：
   - **查询/取消订单**：选择 order_agent，next_action设为"order_management"
     - 示例："我的订单"、"取消订单123"
   - **创建订单**：
     * 如果用户提供了明确的 product_id，选择 order_agent
     * 如果用户只说"我要下单，购买XX商品"（没有 product_id），这应该由任务链处理，但任务链系统已经处理过了，这里不应该出现

3. 知识检索：如果用户问题需要从知识库中检索信息，选择 rag_agent，next_action设为"rag_search"
   - 示例："公司政策是什么"、"如何使用产品"

4. 一般对话：如果是一般性对话或简单问题，选择 chat_agent，next_action设为"chat"

5. 如果问题无法由现有Agent处理，next_action设为"finish"

请仔细分析用户问题，结合意图识别信息（如果有），做出最佳路由决策。"""),
                ("user", "用户问题: {question}\n\n{intent_context}")
            ])

            # 使用结构化输出的LLM进行路由决策
            # with_structured_output会自动确保输出符合RoutingDecision结构
            try:
                routing_decision = self.structured_llm.invoke(
                    routing_prompt.format_messages(
                        agents=agents_description,
                        question=user_message,
                        intent_context=intent_context
                    )
                )

                # 验证选中的Agent是否存在
                selected_agent = routing_decision.selected_agent
                if selected_agent and selected_agent not in self.agents:
                    logger.warning(f"选中的Agent {selected_agent} 不存在，使用chat_agent")
                    selected_agent = "chat_agent" if "chat_agent" in self.agents else None

                result = {
                    "next_action": routing_decision.next_action,
                    "selected_agent": selected_agent,
                    "routing_reason": routing_decision.routing_reason,
                    "confidence": routing_decision.confidence
                }

                # 如果需要清理任务链/待选择状态，添加到结果中
                if needs_cleanup:
                    result["task_chain"] = None
                    result["pending_selection"] = None
                    result["routing_reason"] = f"[清理旧任务链后] {result['routing_reason']}"

                logger.info(f"Supervisor路由决策: {result}")
                return result

            except Exception as e:
                logger.error(f"结构化输出解析失败: {e}, 使用降级策略", exc_info=True)
                # 企业级最佳实践：降级时也使用LLM，但用更简单的prompt和更便宜的模型
                return await self._fallback_routing_with_llm(user_message)

        except Exception as e:
            logger.error(f"Supervisor路由决策错误: {str(e)}", exc_info=True)
            # 企业级最佳实践：降级时也使用LLM
            return await self._fallback_routing_with_llm(user_message if 'user_message' in locals() else "")

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
            # 构建简化的路由提示词（降级策略使用更简单的prompt）
            available_agents = self.get_available_agents()
            agents_description = "\n".join([
                f"- {agent['name']}: {agent['description']}"
                for agent in available_agents
            ])
            
            # 简化的prompt，提高响应速度
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个路由系统。快速分析用户问题，决定调用哪个Agent。

可用Agent：
{agents}

规则：
- 商品搜索 → product_agent (next_action: "product_search")
- 订单管理 → order_agent (next_action: "order_management")
- 知识检索 → rag_agent (next_action: "rag_search")
- 一般对话 → chat_agent (next_action: "chat")
- 无法处理 → finish

快速决策。"""),
                ("user", "问题: {question}")
            ])
            
            # 使用更便宜的模型进行降级路由
            routing_decision = self.fallback_structured_llm.invoke(
                simple_prompt.format_messages(
                    agents=agents_description,
                    question=user_message
                )
            )
            
            # 验证选中的Agent是否存在
            selected_agent = routing_decision.selected_agent
            if selected_agent and selected_agent not in self.agents:
                logger.warning(f"降级策略选中的Agent {selected_agent} 不存在，使用chat_agent")
                selected_agent = "chat_agent" if "chat_agent" in self.agents else None
            
            result = {
                "next_action": routing_decision.next_action,
                "selected_agent": selected_agent,
                "routing_reason": f"降级策略（LLM）: {routing_decision.routing_reason}",
                "confidence": routing_decision.confidence * 0.8  # 降级策略的置信度稍低
            }
            
            logger.info(f"降级策略路由决策: {result}")
            return result
            
        except Exception as e:
            logger.error(f"降级策略LLM路由失败: {e}, 使用最终降级方案", exc_info=True)
            # 最终降级：如果LLM也失败，使用简单的启发式规则
            return self._final_fallback_routing(user_message)
    
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

