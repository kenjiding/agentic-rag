"""Supervisor Agent - 监督者Agent，负责路由和协调

本模块实现了Supervisor Agent，它是多Agent系统的核心协调者。
Supervisor负责分析用户意图，决定调用哪个Agent或工具。

2025-2026 最佳实践（一步一步智能模式）：
- 基于当前状态（entities）进行智能路由决策
- 每次请求都重新进行意图识别和路由
- 支持动态Agent注册
- 提供路由决策的可解释性
- 错误处理和降级策略
- 支持AgentRegistry集成
"""
import json
import re
from typing import Dict, Any, Optional, List, Literal, TYPE_CHECKING, Set
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator
from src.multi_agent.state import MultiAgentState, ConversationPhase
from src.multi_agent.constants import ActionName, AgentName
from src.multi_agent.routing_engine import RoutingEngine
from src.multi_agent.agents.base_agent import BaseAgent
from src.multi_agent.config import get_keywords_config
from src.utils.llm_factory import create_llm_for_agent
import logging

if TYPE_CHECKING:
    from src.multi_agent.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


class RoutingDecision(BaseModel):
    """路由决策结构定义

    使用Pydantic模型定义路由决策的输出结构，确保LLM输出符合预期格式。

    关键约束（双向逻辑一致性）：
    1. 当next_action为"finish"时，selected_agent必须为None
    2. 当selected_agent不为None时，next_action不能为"finish"
    """
    next_action: Literal["rag_search", "chat", "product_search", "order_management", "consultation", "finish"] = Field(
        ...,
        description="下一步行动：rag_search表示需要RAG搜索，chat表示一般对话，product_search表示商品搜索，order_management表示订单管理，consultation表示深度咨询（产品对比、适配性确认等），finish表示结束。注意：如果设置为finish，则selected_agent必须为null。"
    )
    selected_agent: Optional[Literal["rag_agent", "chat_agent", "product_agent", "order_agent", "consultation_agent"]] = Field(
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
        llm: Optional[BaseChatModel] = None,
        agents: Optional[List[BaseAgent]] = None,
        fallback_llm: Optional[BaseChatModel] = None,
        agent_registry: Optional["AgentRegistry"] = None
    ):
        """
        初始化Supervisor

        Args:
            llm: 语言模型实例，用于路由决策，如果为None则使用工厂函数创建默认模型
            agents: 可用的Agent列表
            fallback_llm: 降级策略使用的LLM（可选，如果为None则使用更便宜的模型）
            agent_registry: Agent注册表（可选，用于获取Agent描述）
        """
        self.llm = llm or create_llm_for_agent()
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_registry = agent_registry
        self.routing_engine = RoutingEngine()

        # 创建结构化输出的LLM（使用with_structured_output）
        self.structured_llm = self.llm.with_structured_output(RoutingDecision)

        # 降级策略使用的LLM（使用更便宜的模型，降低成本）
        self.fallback_llm = fallback_llm or create_llm_for_agent(
            model_name="openai:gpt-3.5-turbo",
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

    def set_agent_registry(self, registry: "AgentRegistry"):
        """设置Agent注册表

        允许在初始化后设置注册表，用于从注册表获取Agent描述。

        Args:
            registry: Agent注册表实例
        """
        self.agent_registry = registry
        logger.info("Supervisor设置Agent注册表")

    def get_available_agents(self) -> List[Dict[str, str]]:
        """获取可用Agent列表及其描述

        优先从AgentRegistry获取，如果未设置则从本地agents字典获取。
        """
        # 如果有注册表，使用注册表获取Agent描述
        if self.agent_registry:
            return [
                {
                    "name": descriptor.name,
                    "description": descriptor.description
                }
                for descriptor in self.agent_registry.get_enabled_agents()
            ]

        # 否则从本地agents字典获取
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
                    "next_action": ActionName.FINISH,
                    "selected_agent": None,
                    "routing_reason": "未找到用户消息",
                    "confidence": 0.0
                }

            # 1) 先走规则路由（可测试、可解释）
            rule_result = self.routing_engine.route(state)
            if rule_result:
                return rule_result

            # 2) LLM 路由兜底
            llm_result = await self._do_llm_routing(state, user_message)
            return llm_result

        except Exception as e:
            logger.error(f"Supervisor路由决策错误: {str(e)}", exc_info=True)
            return await self._fallback_routing_with_llm(self._extract_user_message(state) or "")

    def _build_intent_context(self, query_intent: Optional[Dict[str, Any]]) -> str:
        """构建意图识别上下文信息"""
        if not query_intent:
            return "（无意图识别信息）"

        context_parts = []

        intent_type = query_intent.get("intent_type", "unknown")
        complexity = query_intent.get("complexity", "unknown")
        context_parts.append(f"意图类型: {intent_type}")
        context_parts.append(f"复杂度: {complexity}")

        reasoning = query_intent.get("reasoning")
        if reasoning:
            context_parts.append(f"意图推理: {reasoning}")

        intent_entities = query_intent.get("entities")
        if intent_entities:
            if hasattr(intent_entities, "model_dump"):
                entities_dict = intent_entities.model_dump(exclude_none=True)
            elif isinstance(intent_entities, dict):
                entities_dict = intent_entities
            else:
                entities_dict = {}

            if entities_dict:
                compact = {k: v for k, v in entities_dict.items() if v not in (None, [], "")}
                if compact:
                    context_parts.append(f"意图实体: {compact}")

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

        if not all_entities:
            formatted_bundle = self._format_context_bundle(state.context_bundle)
            return formatted_bundle if formatted_bundle else "（无累积实体信息）"

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

        # 检查是否有 product_id 或 product_ids（用户已选定产品）
        product_id = all_entities.get("product_id")
        product_ids = all_entities.get("product_ids")
        has_product_id = bool(product_id)
        has_product_ids = bool(product_ids) and isinstance(product_ids, list) and len(product_ids) >= 2
        
        if has_product_ids:
            context_parts.append(f"  ✓ 已识别多个产品ID (product_ids={product_ids}，共{len(product_ids)}个)")
        elif has_product_id:
            context_parts.append(f"  ✓ 用户已选定产品 (product_id={product_id})")
        else:
            context_parts.append("  ✗ 用户未选定产品 (product_id和product_ids都不存在)")

        # 检查是否有 order_id（订单相关操作）
        order_id = all_entities.get("order_id")
        if not order_id:
            # 尝试从历史消息和状态中提取订单信息（优先从state.response_data提取）
            order_id_from_history = self._extract_order_id_from_messages(state.messages, state=state)
            if order_id_from_history:
                context_parts.append(f"  ✓ 从历史消息/状态中提取到订单信息: {order_id_from_history}")
                # 补充到实体信息中，供路由决策使用
                all_entities["order_id"] = order_id_from_history
                order_id = order_id_from_history
        
        if order_id:
            context_parts.append(f"  ✓ 已识别订单ID (order_id={order_id})")
        else:
            context_parts.append("  ✗ 未识别到订单ID")

        # 【路由决策逻辑】帮助 LLM 做出正确的路由决策
        context_parts.append("\n【路由决策逻辑】")
        has_product_id = bool(all_entities.get("product_id"))
        product_ids = all_entities.get("product_ids")
        has_product_ids = bool(product_ids) and isinstance(product_ids, list) and len(product_ids) >= 2
        has_order_id = bool(order_id)
        
        # 产品对比场景判断
        if has_product_ids:
            context_parts.append(f"  ✓ 已识别多个产品ID（product_ids={product_ids}，共{len(product_ids)}个）")
            context_parts.append("  → 可以路由到 consultation_agent 进行产品对比")
        elif not has_product_id and not has_product_ids:
            context_parts.append("  ⚠️ 关键：用户未选定产品（product_id=None，product_ids=None）")
            context_parts.append("  → 必须先路由到 product_agent 搜索产品！")
        elif has_product_id:
            context_parts.append("  ✓ 用户已选定产品（单个product_id）")
            context_parts.append("  → 可以路由到 order_agent 创建订单（用户已登录，无需手机号）")
        
        # 【订单管理路由逻辑】（新增，关键）
        if has_order_id:
            context_parts.append("  ✓ 已识别订单ID（order_id存在）")
            context_parts.append("  → **必须路由到 order_agent 处理订单相关操作（查询/取消等）！**")
            context_parts.append(f"  → 订单ID: {order_id}（已从entities或历史消息中提取）")

        # 累积实体信息（详细）
        if all_entities:
            context_parts.append("\n【累积实体详细信息】")
            for key, value in all_entities.items():
                if value is not None:
                    context_parts.append(f"  - {key}: {value}")

        return "\n".join(context_parts)
        
    def _build_message_context(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        分组原则：
        - 尽量把连续的对话放在同一个组
        - 只有在「完成了一轮工具调用」（AI调用过工具且所有ToolMessage都回来了）之后，
          再遇到新的HumanMessage，才认为开启了新话题，开始新组
        """
        if not messages:
            return []

        groups = []
        current = None
        pending_tool_ids = set()
        last_round_has_completed_tool = False   # 上一轮是否完整结束工具调用

        for msg in messages:
            if isinstance(msg, HumanMessage):
                # 关键判断：只有上一轮工具调用完整结束，才切新组
                if current is not None and last_round_has_completed_tool:
                    groups.append(current)
                    current = None
                    pending_tool_ids.clear()
                    last_round_has_completed_tool = False

                if current is None:
                    current = {
                        "human_messages": [],
                        "ai_messages": [],
                        "tool_messages": []
                    }

                current["human_messages"].append(msg.content)

            elif isinstance(msg, AIMessage) and current is not None:
                tool_calls = getattr(msg, "tool_calls", []) or []

                current["ai_messages"].append({
                    "content": msg.content or "",
                    "tool_calls": [
                        {"name": tc.get("name", ""), "args": tc.get("args", {})}
                        for tc in tool_calls
                    ]
                })

                for tc in tool_calls:
                    if tc_id := tc.get("id"):
                        pending_tool_ids.add(tc_id)

                # 如果这次 AI 没有调用工具，也算一种“完成”
                if not tool_calls:
                    last_round_has_completed_tool = True

            elif isinstance(msg, ToolMessage) and current is not None:
                if tool_call_id := getattr(msg, "tool_call_id", None):
                    if tool_call_id in pending_tool_ids:
                        # ... 工具结果处理逻辑保持不变 ...
                        # （省略，你原来的处理代码就很好）
                        pending_tool_ids.discard(tool_call_id)

                # 所有待处理的 tool call 都回来了 → 本轮工具调用完整结束
                if not pending_tool_ids:
                    last_round_has_completed_tool = True

        if current:
            groups.append(current)

        return self._format_conversation_groups(groups, max_groups=10)

    def _format_conversation_groups(self, groups: List[Dict[str, Any]], max_groups: int = 3) -> str:
        """
        将提取出的对话组格式化为可读的上下文字符串（用于塞到prompt里）
        """
        if not groups:
            return ""

        context_parts = ["\n【对话历史上下文】"]

        # 只展示最近的几组
        display_groups = groups[-max_groups:] if len(groups) > max_groups else groups

        for idx, group in enumerate(display_groups, 1):
            context_parts.append(f"\n--- 对话组 {idx} ---")

            # 人类消息（可能多条）
            human_text = "\n".join(
                f"用户: {h}" for h in group["human_messages"]
            )
            context_parts.append(human_text)

            # AI 回复 & 工具调用
            for ai_msg in group["ai_messages"]:
                if ai_msg["content"]:
                    content = ai_msg["content"]
                    if len(content) > 200:
                        content = content[:200] + "..."
                    context_parts.append(f"AI回复: {content}")

                for tool_call in ai_msg["tool_calls"]:
                    name = tool_call["name"]
                    args = tool_call["args"]
                    context_parts.append(f"调用工具: {name}")
                    if args:
                        # 只展示关键参数（可按业务调整）
                        key_args = {k: v for k, v in args.items()
                                  if k in ["search_keyword", "product_id", "product_ids", "order_id", "quantity"]}
                        if key_args:
                            context_parts.append(f"参数: {key_args}")

            # 工具返回结果（有针对性的摘要展示）
            for tool_msg in group["tool_messages"]:
                result = tool_msg.get("result", {})
                if not isinstance(result, dict):
                    continue

                if "products" in result and isinstance(result["products"], list):
                    products = result["products"]
                    if products:
                        context_parts.append(f"工具返回: 搜索到 {len(products)} 个产品")
                        for p in products[:3]:
                            name = p.get("name", "N/A")
                            pid = p.get("id") or p.get("product_id", "N/A")
                            price = p.get("price", "")
                            price_str = f" ¥{price}" if price else ""
                            context_parts.append(f" - ID:{pid} {name}{price_str}")
                        if len(products) > 3:
                            context_parts.append(f" ... 还有 {len(products)-3} 个产品")

                elif "product" in result and isinstance(result["product"], dict):
                    p = result["product"]
                    name = p.get("name", "N/A")
                    pid = p.get("id") or p.get("product_id", "N/A")
                    context_parts.append(f"工具返回: 产品详情 - ID:{pid} {name}")

                # 可继续补充其他业务类型的摘要...

        if len(groups) > max_groups:
            context_parts.append(
                f"\n（仅显示最近 {max_groups} 组对话，共 {len(groups)} 组）"
            )

        return "\n".join(context_parts)
        
    def _extract_order_id_from_messages(self, messages: List, state: Optional[Any] = None) -> Optional[str]:
        """从历史消息和状态中提取订单ID
        
        查找顺序：
        1. 从 state.response_data 中查找订单信息（如果提供了 state）
        2. 从最近的 ToolMessage 中查找订单信息（order或orders字段）
        3. 从最近的 AIMessage 中查找订单号模式（如ORD906278）
        4. 从最近的 HumanMessage 中查找订单号模式（如ORD906278）
        
        Args:
            messages: 消息列表
            state: 可选的多Agent状态，用于访问 response_data
            
        Returns:
            订单ID（字符串格式），如果未找到返回None
        """
        import json
        import re
        from langchain_core.messages import AIMessage
        
        # 辅助函数：从订单字典中提取订单号
        def extract_order_number(order: dict) -> Optional[str]:
            """从订单字典中提取订单号"""
            return order.get("order_number") or order.get("order_id")
        
        # 辅助函数：从订单列表中提取订单号（优先返回单一订单，否则返回最新订单）
        def extract_from_orders(orders: List[dict]) -> Optional[str]:
            """从订单列表中提取订单号"""
            if not orders:
                return None
            if len(orders) == 1:
                return extract_order_number(orders[0])
            # 多个订单时，返回最新的（最后一个）
            return extract_order_number(orders[-1])
        
        # 辅助函数：从文本中提取订单号模式
        def extract_from_text(content: str) -> Optional[str]:
            """从文本中提取订单号模式"""
            if not content:
                return None
            
            # 优先匹配包含ORD的模式（返回完整的ORD+数字）
            # 匹配 "ORD424929" 或 "订单号: ORD424929" 等格式
            ord_pattern = r'ORD\s*(\d+)'
            match = re.search(ord_pattern, content, re.IGNORECASE)
            if match:
                # 提取数字部分，然后组合成完整的订单号
                digits = match.group(1)
                return f"ORD{digits}"
            
            # 其次匹配纯数字订单号模式（至少6位）
            digit_pattern = r'订单[号]?\s*[:：]?\s*(\d{6,})'
            match = re.search(digit_pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
            
            return None
        
        # 优先级1：从 state.response_data 中查找（最可靠）
        if state and hasattr(state, 'response_data') and state.response_data:
            response_data = state.response_data
            # 检查单个订单
            if "order" in response_data and isinstance(response_data["order"], dict):
                order_number = extract_order_number(response_data["order"])
                if order_number:
                    logger.info(f"从state.response_data（订单详情）提取到订单号: {order_number}")
                    return str(order_number)
            # 检查订单列表
            if "orders" in response_data:
                orders = response_data.get("orders", [])
                order_number = extract_from_orders(orders)
                if order_number:
                    logger.info(f"从state.response_data（订单列表）提取到订单号: {order_number}")
                    return str(order_number)
        
        # 优先级2-4：在一个循环中按优先级顺序处理消息
        # 消息类型优先级：ToolMessage > AIMessage > HumanMessage
        message_type_priority = {
            'ToolMessage': 2,
            'AIMessage': 3,
            'HumanMessage': 4
        }
        
        # 按优先级分组消息（相同优先级的消息保持时间顺序）
        prioritized_messages = []
        for msg in reversed(messages):  # 从最新到最旧
            msg_class_name = msg.__class__.__name__ if hasattr(msg, '__class__') else str(type(msg))
            priority = message_type_priority.get(msg_class_name, 999)  # 未知类型优先级最低
            prioritized_messages.append((priority, msg_class_name, msg))
        
        # 按优先级排序（优先级数字越小越优先）
        prioritized_messages.sort(key=lambda x: x[0])
        
        # 在一个循环中按优先级顺序处理
        for priority, msg_class_name, msg in prioritized_messages:
            # 优先级2：ToolMessage（工具返回的JSON数据）
            if msg_class_name == 'ToolMessage':
                try:
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(tool_result, dict):
                        # 优先查找单个订单
                        if "order" in tool_result and isinstance(tool_result["order"], dict):
                            order_number = extract_order_number(tool_result["order"])
                            if order_number:
                                logger.info(f"从ToolMessage（订单详情）提取到订单号: {order_number}")
                                return str(order_number)
                        # 其次查找订单列表
                        if "orders" in tool_result:
                            orders = tool_result.get("orders", [])
                            order_number = extract_from_orders(orders)
                            if order_number:
                                logger.info(f"从ToolMessage（订单列表）提取到订单号: {order_number}")
                                return str(order_number)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    continue
            
            # 优先级3：AIMessage（AI回复的文本）
            elif msg_class_name == 'AIMessage' or isinstance(msg, AIMessage):
                content = str(getattr(msg, 'content', ''))
                order_id = extract_from_text(content)
                if order_id:
                    logger.info(f"从AIMessage提取到订单号: {order_id}")
                    return order_id
            
            # 优先级4：HumanMessage（用户输入）
            elif msg_class_name == 'HumanMessage':
                content = str(getattr(msg, 'content', ''))
                order_id = extract_from_text(content)
                if order_id:
                    logger.info(f"从HumanMessage提取到订单号: {order_id}")
                    return order_id
        
        return None

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

    def _normalize_agent(self, agent_name: Optional[str]) -> Optional[AgentName]:
        """验证并返回有效的 Agent 名称"""
        if not agent_name:
            return None
        if agent_name not in self.agents:
            logger.warning(f"选中的 Agent {agent_name} 不存在，使用 chat_agent")
            return AgentName.CHAT_AGENT if AgentName.CHAT_AGENT.value in self.agents else None
        return AgentName(agent_name)

    def _normalize_action(self, action: str) -> ActionName:
        return ActionName(action)

    def _extract_user_message(self, state: MultiAgentState) -> Optional[str]:
        """从状态中提取最新的用户消息"""
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None

    async def _do_llm_routing(self, state: MultiAgentState, user_message: str) -> Dict[str, Any]:
        """
        执行 LLM 单步路由（一步一步智能模式）

        改进说明：
        - user_message 只包含当前查询（最后一条 HumanMessage）
        - 历史上下文由 context_bundle 提供
        """
        query_intent = state.query_intent
        intent_context = self._build_intent_context(query_intent)

        # 【改进】优先使用统一上下文包
        entity_context = self._format_context_bundle(state.context_bundle)

        agents_description = self._get_agents_description()

        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是路由决策助手。系统已通过规则引擎处理确定性路由。
你的任务：仅在规则无法匹配时，基于上下文选择最合适的Agent。

可用Agent：
{agents}

要求：
1. 输出必须符合RoutingDecision结构。
2. 若无法判断，优先选择chat_agent而非finish。
3. 仅使用上下文中的事实，不要自行假设。

上下文：
{entity_context}

意图识别（仅供参考）：
{intent_context}"""),
            ("human", "用户问题: 我想买一台65寸电视，有什么推荐？"),
            ("assistant", """{
  "next_action": "product_search",
  "selected_agent": "product_agent",
  "routing_reason": "用户明确表达购买需求且未指定产品，需先搜索产品",
  "confidence": 0.75
}"""),
            ("human", "用户问题: 帮我取消订单ORD123456"),
            ("assistant", """{
  "next_action": "order_management",
  "selected_agent": "order_agent",
  "routing_reason": "包含订单号且是取消需求，需订单管理",
  "confidence": 0.8
}"""),
            ("human", "用户问题: 这两个型号X1和X2有什么区别？"),
            ("assistant", """{
  "next_action": "consultation",
  "selected_agent": "consultation_agent",
  "routing_reason": "涉及两个产品对比，需咨询/对比能力",
  "confidence": 0.7
}"""),
            ("human", "用户问题: 谢谢你！"),
            ("assistant", """{
  "next_action": "chat",
  "selected_agent": "chat_agent",
  "routing_reason": "普通致谢/闲聊，走通用对话",
  "confidence": 0.6
}"""),
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

            selected_agent = self._normalize_agent(routing_decision.selected_agent)
            next_action = self._normalize_action(routing_decision.next_action)

            result = {
                "next_action": next_action,
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

            selected_agent = self._normalize_agent(routing_decision.selected_agent)
            next_action = self._normalize_action(routing_decision.next_action)

            # 防御性检查
            if next_action == ActionName.FINISH and selected_agent is not None:
                logger.warning(
                    f"降级策略检测到逻辑不一致：next_action='finish'但selected_agent={selected_agent}，"
                    f"强制将selected_agent设置为None"
                )
                selected_agent = None

            result = {
                "next_action": next_action,
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
                "next_action": ActionName.RAG_SEARCH,
                "selected_agent": AgentName.RAG_AGENT if AgentName.RAG_AGENT.value in self.agents else None,
                "routing_reason": "最终降级策略：基于通用问题模式检测",
                "confidence": 0.4
            }

        # 默认使用chat_agent
        return {
            "next_action": ActionName.CHAT,
            "selected_agent": AgentName.CHAT_AGENT if AgentName.CHAT_AGENT.value in self.agents else None,
            "routing_reason": "最终降级策略：默认使用chat_agent",
            "confidence": 0.3
        }

    def _format_context_bundle(
        self,
        context_bundle: Optional[Dict[str, Any]],
    ) -> str:
        """格式化统一上下文包，必要时回退到summary."""
        if not context_bundle:
            return "（无上下文信息）"

        parts = ["【统一上下文包】"]

        short_term = context_bundle.get("short_term_context", {})
        task_input = context_bundle.get("task_input", {})

        # 对话阶段
        phase = task_input.get("conversation_phase")
        if phase:
            parts.append(f"【对话阶段】: {phase}")

        # 关键实体
        entities = task_input.get("entities", {})
        if entities:
            parts.append("【关键实体】:")
            for k, v in entities.items():
                parts.append(f"  - {k}: {v}")

        # 来自上下文摘要的关键实体（避免entities缺失时丢信息）
        key_entities = short_term.get("key_entities", {})
        if key_entities:
            parts.append("【摘要关键实体】:")
            for k, v in key_entities.items():
                parts.append(f"  - {k}: {v}")

        # 最近对话（从short_term_context复用）
        history = short_term.get("conversation_history", [])
        if history:
            parts.append(f"\n【对话历史】(最近{len(history)}轮):")
            for idx, turn in enumerate(history[-3:], 1):
                parts.append(f"\n  轮次{idx}:")
                if turn.get("human"):
                    human_msg = turn['human']
                    if len(human_msg) > 100:
                        human_msg = human_msg[:100] + "..."
                    parts.append(f"    用户: {human_msg}")
                if turn.get("ai"):
                    ai_msg = turn['ai']
                    if len(ai_msg) > 100:
                        ai_msg = ai_msg[:100] + "..."
                    parts.append(f"    AI: {ai_msg}")

        tool_calls = short_term.get("recent_tool_calls", [])
        if tool_calls:
            parts.append("\n【最近工具调用】:")
            for tc in tool_calls[-5:]:
                name = tc.get('name', 'unknown')
                summary = tc.get('summary', '')
                parts.append(f"  - {name}: {summary}")

        # 意图信息（仅供路由参考）
        intent = task_input.get("intent")
        if intent:
            parts.append("\n【意图信息】:")
            if isinstance(intent, dict):
                intent_type = intent.get("intent_type")
                complexity = intent.get("complexity")
                if intent_type:
                    parts.append(f"  - intent_type: {intent_type}")
                if complexity:
                    parts.append(f"  - complexity: {complexity}")

        return "\n".join(parts) if parts else "（无上下文信息）"
