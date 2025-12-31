"""Order Agent - 订单管理 Agent

处理与订单相关的查询和操作：
- 订单查询（列表、详情）
- 订单取消（需要用户确认）
- 订单创建（需要用户确认）
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage

from src.tools.order_tools import get_order_tools
from src.multi_agent.state import MultiAgentState
from src.multi_agent.utils import clean_messages_for_llm
from src.multi_agent.config import get_keywords_config
from src.confirmation import get_confirmation_manager, ConfirmationManager, ConfirmationStatus

logger = logging.getLogger(__name__)


# System Prompt
ORDER_AGENT_SYSTEM_PROMPT = """你是一个专业的电商客服助手 - 订单管理专家。

你的职责是帮助用户处理订单相关事务，包括：
1. 查询订单（列表、详情）
2. 取消订单（需要用户确认）
3. 创建新订单（需要用户确认）

工具使用指南：
- query_user_orders: 查询用户订单列表
  * user_phone: 用户手机号（必填）
  * status: 按状态筛选（可选）
  * limit: 返回数量限制

- query_order_detail: 查询订单详细信息
  * order_id: 订单ID（二选一）
  * order_number: 订单号（二选一）

取消订单流程（两步）：
1. prepare_cancel_order: 准备取消，显示确认信息
   * order_id: 订单ID
   * user_phone: 用户手机号
   * reason: 取消原因（可选）
2. confirm_cancel_order: 用户确认后执行取消
   * order_id: 订单ID
   * user_phone: 用户手机号

创建订单流程（两步）：
1. prepare_create_order: 准备创建，显示确认信息
   * user_phone: 用户手机号
   * items: 商品列表 JSON，如: [{"product_id": 1, "quantity": 2}]
   * notes: 备注（可选）
2. confirm_create_order: 用户确认后执行创建
   * 同上参数

重要注意事项：
- 取消和创建订单前必须先调用 prepare_* 方法展示确认信息
- 用户明确确认（说"确认"、"是"、"好的"等）后，才调用 confirm_* 方法
- 如果用户说"不"、"取消"等，则中止操作
- 需要用户提供手机号来验证身份

上下文理解（重要）：
- 用户可能分多轮提供信息（如先选择商品，后提供手机号、地址等）
- **必须仔细分析完整的对话历史**，从所有历史消息中提取用户已提供的信息
- **如果工具所需的参数（如 user_phone、items、order_id 等）在对话历史中已经出现过，必须直接使用，不要重复询问**
- 提取信息的优先级：
  1. 首先检查对话历史中用户明确提供的信息（如"我的手机号是138..."、"我要买3个"等）
  2. 其次检查上下文信息（entities）
  3. 如果都没有，才询问用户
- 特别关注任务链上下文：如果处于多步骤流程中，要结合之前的步骤结果来理解用户意图
- 示例：如果用户之前说过"我的手机号是13444444343"，后续需要手机号时，必须从历史消息中提取使用，不要再次询问

回复风格：
- 使用友好的语气，用 emoji 让回复更生动
- 涉及金额时精确到小数点后两位
- 操作完成后提供清晰的反馈
"""




class OrderAgent:
    """订单管理 Agent

    实现确认机制：
    1. prepare_* 操作后，通过 ConfirmationManager 创建待确认操作
    2. 用户回复后，ConfirmationManager 判断是否确认
    3. 确认后执行 confirm_* 操作

    确认机制支持跨请求持久化，用户可通过文本或 UI 按钮进行确认
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        tools: list | None = None,
        confirmation_manager: ConfirmationManager | None = None,
    ):
        """初始化 Order Agent

        Args:
            llm: LangChain LLM 实例
            tools: 订单工具列表，默认使用内置工具
            confirmation_manager: 确认管理器，默认使用全局单例
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )
        self.tools = tools or get_order_tools()
        self.name = "order_agent"
        self.confirmation_manager = confirmation_manager or get_confirmation_manager()

        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def get_name(self) -> str:
        """获取 Agent 名称"""
        return self.name

    def get_description(self) -> str:
        """获取 Agent 描述"""
        return "订单管理专家 - 处理订单查询、取消、创建等操作（含用户确认机制）"

    def _check_confirmation(self, user_input: str) -> bool | None:
        """检查用户输入是否为确认

        使用配置化的关键词列表，支持扩展和多语言。

        Args:
            user_input: 用户输入文本

        Returns:
            True: 确认
            False: 否认
            None: 无法判断（非确认相关输入）
        """
        user_input_lower = user_input.strip().lower()
        keywords_config = get_keywords_config()

        # 检查确认（使用配置化关键词）
        for keyword in keywords_config.confirm_yes_keywords:
            if keyword.lower() in user_input_lower:
                return True

        # 检查否认（使用配置化关键词）
        for keyword in keywords_config.confirm_no_keywords:
            if keyword.lower() in user_input_lower:
                return False

        return None

    def _get_entity(self, state: MultiAgentState, key: str, default: Any = None) -> Any:
        """从 state 中获取实体值

        Args:
            state: 多Agent状态
            key: 实体键名
            default: 默认值

        Returns:
            实体值
        """
        entities = state.entities
        return entities.get(key, default)

    def _find_order_id_from_context(self, state: MultiAgentState, messages: list) -> int | None:
        """从上下文中查找订单ID

        查找顺序：
        1. entities 中的 order_id
        2. agent_results 中的单一订单
        3. 消息历史中的 ToolMessage 中的单一订单

        Args:
            state: 多Agent状态
            messages: 消息列表

        Returns:
            订单ID，如果未找到返回 None
        """
        # 首先从 entities 中获取
        order_id = self._get_entity(state, "order_id")
        if order_id:
            return int(order_id)

        # 从 agent_results 中查找
        order_result = state.agent_results.get("order_agent", {})
        if isinstance(order_result, dict) and "orders" in order_result:
            orders = order_result.get("orders", [])
            if orders and len(orders) == 1:
                order_id = orders[0].get("id")
                logger.info(f"从 agent_results 获取到单一订单: id={order_id}")
                return order_id

        # 从消息历史中的 ToolMessage 查找
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                try:
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(tool_result, dict) and "orders" in tool_result:
                        orders = tool_result.get("orders", [])
                        if orders and len(orders) == 1:
                            order_id = orders[0].get("id")
                            logger.info(f"从历史消息获取到单一订单: id={order_id}")
                            return order_id
                except (json.JSONDecodeError, TypeError):
                    continue

        return None

    def _find_order_info_from_messages(self, messages: list, order_id: int) -> Dict[str, Any] | None:
        """从消息历史中查找指定订单的完整信息

        Args:
            messages: 消息列表
            order_id: 订单ID

        Returns:
            订单信息字典，如果未找到返回 None
        """
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                try:
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(tool_result, dict) and "orders" in tool_result:
                        orders = tool_result.get("orders", [])
                        for order in orders:
                            if order.get("id") == order_id or order.get("id") == int(order_id):
                                return order
                except (json.JSONDecodeError, TypeError):
                    continue
        return None

    def _parse_tool_result(self, result: str | Dict[str, Any]) -> Dict[str, Any]:
        """解析工具执行结果

        Args:
            result: 工具执行结果（可能是字符串或字典）

        Returns:
            解析后的字典
        """
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"text": result}
        return result if isinstance(result, dict) else {}

    def _get_tool(self, tool_name: str):
        """获取指定名称的工具

        Args:
            tool_name: 工具名称

        Returns:
            工具实例，如果未找到返回 None
        """
        return next((t for t in self.tools if t.name == tool_name), None)

    def _format_order_status_emoji(self, status: str) -> str:
        """格式化订单状态为带emoji的文本

        Args:
            status: 订单状态

        Returns:
            格式化后的状态文本
        """
        status_map = {
            "pending": "⏳ 待支付",
            "paid": "💰 已支付",
            "shipped": "🚚 已发货",
            "delivered": "✅ 已收货",
            "cancelled": "❌ 已取消",
        }
        return status_map.get(status, status)

    def _build_order_list_text(self, orders: list) -> str:
        """构建订单列表的文本描述

        Args:
            orders: 订单列表

        Returns:
            格式化的订单列表文本
        """
        if not orders:
            return "暂无订单"

        text = f"找到 {len(orders)} 个订单：\n"
        for order in orders:
            status_emoji = self._format_order_status_emoji(order.get("status", ""))
            text += f"\n订单号: {order.get('order_number')} - {status_emoji} - ¥{order.get('total_amount', 0):.2f}"
        return text

    def _detect_intent(self, content: str) -> Dict[str, bool]:
        """检测用户意图

        Args:
            content: 用户输入内容

        Returns:
            包含意图检测结果的字典
        """
        keywords_config = get_keywords_config()
        
        # 使用正则表达式模式匹配意图
        def _match_intent(patterns: List[str]) -> bool:
            """匹配意图：使用正则表达式模式匹配"""
            if not patterns:
                return False
            
            # 合并所有模式为一个正则表达式（用 | 连接）
            combined_pattern = '|'.join(patterns)
            return bool(re.search(combined_pattern, content, re.IGNORECASE))
        
        return {
            "is_query": _match_intent(
                getattr(keywords_config, 'query_order_patterns', [])
            ),
            "is_cancel": _match_intent(
                getattr(keywords_config, 'cancel_order_patterns', [])
            ),
        }

    async def _handle_query_intent(
        self, state: MultiAgentState, messages: list, content: str
    ) -> Dict[str, Any] | None:
        """处理查询订单意图

        Args:
            state: 多Agent状态
            messages: 消息列表
            content: 用户输入内容

        Returns:
            如果成功处理返回结果字典，否则返回 None
        """
        logger.info(f"检测到查询订单意图: {content[:50]}...")

        user_phone = self._get_entity(state, "user_phone")
        if not user_phone:
            logger.info("查询意图但缺少手机号，继续正常处理")
            return None

        query_tool = self._get_tool("query_user_orders")
        if not query_tool:
            logger.warning("未找到 query_user_orders 工具")
            return None

        try:
            query_result = await query_tool.ainvoke({
                "user_phone": user_phone,
                "status": None,
                "limit": 20
            })

            result_data = self._parse_tool_result(query_result)
            orders = result_data.get("orders", [])

            # 构建消息序列
            tool_call_id = f"call_query_{user_phone}_{hash(content) % 100000}"
            ai_message_with_tool = AIMessage(
                content="",
                tool_calls=[{
                    "id": tool_call_id,
                    "name": "query_user_orders",
                    "args": {"user_phone": user_phone, "status": None, "limit": 20}
                }]
            )

            tool_message = ToolMessage(content=query_result, tool_call_id=tool_call_id)
            order_text = self._build_order_list_text(orders)
            final_ai_message = AIMessage(content=order_text)

            logger.info(f"查询完成: 找到{len(orders)}个订单")
            for order in orders:
                logger.info(f"  - 订单ID: {order.get('id')}, 订单号: {order.get('order_number')}, 状态: {order.get('status')}")

            return {
                "messages": messages + [ai_message_with_tool] + [tool_message] + [final_ai_message],
                "current_agent": self.name,
                "tools_used": state.tools_used + [{
                    "agent": self.name,
                    "tool": "query_user_orders",
                    "args": {"user_phone": user_phone}
                }]
            }
        except Exception as e:
            logger.error(f"查询订单失败: {e}", exc_info=True)
            return None

    async def _handle_cancel_intent(
        self, state: MultiAgentState, messages: list, session_id: str, content: str
    ) -> Dict[str, Any] | None:
        """处理取消订单意图

        Args:
            state: 多Agent状态
            messages: 消息列表
            session_id: 会话ID
            content: 用户输入内容

        Returns:
            如果成功处理返回结果字典，否则返回 None
        """
        logger.info(f"检测到取消订单意图: {content[:50]}...")

        order_id = self._find_order_id_from_context(state, messages)
        user_phone = self._get_entity(state, "user_phone")

        if not order_id or not user_phone:
            logger.info(f"取消意图但缺少信息: order_id={order_id}, user_phone={user_phone}，使用 LLM 处理")
            return None

        logger.info(f"调用 prepare_cancel_order: order_id={order_id}, phone={user_phone}")

        order_info = self._find_order_info_from_messages(messages, order_id)
        prepare_tool = self._get_tool("prepare_cancel_order")

        if not prepare_tool:
            logger.warning("未找到 prepare_cancel_order 工具")
            return None

        try:
            prepare_result = await prepare_tool.ainvoke({
                "order_id": int(order_id),
                "user_phone": user_phone,
                "reason": "用户请求取消"
            })

            result_data = self._parse_tool_result(prepare_result)

            if not result_data.get("can_cancel", False):
                return {
                    "messages": messages + [AIMessage(content=result_data.get("text", "无法取消订单"))],
                    "current_agent": self.name,
                }

            display_message = result_data.get("text", "请确认是否取消订单")
            display_data = {
                "order_id": order_id,
                "order": order_info
            }

            confirmation = await self.confirmation_manager.request_confirmation(
                session_id=session_id,
                action_type="cancel_order",
                action_data={"order_id": int(order_id), "user_phone": user_phone},
                agent_name=self.name,
                display_message=display_message,
                display_data=display_data
            )

            logger.info(f"创建取消订单确认: confirmation_id={confirmation.confirmation_id}")

            return {
                "messages": messages + [AIMessage(content=display_message)],
                "current_agent": self.name,
                "confirmation_pending": {
                    "confirmation_id": confirmation.confirmation_id,
                    "action_type": "cancel_order",
                    "display_message": display_message,
                    "display_data": display_data
                },
                "tools_used": state.tools_used + [{
                    "agent": self.name,
                    "tool": "prepare_cancel_order",
                    "args": {"order_id": order_id, "user_phone": user_phone}
                }]
            }
        except Exception as e:
            logger.error(f"prepare_cancel_order 失败: {e}", exc_info=True)
            return None

    def _build_system_prompt_hints(self, state: MultiAgentState) -> str:
        """构建系统提示的上下文信息

        通用解决方案：只提供累积的上下文信息，不做硬编码的条件判断。
        LLM 会根据工具描述和这些上下文信息，自动判断是否可以执行工具，
        或者需要向用户询问什么信息。

        Args:
            state: 多Agent状态，从中提取所有可用的上下文信息

        Returns:
            提示文本
        """
        # 收集所有可用的实体信息
        all_entities = state.entities

        # 构建上下文提示，让 LLM 自己判断如何使用这些信息
        hints = []

        # 【关键】明确告诉 LLM 要从对话历史中提取信息
        hints.append("\n\n=== 重要提示：信息提取优先级 ===")
        hints.append("1. **首先检查对话历史**：仔细阅读所有历史消息，提取用户已明确提供的信息")
        hints.append("   - 用户可能在之前的对话中提供过手机号、数量、地址等信息")
        hints.append("   - 如果工具需要的参数在历史消息中已存在，必须直接使用，不要重复询问")
        hints.append("2. 其次检查以下上下文信息（如果已收集）：")

        if all_entities:
            hints.append("\n=== 已收集的上下文信息 ===")
            for key, value in all_entities.items():
                if value is not None:
                    hints.append(f"- {key}: {value}")
        else:
            hints.append("（当前没有已收集的上下文信息）")
        
        hints.append("\n请根据对话历史、上下文信息和工具描述，判断是否可以执行操作，或需要向用户询问什么信息。")
        
        # 如果有任务链，添加任务链上下文
        task_chain = state.task_chain
        if task_chain:
            current_index = task_chain.current_step_index
            steps = task_chain.steps
            if current_index < len(steps):
                current_step = steps[current_index]
                step_type = current_step.step_type
                if step_type == "order_creation":
                    hints.append("\n当前处于任务链的订单创建步骤。请根据对话历史和上下文判断是否可以创建订单，或需要用户提供什么信息。")
        
        return "\n".join(hints)

    async def _handle_with_llm(
        self,
        state: MultiAgentState,
        messages: list,
        session_id: str
    ) -> Dict[str, Any]:
        """使用 LLM 处理请求

        通用解决方案：不传递硬编码的参数，让 LLM 从 state 和上下文中自己获取信息。
        LLM 会根据工具描述和上下文，自动判断需要什么信息，并决定是调用工具还是询问用户。

        Args:
            state: 多Agent状态（包含所有上下文信息）
            messages: 消息列表
            session_id: 会话ID

        Returns:
            处理结果
        """
        hints = self._build_system_prompt_hints(state)
        # 保留更多历史消息，确保 LLM 能看到用户之前提供的信息（如手机号、数量等）
        # 清理消息历史，确保消息序列完整性（过滤无效的 ToolMessage）
        cleaned_messages = clean_messages_for_llm(messages, keep_recent_n=20)

        agent_messages = [
            SystemMessage(content=ORDER_AGENT_SYSTEM_PROMPT + hints)
        ]
        agent_messages.extend(cleaned_messages)

        logger.info(f"准备调用 LLM 处理请求，消息数量: {len(cleaned_messages)}")

        response = await self.llm_with_tools.ainvoke(agent_messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            result = await self._handle_llm_tool_calls(
                state, messages, session_id, agent_messages, response
            )
        else:
            result = {
                "messages": messages + [response],
                "current_agent": self.name,
            }

        return result

    async def _handle_llm_tool_calls(
        self,
        state: MultiAgentState,
        messages: list,
        session_id: str,
        agent_messages: list,
        response: AIMessage
    ) -> Dict[str, Any]:
        """处理 LLM 返回的工具调用

        Args:
            state: 多Agent状态
            messages: 消息列表
            session_id: 会话ID
            agent_messages: Agent消息列表
            response: LLM响应（包含工具调用）

        Returns:
            处理结果
        """
        needs_confirmation = False
        confirmation_data = None
        tool_messages = []
        tool_used_info = []
        order_info_dict = {}  # 用于保存 order_info（供任务链使用）

        for tool_call in response.tool_calls:
            tool = self._get_tool(tool_call["name"])
            if not tool:
                continue

            try:
                tool_result = await tool.ainvoke(tool_call["args"])

                # 检查是否需要确认
                if tool_call["name"] in ["prepare_cancel_order", "prepare_create_order"]:
                    needs_confirmation = True
                    parsed_result = self._parse_tool_result(tool_result)
                    confirmation_data = {
                        "action_type": tool_call["name"].replace("prepare_", ""),
                        "action_data": tool_call["args"],
                        "display_message": parsed_result.get("text", "请确认操作"),
                        "display_data": {
                            "items": parsed_result.get("items"),
                            "total_amount": parsed_result.get("total_amount"),
                            "order": parsed_result.get("order"),
                        },
                    }
                    
                    # 如果是 prepare_create_order，提取 order_info 供任务链使用
                    # 通用方案：从工具调用参数和结果中提取，不硬编码字段检查
                    if tool_call["name"] == "prepare_create_order":
                        order_info = {
                            "user_phone": tool_call["args"].get("user_phone"),
                            "items": tool_call["args"].get("items"),
                            "items_data": parsed_result.get("items"),
                            "total_amount": parsed_result.get("total_amount"),
                            "text": parsed_result.get("text", "订单信息已准备"),
                            "can_create": parsed_result.get("can_create", True)
                        }
                        # 保存到字典中，供任务链使用
                        order_info_dict["order_info"] = order_info

                tool_messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
                tool_used_info.append({
                    "agent": self.name,
                    "tool": tool_call["name"],
                    "args": tool_call["args"],
                })
            except Exception as e:
                tool_messages.append(
                    ToolMessage(content=f"错误: {str(e)}", tool_call_id=tool_call["id"])
                )

        followup_messages = agent_messages + [response] + tool_messages
        final_response = self.llm.invoke(followup_messages)

        result = {
            "messages": messages + [response] + tool_messages + [final_response],
            "current_agent": self.name,
            "tools_used": state.tools_used + tool_used_info,
        }

        # 如果提取了 order_info，添加到结果中（供任务链使用）
        if order_info_dict:
            result.update(order_info_dict)

        if needs_confirmation and confirmation_data:
            confirmation = await self.confirmation_manager.request_confirmation(
                session_id=session_id,
                action_type=confirmation_data["action_type"],
                action_data=confirmation_data["action_data"],
                agent_name=self.name,
                display_message=confirmation_data["display_message"],
                display_data=confirmation_data["display_data"],
            )

            result["confirmation_pending"] = {
                "confirmation_id": confirmation.confirmation_id,
                "action_type": confirmation.action_type,
                "display_message": confirmation.display_message,
                "display_data": confirmation.display_data,
            }

        return result

    async def execute(self, state: MultiAgentState, session_id: str = "default") -> Dict[str, Any]:
        """执行订单操作

        Args:
            state: 当前多 Agent 状态
            session_id: 用户会话 ID，用于确认机制

        Returns:
            更新后的状态片段
        """
        messages = state.messages
        if not messages:
            return {
                "messages": [
                    AIMessage(content="您好！我是订单管理助手。查询订单需要提供手机号，请问有什么可以帮您？")
                ],
                "current_agent": self.name,
            }

        # 获取最新消息
        latest_message = messages[-1]

        # 首先检查是否有待确认操作（通过 ConfirmationManager）
        pending_confirmation = await self.confirmation_manager.get_pending_confirmation(session_id)
        if pending_confirmation and pending_confirmation.agent_name == self.name:
            # 有待确认操作，检查用户输入是否为确认响应
            if hasattr(latest_message, "content"):
                user_input = latest_message.content
                result = await self.confirmation_manager.check_and_resolve_from_text(
                    session_id, user_input
                )

                if result:
                    if result.status == ConfirmationStatus.CONFIRMED:
                        # 用户确认，操作已执行
                        exec_result = result.execution_result or {}
                        execution_success = exec_result.get("success", False)
                        message = exec_result.get("text", "操作已完成")
                        if result.error:
                            message = f"操作执行失败: {result.error}"

                        logger.info(f"用户确认操作: action_type={result.action_type}, success={execution_success}")
                        if result.action_type == "cancel_order":
                            logger.info(f"取消订单结果: order_id={result.action_data.get('order_id')}, status={exec_result.get('order_status')}")
                        
                        # 【关键修复】根据执行结果决定是否清理 confirmation_pending
                        # 如果订单创建失败，保留 confirmation_pending，让 AI 能够继续处理错误
                        if execution_success:
                            # 执行成功：清理 confirmation_pending
                            return {
                                "messages": messages + [AIMessage(content=message)],
                                "current_agent": self.name,
                                "confirmation_pending": None,
                            }
                        else:
                            # 执行失败：保留 confirmation_pending，让 AI 继续处理错误
                            # 添加错误提示消息，引导用户重新下单
                            error_message = f"{message}\n\n订单创建出错了，需要重新下单吗？"
                            logger.warning(f"订单创建失败，保留 confirmation_pending 以便 AI 处理错误: session={session_id}")
                            return {
                                "messages": messages + [AIMessage(content=error_message)],
                                "current_agent": self.name,
                                # 不设置 confirmation_pending，保留原有的值
                            }
                    elif result.status == ConfirmationStatus.CANCELLED:
                        # 用户取消
                        return {
                            "messages": messages + [
                                AIMessage(content="👌 已取消操作，请问还有其他需要帮助的吗？")
                            ],
                            "current_agent": self.name,
                            "confirmation_pending": None,
                        }
                # result 为 None 表示用户输入不是确认响应
                # 【关键修复】check_and_resolve_from_text 已经自动取消了确认，
                # 但我们需要确保返回 confirmation_pending: None 以清理 state
                logger.info(f"用户输入不是确认响应，确认已自动取消，清理 confirmation_pending: session={session_id}")
                # 继续正常处理，但确保返回 confirmation_pending: None
        elif not pending_confirmation:
            # 【关键修复】如果没有待确认操作，但 state 中可能还有旧的 confirmation_pending
            # 确保返回 None 以清理 state
            state_confirmation = state.confirmation_pending
            if state_confirmation:
                logger.info(f"检测到 state 中有旧的 confirmation_pending，但 ConfirmationManager 中已无待确认操作，清理: session={session_id}")
                # 继续正常处理，但确保返回 confirmation_pending: None

        # 获取用户输入内容
        latest_content = latest_message.content if hasattr(latest_message, "content") else ""

        # 检测用户意图
        intent = self._detect_intent(latest_content)

        # 处理查询订单意图
        if intent["is_query"]:
            result = await self._handle_query_intent(state, messages, latest_content)
            if result:
                # 【关键修复】确保清理旧的 confirmation_pending
                if "confirmation_pending" not in result:
                    pending_confirmation = await self.confirmation_manager.get_pending_confirmation(session_id)
                    if not pending_confirmation:
                        result["confirmation_pending"] = None
                return result

        # 处理取消订单意图
        if intent["is_cancel"]:
            result = await self._handle_cancel_intent(state, messages, session_id, latest_content)
            if result:
                # _handle_cancel_intent 会返回 confirmation_pending（如果创建了新的确认）
                # 如果没有创建新的确认，确保清理旧的
                if "confirmation_pending" not in result:
                    pending_confirmation = await self.confirmation_manager.get_pending_confirmation(session_id)
                    if not pending_confirmation:
                        result["confirmation_pending"] = None
                return result

        # 使用 LLM 统一处理（包括任务链模式）
        # LLM 会根据工具描述和上下文自动判断需要什么信息
        result = await self._handle_with_llm(state, messages, session_id)
        
        # 【关键修复】确保如果没有创建新的确认，就清理旧的 confirmation_pending
        if "confirmation_pending" not in result:
            # 检查是否还有待确认操作
            pending_confirmation = await self.confirmation_manager.get_pending_confirmation(session_id)
            if not pending_confirmation:
                # 没有待确认操作，清理 state 中的 confirmation_pending
                result["confirmation_pending"] = None
        
        return result
