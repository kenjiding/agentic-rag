"""Order Agent - 订单管理 Agent

处理与订单相关的查询和操作：
- 订单查询（列表、详情）
- 订单取消（需要用户确认）
- 订单创建（需要用户确认）
"""

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt

from src.tools.order_tools import get_order_tools
from src.utils.llm_factory import create_llm_for_agent
from src.multi_agent.state import MultiAgentState
from src.multi_agent.utils import clean_messages_for_llm
from src.multi_agent.response_models import OrderListResponse, TextResponse, ConfirmationResponse
from src.confirmation import get_confirmation_manager, ConfirmationManager, ConfirmationStatus
from src.multi_agent.interrupt_framework import (
    create_confirmation_interrupt,
    is_resume_confirm,
)

logger = logging.getLogger(__name__)


# System Prompt
ORDER_AGENT_SYSTEM_PROMPT = """你是一个专业的电商客服助手 - 订单管理专家。

你的职责是帮助用户处理订单相关事务，包括：
1. 查询订单（列表、详情）
2. 取消订单（需要用户确认）
3. 创建新订单（需要用户确认）

工具使用指南：
- query_order: 统一的订单查询工具
  * user_id: 系统会自动填充为当前 session_id，你不需要提供此参数（工具定义要求，但会被系统自动覆盖）
  * order_id: 订单ID（可选），支持订单号格式如ORD123456或纯数字字符串如'123'
    - 如果提供 order_id：优先查询特定订单详情，并验证订单归属权限
    - 如果不提供 order_id：查询用户所有订单列表
  * status: 订单状态筛选（仅在查询所有订单时生效，可选）
  * limit: 返回结果数量限制（仅在查询所有订单时生效）

取消订单流程（两步）：
1. prepare_cancel_order: 准备取消，显示确认信息
   * order_id: 订单ID（必填）
   * reason: 取消原因（可选）
   * user_id: 系统会自动填充为当前 session_id，你不需要提供此参数
   * 注意：系统会自动验证订单归属
2. confirm_cancel_order: 用户确认后执行取消
   * order_id: 订单ID（必填）
   * user_id: 系统会自动填充为当前 session_id，你不需要提供此参数
   * 注意：系统会自动验证订单归属

创建订单流程（两步）：
1. prepare_create_order: 准备创建，显示确认信息
   * items: 商品列表 JSON，如: [{"product_id": 1, "quantity": 2}]
   * notes: 备注（可选）
   * user_id: 系统会自动填充为当前 session_id，你不需要提供此参数（工具定义要求，但会被系统自动覆盖）
   * 注意：系统会自动识别用户身份
2. confirm_create_order: 用户确认后执行创建
   * items: 商品列表 JSON
   * notes: 备注（可选）
   * user_id: 系统会自动填充为当前 session_id，你不需要提供此参数

重要注意事项：
- 取消和创建订单前必须先调用 prepare_* 方法展示确认信息
- 用户明确确认（说"确认"、"是"、"好的"等）后，才调用 confirm_* 方法
- 如果用户说"不"、"取消"等，则中止操作
- **用户已登录，无需提供手机号，系统会自动从session中获取用户信息**

上下文理解（重要）：
- 用户可能分多轮提供信息（如先选择商品，后确认购买等）
- **必须仔细分析完整的对话历史**，从所有历史消息中提取用户已提供的信息
- **如果工具所需的参数（如 items、order_id 等）在对话历史中已经出现过，必须直接使用，不要重复询问**
- 提取信息的优先级：
  1. 首先检查对话历史中用户明确提供的信息（如"我要买3个"、"订单ID是123"等）
  2. 其次检查上下文信息（entities，如 product_id、order_id 等）
  3. 如果都没有，才询问用户
- 特别关注任务链上下文：如果处于多步骤流程中，要结合之前的步骤结果来理解用户意图

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
        llm: BaseChatModel | None = None,
        tools: list | None = None,
        confirmation_manager: ConfirmationManager | None = None,
    ):
        """初始化 Order Agent

        Args:
            llm: LangChain LLM 实例，如果为None则使用工厂函数创建默认模型
            tools: 订单工具列表，默认使用内置工具
            confirmation_manager: 确认管理器，默认使用全局单例
        """
        self.llm = llm or create_llm_for_agent(temperature=0.7)
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

    async def _build_error_response(
        self,
        messages: list,
        content: str,
        session_id: str,
        conversation_phase: str = "idle",
        cleanup_confirmation: bool = True
    ) -> Dict[str, Any]:
        """构建错误/取消响应的通用方法
        
        Args:
            messages: 当前消息列表
            content: 响应内容
            session_id: 用户会话ID
            conversation_phase: 对话阶段（默认 "idle"）
            cleanup_confirmation: 是否清理确认数据（默认 True）
            
        Returns:
            更新后的状态片段
        """
        if cleanup_confirmation:
            # 清理确认数据
            await self.confirmation_manager.cancel_pending(session_id)
        
        response_model = TextResponse(content=content)
        return {
            "messages": [AIMessage(content=content)],  # 只返回新增消息
            "current_agent": self.name,
            "confirmation_pending": None,
            "conversation_phase": conversation_phase,
            **response_model.to_full_response()
        }

    def _normalize_items_to_json(self, items: Any) -> str:
        """将 items 参数标准化为 JSON 字符串
        
        处理 items 参数的不同格式：
        - 如果已经是 JSON 字符串，直接返回
        - 如果是列表，转换为 JSON 字符串
        - 其他类型，尝试转换为列表再编码
        
        Args:
            items: items 参数（可能是字符串、列表或其他类型）
            
        Returns:
            JSON 字符串格式的 items
        """
        if isinstance(items, str):
            # 如果已经是 JSON 字符串，直接使用
            return items
        elif isinstance(items, list):
            # 如果是列表，转换为 JSON 字符串
            return json.dumps(items, ensure_ascii=False)
        else:
            # 其他类型，尝试转换为列表再编码
            return json.dumps(list(items) if items else [], ensure_ascii=False)

    async def _execute_confirmation_action(
        self,
        state: MultiAgentState,
        action_type: str,
        action_data: Dict[str, Any],
        tool_name: str,
        session_id: str,
        success_phase: str = "idle"
    ) -> Dict[str, Any]:
        """执行确认操作的通用方法
        
        Args:
            state: 当前多Agent状态
            action_type: 操作类型（create_order 或 cancel_order）
            action_data: 操作数据
            tool_name: 工具名称（confirm_create_order 或 confirm_cancel_order）
            session_id: 用户会话ID
            success_phase: 成功后的对话阶段（create_order 使用 "order_completed"，其他使用 "idle"）
            
        Returns:
            更新后的状态片段
        """
        messages = state.messages
        
        # 获取工具
        confirm_tool = self._get_tool(tool_name)
        if not confirm_tool:
            raise ValueError(f"未找到 {tool_name} 工具")
        
        # 准备工具参数
        # 【重要】始终使用真实的 session_id，覆盖 action_data 中可能存在的错误值
        # 这样可以确保即使用户在准备阶段错误地传递了 "seesion_id" 等字符串，也能被纠正
        if action_type == "create_order":
            # 处理 items 参数
            items_json = self._normalize_items_to_json(action_data.get("items", []))
            tool_args = {
                "user_id": session_id,  # 强制使用真实的 session_id，忽略 action_data 中的值
                "items": items_json,
            }
            if "notes" in action_data:
                tool_args["notes"] = action_data.get("notes")
            logger.info(f"✅ [ORDER_AGENT] 确认创建订单，使用 session_id: {session_id}")
        elif action_type == "cancel_order":
            tool_args = {
                "order_id": action_data.get("order_id"),
                "user_id": session_id,  # 强制使用真实的 session_id，忽略 action_data 中的值
            }
            logger.info(f"✅ [ORDER_AGENT] 确认取消订单，使用 session_id: {session_id}")
        else:
            raise ValueError(f"未知的操作类型: {action_type}")
        
        # 执行工具
        tool_result = await confirm_tool.ainvoke(tool_args)
        result_data = self._parse_tool_result(tool_result)
        
        execution_success = result_data.get("success", False)
        default_message = "订单创建完成" if action_type == "create_order" else "订单取消完成"
        message = result_data.get("text", default_message)
        
        # 清理确认数据
        await self.confirmation_manager.cancel_pending(session_id)
        
        # 构建返回结果
        if execution_success:
            # 执行成功
            response_model = TextResponse(content=message)
            return {
                "messages": [AIMessage(content=message)],  # 只返回新增消息
                "current_agent": self.name,
                "confirmation_pending": None,
                "conversation_phase": success_phase,
                "tools_used": state.tools_used + [{
                    "agent": self.name,
                    "tool": tool_name,
                    "args": tool_args
                }],
                **response_model.to_full_response()
            }
        else:
            # 执行失败
            if action_type == "create_order":
                error_message = f"{message}\n\n订单创建出错了，需要重新下单吗？"
            else:
                error_message = message
            response_model = TextResponse(content=error_message)
            return {
                "messages": [AIMessage(content=error_message)],  # 只返回新增消息
                "current_agent": self.name,
                "confirmation_pending": None,
                "conversation_phase": "idle",
                "tools_used": state.tools_used + [{
                    "agent": self.name,
                    "tool": tool_name,
                    "args": tool_args
                }],
                **response_model.to_full_response()
            }

    def _get_tool(self, tool_name: str):
        """获取指定名称的工具

        Args:
            tool_name: 工具名称

        Returns:
            工具实例，如果未找到返回 None
        """
        return next((t for t in self.tools if t.name == tool_name), None)

    def _detect_order_intent(self, state: MultiAgentState) -> Dict[str, bool]:
        """基于意图识别结果检测订单相关意图

        完全依赖意图识别节点的结果，从 state.query_intent 和 state.entities 中提取订单意图信息。
        不再使用关键词匹配，所有意图判断都在意图识别阶段完成。

        Args:
            state: 多Agent状态（包含意图识别结果）

        Returns:
            包含意图检测结果的字典：{"is_query": bool, "is_cancel": bool}
        """
        query_intent = state.query_intent or {}
        entities = state.entities or {}
        
        # 从 entities 中获取 order_id
        order_id = entities.get("order_id")
        
        # 从 query_intent 中获取 reasoning（包含业务意图标识）
        reasoning = query_intent.get("reasoning", "").lower() if query_intent else ""
        
        # 基于 reasoning 中的业务意图标识判断（意图识别节点已在 prompt 中要求明确标识）
        is_query = "订单查询意图" in reasoning
        is_cancel = "订单取消意图" in reasoning
        
        # 如果 reasoning 中没有明确标识但有 order_id，则基于实体和意图类型进行推断
        # 如果提取到 order_id，通常默认是查询意图（除非明确标识为取消意图）
        if order_id and not is_query and not is_cancel:
            # 有订单号但意图不明确，默认视为查询意图
            is_query = True
        
        return {
            "is_query": is_query,
            "is_cancel": is_cancel,
        }

    async def _handle_query_intent(
        self, state: MultiAgentState, messages: list, content: str, session_id: str
    ) -> Dict[str, Any] | None:
        """处理查询订单意图

        使用统一的 query_order 工具，业务逻辑已收敛到 tool 中。
        - 如果提供了 order_id：tool 会优先查询特定订单并验证权限
        - 如果没有提供 order_id：tool 会查询用户所有订单

        Args:
            state: 多Agent状态
            messages: 消息列表
            content: 用户输入内容
            session_id: 会话ID（作为用户标识）

        Returns:
            如果成功处理返回结果字典，否则返回 None
        """
        logger.info(f"检测到查询订单意图: {content[:50]}...，使用session_id: {session_id}")

        # 从意图识别结果中提取 order_id
        entities = state.entities or {}
        order_id = entities.get("order_id")

        query_tool = self._get_tool("query_order")
        if not query_tool:
            logger.warning("未找到 query_order 工具")
            return None

        try:
            # 调用统一的查询工具，业务逻辑在 tool 中处理
            query_result = await query_tool.ainvoke({
                "user_id": session_id,
                "order_id": order_id,  # 可选，如果提供则查询特定订单，否则查询所有订单
                "status": None,
                "limit": 20
            })

            result_data = self._parse_tool_result(query_result)
            orders = result_data.get("orders", [])
            order_text = result_data.get("text", "")

            # 构建消息序列
            tool_call_id = f"call_query_{session_id}_{hash(content) % 100000}"
            ai_message_with_tool = AIMessage(
                content="",
                tool_calls=[{
                    "id": tool_call_id,
                    "name": "query_order",
                    "args": {"user_id": session_id, "order_id": order_id, "status": None, "limit": 20}
                }]
            )

            tool_message = ToolMessage(content=query_result, tool_call_id=tool_call_id)
            final_ai_message = AIMessage(content=order_text)

            logger.info(f"查询订单完成: 找到{len(orders)}个订单，order_id={order_id}")

            # 使用OrderListResponse构建完整响应（包含AI消息content）
            response_model = OrderListResponse(
                orders=orders,
                total=len(orders),
                content=order_text
            )
            return {
                "messages": [ai_message_with_tool] + [tool_message] + [final_ai_message],
                "current_agent": self.name,
                "tools_used": state.tools_used + [{
                    "agent": self.name,
                    "tool": "query_order",
                    "args": {"user_id": session_id, "order_id": order_id}
                }],
                **response_model.to_full_response()
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
        logger.info(f"检测到取消订单意图: {content[:50]}...，使用session_id: {session_id}")

        # 直接从 entities 中获取 order_id（订单号，字符串）
        entities = state.entities or {}
        order_id = entities.get("order_id")
        
        if not order_id:
            logger.info(f"取消意图但缺少订单ID，使用 LLM 处理")
            return None
        
        # 确保 order_id 是字符串类型
        if not isinstance(order_id, str):
            order_id = str(order_id)

        logger.info(f"调用 prepare_cancel_order: order_id={order_id}, session_id={session_id}")

        prepare_tool = self._get_tool("prepare_cancel_order")

        if not prepare_tool:
            logger.warning("未找到 prepare_cancel_order 工具")
            return None

        try:
            prepare_result = await prepare_tool.ainvoke({
                "order_id": order_id,  # 直接使用订单号（字符串）
                "user_id": session_id,  # 使用 session_id 作为用户标识
                "reason": "用户请求取消"
            })

            result_data = self._parse_tool_result(prepare_result)

            if not result_data.get("can_cancel", False):
                response_model = TextResponse(content=result_data.get("text", "无法取消订单"))
                return {
                    "messages": [AIMessage(content=response_model.content)],  # 只返回新增消息
                    "current_agent": self.name,
                    **response_model.to_full_response()
                }

            display_message = result_data.get("text", "请确认是否取消订单")
            # 使用工具返回的 order_id（订单号）
            display_order_id = result_data.get("order_id", order_id)
            display_data = {
                "order_id": display_order_id
            }

            confirmation = await self.confirmation_manager.request_confirmation(
                session_id=session_id,
                action_type="cancel_order",
                action_data={"order_id": order_id, "user_id": session_id},  # 使用订单号（字符串）
                agent_name=self.name,
                display_message=display_message,
                display_data=display_data
            )

            logger.info(f"创建取消订单确认: confirmation_id={confirmation.confirmation_id}")

            # 构建基础结果字典
            result = {
                "messages": [AIMessage(content=display_message)],  # 只返回新增消息
                "current_agent": self.name,
                "tools_used": state.tools_used + [{
                    "agent": self.name,
                    "tool": "prepare_cancel_order",
                    "args": {"order_id": order_id, "user_id": session_id}
                }],
            }

            # 设置 conversation_phase 为 order_creating，表示正在等待用户确认
            result["conversation_phase"] = "order_creating"

            result["confirmation_pending"] = {
                "confirmation_id": confirmation.confirmation_id,
                "action_type": confirmation.action_type,
                "display_message": confirmation.display_message,
                "display_data": confirmation.display_data,
            }

            # 使用ConfirmationResponse覆盖默认的text类型（包含AI消息content）
            confirmation_model = ConfirmationResponse(
                confirmation_id=confirmation.confirmation_id,
                action_type=confirmation.action_type,
                display_message=confirmation.display_message,
                display_data=confirmation.display_data,
                content=display_message  # AI消息内容
            )
            result.update(confirmation_model.to_full_response())  # 更新所有前端字段

            # 【核心修复】使用 interrupt() 中断图执行，等待用户确认
            # 创建标准化的中断数据
            interrupt_data = create_confirmation_interrupt(
                action_type="cancel_order",
                action_data={"order_id": order_id, "user_id": session_id},  # 使用订单号（字符串）
                display_message=display_message,
                display_data=display_data,
                confirmation_id=confirmation.confirmation_id
            )
            
            # 【关键修复】将 result 中的所有前端展示信息添加到中断数据中
            # 因为 interrupt() 抛出异常后，result 中的信息不会自动传递
            # 需要将这些信息包含在 interrupt_data 中，以便前端正确显示
            interrupt_data["confirmation_pending"] = result["confirmation_pending"]
            interrupt_data["conversation_phase"] = result.get("conversation_phase", "order_creating")
            interrupt_data["content"] = result.get("content", display_message)
            interrupt_data["response_type"] = result.get("response_type", "confirmation")
            interrupt_data["role"] = result.get("role", "assistant")
            # 如果有 response_data，也包含进去
            if "response_data" in result:
                interrupt_data["response_data"] = result["response_data"]
            
            # 调用 interrupt() 中断执行
            # 第一次调用（中断时）：会抛出 GraphInterrupt 异常
            # 恢复执行时：会返回 resume_data 值
            try:
                resume_value = interrupt(interrupt_data)
                # 如果 interrupt() 返回了值，说明这是恢复执行
                # 恢复执行时，直接处理 resume 值并执行确认操作
                resume_result = await self._handle_resume_execution(state, resume_value, session_id)
                # 返回恢复执行的结果，不再继续执行后续代码
                return resume_result
            except GraphInterrupt as e:
                # 第一次调用（中断时）：抛出 GraphInterrupt 异常
                raise
            except Exception as e:
                # 记录其他异常
                logger.error(f"interrupt() 调用失败: {e}", exc_info=True)
                raise
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

        # 【场景处理】当有 order_id 时（查询订单详情）
        if all_entities.get("order_id"):
            hints.append("\n=== 当前场景：查询订单详情 ===")
            hints.append("检测到用户提供了订单ID（订单号），应该调用 query_order 工具查询订单详情。")
            hints.append(f"- 订单ID: {all_entities.get('order_id')}（使用 order_id 参数，字符串类型）")
            hints.append("注意：query_order 工具接受 user_id（必填）和 order_id（可选）参数。如果提供 order_id，会优先查询特定订单并验证权限；如果不提供 order_id，会查询用户所有订单。")

        # 【场景处理】当有 product_id 时（用户已登录，无需手机号）
        if all_entities.get("product_id"):
            hints.append("\n=== 当前场景：可以创建订单 ===")
            hints.append("检测到用户已选定产品（product_id存在）。用户已登录，系统会自动识别用户身份。")
            hints.append("你应该立即调用 prepare_create_order 工具来创建订单，不需要再询问用户。")
            hints.append(f"- 产品ID: {all_entities.get('product_id')}")
            hints.append(f"- 数量: {all_entities.get('quantity', 1)}")

        hints.append("\n请根据对话历史、上下文信息和工具描述，判断是否可以执行操作，或需要向用户询问什么信息。")

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
                "messages": [response],  # 只返回新增消息
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

            # 自动注入正确的 session_id，修复 LLM 可能错误传递的 user_id 参数
            # 所有需要 user_id 的订单工具都应该使用真实的 session_id
            tool_args = dict(tool_call["args"])
            tools_requiring_user_id = [
                "prepare_create_order",
                "confirm_create_order",
                "query_order",
                "prepare_cancel_order",
                "confirm_cancel_order",
            ]
            
            if tool_call["name"] in tools_requiring_user_id:
                # 始终使用真实的 session_id，覆盖 LLM 可能错误传递的值（如字符串 "seesion_id"）
                tool_args["user_id"] = session_id
                logger.info(f"✅ [ORDER_AGENT] 自动注入 session_id 到工具 {tool_call['name']}: {session_id}")

            try:
                # 【关键日志】记录工具调用参数，特别是 items 参数
                if tool_call["name"] in ["prepare_create_order", "confirm_create_order"]:
                    logger.info(f"🔧 [ORDER_AGENT] 调用工具 {tool_call['name']}，参数: {tool_args}")
                    if "items" in tool_args:
                        try:
                            import json
                            items_data = json.loads(tool_args["items"]) if isinstance(tool_args["items"], str) else tool_args["items"]
                            logger.info(f"🔧 [ORDER_AGENT] items 内容: {items_data}")
                            for idx, item in enumerate(items_data):
                                logger.info(f"🔧 [ORDER_AGENT] 订单项 {idx+1}: product_id={item.get('product_id')}, quantity={item.get('quantity')}")
                        except Exception as e:
                            logger.warning(f"🔧 [ORDER_AGENT] 解析 items 失败: {e}")
                
                tool_result = await tool.ainvoke(tool_args)

                # 检查是否需要确认
                if tool_call["name"] in ["prepare_cancel_order", "prepare_create_order"]:
                    needs_confirmation = True
                    parsed_result = self._parse_tool_result(tool_result)
                    confirmation_data = {
                        "action_type": tool_call["name"].replace("prepare_", ""),
                        "action_data": tool_args,  # 使用修正后的参数（包含正确的 user_id）
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
                            "user_id": tool_args.get("user_id", session_id),  # 使用修正后的参数中的 user_id
                            "items": tool_args.get("items"),
                            "items_data": parsed_result.get("items"),
                            "total_amount": parsed_result.get("total_amount"),
                            "text": parsed_result.get("text", "订单信息已准备"),
                            "can_create": parsed_result.get("can_create", True)
                        }
                        # 保存到字典中，供任务链使用
                        order_info_dict["order_info"] = order_info
                        # 【关键日志】记录准备创建订单时的 items
                        logger.info(f"🔧 [ORDER_AGENT] prepare_create_order 准备的数据: items={order_info.get('items')}, items_data={order_info.get('items_data')}")

                tool_messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
                tool_used_info.append({
                    "agent": self.name,
                    "tool": tool_call["name"],
                    "args": tool_args,  # 记录修正后的参数
                })
            except Exception as e:
                tool_messages.append(
                    ToolMessage(content=f"错误: {str(e)}", tool_call_id=tool_call["id"])
                )

        followup_messages = agent_messages + [response] + tool_messages
        # 使用异步LLM调用提高性能
        final_response = await self.llm.ainvoke(followup_messages)

        # 默认使用TextResponse
        response_model = TextResponse(content=final_response.content)
        result = {
            "messages": [response] + tool_messages + [final_response],  # 只返回新增消息
            "current_agent": self.name,
            "tools_used": state.tools_used + tool_used_info,
            **response_model.to_full_response()
        }

        # 如果提取了 order_info，添加到结果中（供任务链使用）
        if order_info_dict:
            result.update(order_info_dict)

        if needs_confirmation and confirmation_data:
            # 设置 conversation_phase 为 order_creating，表示正在等待用户确认
            result["conversation_phase"] = "order_creating"
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

            # 使用ConfirmationResponse覆盖默认的text类型（包含AI消息content）
            confirmation_model = ConfirmationResponse(
                confirmation_id=confirmation.confirmation_id,
                action_type=confirmation.action_type,
                display_message=confirmation.display_message,
                display_data=confirmation.display_data,
                content=final_response.content  # AI消息内容
            )
            result.update(confirmation_model.to_full_response())  # 更新所有前端字段

            # 【核心修复】使用 interrupt() 中断图执行，等待用户确认
            # 创建标准化的中断数据
            interrupt_data = create_confirmation_interrupt(
                action_type=confirmation_data["action_type"],
                action_data=confirmation_data["action_data"],
                display_message=confirmation_data["display_message"],
                display_data=confirmation_data["display_data"],
                confirmation_id=confirmation.confirmation_id
            )
            
            # 【关键修复】将 result 中的所有前端展示信息添加到中断数据中
            # 因为 interrupt() 抛出异常后，result 中的信息不会自动传递
            # 需要将这些信息包含在 interrupt_data 中，以便前端正确显示
            interrupt_data["confirmation_pending"] = result["confirmation_pending"]
            interrupt_data["conversation_phase"] = result.get("conversation_phase", "order_creating")
            interrupt_data["content"] = result.get("content", final_response.content)
            interrupt_data["response_type"] = result.get("response_type", "confirmation")
            interrupt_data["role"] = result.get("role", "assistant")
            # 如果有 response_data，也包含进去
            if "response_data" in result:
                interrupt_data["response_data"] = result["response_data"]
            
            # 调用 interrupt() 中断执行
            # 第一次调用（中断时）：会抛出 GraphInterrupt 异常
            # 恢复执行时：会返回 resume_data 值
            try:
                resume_value = interrupt(interrupt_data)
                # 如果 interrupt() 返回了值，说明这是恢复执行
                # 恢复执行时，直接处理 resume 值并执行确认操作
                resume_result = await self._handle_resume_execution(state, resume_value, session_id)
                # 返回恢复执行的结果，不再继续执行后续代码
                return resume_result
            except GraphInterrupt as e:
                # 第一次调用（中断时）：抛出 GraphInterrupt 异常
                raise
            except Exception as e:
                # 记录其他异常
                logger.error(f"interrupt() 调用失败: {e}", exc_info=True)
                raise

        return result

    async def _handle_resume_execution(
        self,
        state: MultiAgentState,
        resume_value: Any,
        session_id: str
    ) -> Dict[str, Any]:
        """处理恢复执行后的逻辑
        
        当用户确认/取消后，图恢复执行，interrupt() 返回 resume_data 值。
        根据 resume_data 中的 confirmed 状态执行相应操作。
        
        Args:
            state: 当前多Agent状态
            resume_value: interrupt() 返回的 resume 值
            session_id: 用户会话ID
            
        Returns:
            更新后的状态片段
        """
        messages = state.messages
        
        # 解析 resume 值
        confirmed = is_resume_confirm(resume_value)
        
        if confirmed is None:
            # resume 值无效，取消操作
            return await self._build_error_response(
                messages=messages,
                content="操作已取消，请重新开始",
                session_id=session_id,
                conversation_phase="idle"
            )
        
        # 获取待确认操作信息（从 resume_value 或 state 中）
        resume_dict = resume_value if isinstance(resume_value, dict) else {}
        confirmation_id = resume_dict.get("confirmation_id")
        
        # 获取确认信息
        confirmation = None
        if confirmation_id:
            confirmation = await self.confirmation_manager.get_confirmation(confirmation_id)
        
        if not confirmation:
            # 确认信息不存在，可能已过期或被清理
            return await self._build_error_response(
                messages=messages,
                content="确认信息已过期，请重新开始",
                session_id=session_id,
                conversation_phase="idle"
            )
        
        action_type = confirmation.action_type
        action_data = confirmation.action_data
        
        if not confirmed:
            # 用户取消操作
            return await self._build_error_response(
                messages=messages,
                content="👌 已取消操作，请问还有其他需要帮助的吗？",
                session_id=session_id,
                conversation_phase="idle"
            )
        
        # 用户确认，执行操作
        try:
            if action_type == "create_order":
                # 执行创建订单
                return await self._execute_confirmation_action(
                    state=state,
                    action_type=action_type,
                    action_data=action_data,
                    tool_name="confirm_create_order",
                    session_id=session_id,
                    success_phase="order_completed"
                )
            elif action_type == "cancel_order":
                # 执行取消订单
                return await self._execute_confirmation_action(
                    state=state,
                    action_type=action_type,
                    action_data=action_data,
                    tool_name="confirm_cancel_order",
                    session_id=session_id,
                    success_phase="idle"
                )
            else:
                # 未知的操作类型
                return await self._build_error_response(
                    messages=messages,
                    content="未知的操作类型，请重新开始",
                    session_id=session_id,
                    conversation_phase="idle"
                )
                
        except Exception as e:
            # 执行失败
            logger.error(f"执行操作失败: action_type={action_type}, error={e}", exc_info=True)
            error_message = f"操作执行失败: {str(e)}"
            return await self._build_error_response(
                messages=messages,
                content=error_message,
                session_id=session_id,
                conversation_phase="idle"
            )

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
            response_model = TextResponse(
                content="您好！我是订单管理助手。查询订单需要提供手机号，请问有什么可以帮您？"
            )
            return {
                "messages": [AIMessage(content=response_model.content)],
                "current_agent": self.name,
                **response_model.to_full_response()
            }

        # 获取最新消息
        latest_message = messages[-1]

        # 【已废弃】旧的确认机制（通过 ConfirmationManager 和文本解析）
        # 现在使用 interrupt/resume 机制，恢复执行在 _handle_llm_tool_calls 中处理
        # 这里只保留作为后备（用于文本确认）
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
                            # 执行成功：清理 confirmation_pending，设置 conversation_phase 为 order_completed
                            response_model = TextResponse(content=message)
                            return {
                                "messages": [AIMessage(content=response_model.content)],  # 只返回新增消息
                                "current_agent": self.name,
                                **response_model.to_full_response(),
                                "confirmation_pending": None,
                                "conversation_phase": "order_completed" if result.action_type == "create_order" else "idle",
                            }
                        else:
                            # 执行失败：保留 confirmation_pending，让 AI 继续处理错误
                            # 添加错误提示消息，引导用户重新下单
                            error_message = f"{message}\n\n订单创建出错了，需要重新下单吗？"
                            logger.warning(f"订单创建失败，保留 confirmation_pending 以便 AI 处理错误: session={session_id}")
                            response_model = TextResponse(content=error_message)
                            return {
                                "messages": [AIMessage(content=response_model.content)],  # 只返回新增消息
                                "current_agent": self.name,
                                **response_model.to_full_response()
                                # 不设置 confirmation_pending，保留原有的值
                            }
                    elif result.status == ConfirmationStatus.CANCELLED:
                        # 用户取消
                        response_model = TextResponse(content="👌 已取消操作，请问还有其他需要帮助的吗？")
                        return {
                            "messages": [AIMessage(content=response_model.content)],  # 只返回新增消息
                            "current_agent": self.name,
                            **response_model.to_full_response(),
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

        # 基于意图识别结果检测订单相关意图（完全依赖意图识别节点的结果）
        intent = self._detect_order_intent(state)

        # 处理查询订单意图
        if intent["is_query"]:
            result = await self._handle_query_intent(state, messages, latest_content, session_id)
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
