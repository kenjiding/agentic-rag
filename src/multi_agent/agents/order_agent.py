"""Order Agent - 订单管理 Agent

处理与订单相关的查询和操作：
- 订单查询（列表、详情）
- 订单取消（需要用户确认）
- 订单创建（需要用户确认）
"""

import json
import logging
from typing import Any, Dict

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

回复风格：
- 使用友好的语气，用 emoji 让回复更生动
- 涉及金额时精确到小数点后两位
- 操作完成后提供清晰的反馈
"""


# 注意：确认关键词已移至 src/multi_agent/config.py 中的 KeywordsConfig
# 使用 get_keywords_config() 获取配置化的关键词列表


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

    async def invoke(self, state: MultiAgentState, session_id: str = "default") -> Dict[str, Any]:
        """执行订单操作

        Args:
            state: 当前多 Agent 状态
            session_id: 用户会话 ID，用于确认机制

        Returns:
            更新后的状态片段
        """
        messages = state.get("messages", [])
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
                        message = exec_result.get("text", "操作已完成")
                        if result.error:
                            message = f"操作执行失败: {result.error}"
                        return {
                            "messages": messages + [AIMessage(content=message)],
                            "current_agent": self.name,
                            "confirmation_pending": None,
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
                # result 为 None 表示用户输入不是确认响应，继续正常处理

        # === 从 state["entities"] 获取实体信息（2025最佳实践）===
        entities = state.get("entities", {})
        context_data = state.get("context_data", {})

        # === 【关键修复】检测取消订单意图，强制使用 prepare_cancel_order ===
        latest_content = latest_message.content if hasattr(latest_message, "content") else ""
        is_cancel_intent = any(kw in latest_content for kw in CANCEL_ORDER_KEYWORDS)

        if is_cancel_intent:
            logger.info(f"🔍 [ORDER_AGENT] 检测到取消订单意图: {latest_content[:50]}...")

            # 尝试从上下文中获取订单信息
            order_id = entities.get("order_id") or context_data.get("order_id")
            user_phone = entities.get("user_phone") or context_data.get("user_phone")

            # 如果没有在 entities 中，尝试从之前的消息中查找订单信息
            if not order_id:
                # 从 agent_results 中查找订单信息
                order_result = state.get("agent_results", {}).get("order_agent", {})
                if isinstance(order_result, dict) and "orders" in order_result:
                    orders = order_result.get("orders", [])
                    if orders and len(orders) == 1:
                        # 如果只有一个订单，自动选择
                        order_id = orders[0].get("id")
                        logger.info(f"🔍 [ORDER_AGENT] 从 agent_results 获取到单一订单: id={order_id}")

                # 如果还没有，从消息历史中的 ToolMessage 查找
                if not order_id:
                    for msg in reversed(messages):
                        if isinstance(msg, ToolMessage):
                            try:
                                tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                                if isinstance(tool_result, dict) and "orders" in tool_result:
                                    orders = tool_result.get("orders", [])
                                    if orders and len(orders) == 1:
                                        order_id = orders[0].get("id")
                                        logger.info(f"🔍 [ORDER_AGENT] 从历史消息获取到单一订单: id={order_id}")
                                        break
                            except (json.JSONDecodeError, TypeError):
                                continue

            # 如果有订单 ID 和用户手机号，直接调用 prepare_cancel_order
            if order_id and user_phone:
                logger.info(f"🔍 [ORDER_AGENT] 强制调用 prepare_cancel_order: order_id={order_id}, phone={user_phone}")

                # 先获取完整的订单信息用于前端展示
                order_info = None
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage):
                        try:
                            tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                            if isinstance(tool_result, dict) and "orders" in tool_result:
                                orders = tool_result.get("orders", [])
                                for o in orders:
                                    if o.get("id") == order_id or o.get("id") == int(order_id):
                                        order_info = o
                                        break
                                if order_info:
                                    break
                        except (json.JSONDecodeError, TypeError):
                            continue

                prepare_tool = next((t for t in self.tools if t.name == "prepare_cancel_order"), None)
                if prepare_tool:
                    try:
                        prepare_result = prepare_tool.invoke({
                            "order_id": int(order_id),
                            "user_phone": user_phone,
                            "reason": "用户请求取消"
                        })

                        # 解析结果
                        result_data = json.loads(prepare_result) if isinstance(prepare_result, str) else prepare_result

                        if result_data.get("can_cancel", False):
                            # 创建 confirmation_pending
                            display_message = result_data.get("text", "请确认是否取消订单")

                            # 构建完整的展示数据（包含订单信息）
                            display_data = {
                                "order_id": order_id,
                                "order": order_info  # 包含订单详情，供前端渲染
                            }

                            confirmation = await self.confirmation_manager.request_confirmation(
                                session_id=session_id,
                                action_type="cancel_order",
                                action_data={
                                    "order_id": int(order_id),
                                    "user_phone": user_phone
                                },
                                agent_name=self.name,
                                display_message=display_message,
                                display_data=display_data
                            )

                            logger.info(f"✅ [ORDER_AGENT] 创建取消订单确认: confirmation_id={confirmation.confirmation_id}")

                            return {
                                "messages": messages + [AIMessage(content=display_message)],
                                "current_agent": self.name,
                                "confirmation_pending": {
                                    "confirmation_id": confirmation.confirmation_id,
                                    "action_type": "cancel_order",
                                    "display_message": display_message,
                                    "display_data": display_data
                                },
                                "tools_used": state.get("tools_used", []) + [{
                                    "agent": self.name,
                                    "tool": "prepare_cancel_order",
                                    "args": {"order_id": order_id, "user_phone": user_phone}
                                }]
                            }
                        else:
                            # 无法取消，返回原因
                            return {
                                "messages": messages + [AIMessage(content=result_data.get("text", "无法取消订单"))],
                                "current_agent": self.name,
                            }
                    except Exception as e:
                        logger.error(f"❌ [ORDER_AGENT] prepare_cancel_order 失败: {e}", exc_info=True)

            # 如果没有足够的信息，记录日志但继续使用 LLM 处理
            if not order_id or not user_phone:
                logger.info(f"🔍 [ORDER_AGENT] 取消意图但缺少信息: order_id={order_id}, user_phone={user_phone}，使用 LLM 处理")

        # 优先从 entities 读取，其次从 context_data 读取（向后兼容任务链）
        user_phone = entities.get("user_phone") or context_data.get("user_phone")
        selected_product_id = entities.get("selected_product_id") or context_data.get("selected_product_id")
        selected_quantity = entities.get("quantity") or context_data.get("quantity", 1)

        # === 任务链模式：强制调用 prepare_create_order（跳过 LLM 判断）===
        # 检查是否在任务链模式下且有完整信息
        task_chain = state.get("task_chain")
        is_task_chain_mode = False

        if task_chain:
            current_index = task_chain.get("current_step_index", 0)
            steps = task_chain.get("steps", [])
            if current_index < len(steps):
                current_step = steps[current_index]
                if current_step.get("step_type") == "order_creation":
                    is_task_chain_mode = True

        # 任务链模式 + 完整信息：强制调用 prepare_create_order
        if is_task_chain_mode and selected_product_id and user_phone:
            logger.info(
                f"任务链模式：强制调用 prepare_create_order，"
                f"product_id={selected_product_id}, quantity={selected_quantity}, phone={user_phone}"
            )

            # 直接调用 prepare_create_order
            prepare_tool = next((t for t in self.tools if t.name == "prepare_create_order"), None)
            if not prepare_tool:
                return {
                    "messages": messages + [
                        AIMessage(content="❌ 订单创建工具未找到，请联系管理员")
                    ],
                    "current_agent": self.name,
                }

            try:
                # 调用 prepare_create_order
                # 注意：prepare_create_order 期望 items 是 JSON 字符串，不是列表
                items_list = [{"product_id": int(selected_product_id), "quantity": int(selected_quantity)}]
                items_json = json.dumps(items_list, ensure_ascii=False)
                prepare_result = prepare_tool.invoke({
                    "user_phone": user_phone,
                    "items": items_json,
                    "notes": None
                })

                # 解析结果
                if isinstance(prepare_result, str):
                    try:
                        result_data = json.loads(prepare_result)
                    except:
                        result_data = {"text": prepare_result}
                else:
                    result_data = prepare_result

                # 构建友好消息
                result_message = result_data.get("text", "订单信息已确认")
                display_message = f"请确认订单信息：\n{result_message}"

                # 创建 confirmation_pending
                # action_data 中保存原始列表格式，供 confirm_create_order 使用
                confirmation = await self.confirmation_manager.request_confirmation(
                    session_id=session_id,
                    action_type="create_order",
                    action_data={
                        "user_phone": user_phone,
                        "items": items_json,  # 保存 JSON 字符串格式，与工具期望的格式一致
                        "notes": None
                    },
                    agent_name=self.name,
                    display_message=display_message,
                    display_data={
                        "items": result_data.get("items"),
                        "total_amount": result_data.get("total_amount"),
                    },
                )

                logger.info(f"任务链模式：已创建订单确认请求，confirmation_id={confirmation.confirmation_id}")

                # 返回确认信息
                return {
                    "messages": messages + [AIMessage(content=display_message)],
                    "current_agent": self.name,
                    "confirmation_pending": {
                        "confirmation_id": confirmation.confirmation_id,
                        "action_type": confirmation.action_type,
                        "display_message": confirmation.display_message,
                        "display_data": confirmation.display_data,
                    },
                    "tools_used": state.get("tools_used", []) + [{
                        "agent": self.name,
                        "tool": "prepare_create_order",
                        "args": {"user_phone": user_phone, "items": items_json}
                    }]
                }
            except Exception as e:
                logger.error(f"任务链模式准备订单失败: {e}", exc_info=True)
                return {
                    "messages": messages + [
                        AIMessage(content=f"❌ 准备订单失败: {str(e)}")
                    ],
                    "current_agent": self.name,
                }

        # === 正常模式：使用 LLM 处理 ===
        # 构建手机号提示
        phone_hint = f"\n用户手机号: {user_phone}" if user_phone else "\n注意: 需要用户提供手机号才能查询订单"

        # 如果有选中的商品（任务链模式），添加明确的上下文提示
        product_hint = ""
        if selected_product_id and user_phone:
            # 任务链模式：已有完整信息，直接创建订单
            product_hint = f"""

=== 任务链上下文（重要）===
用户已通过多步骤流程选择商品并提供了必要信息：
- 商品 ID: {selected_product_id}
- 购买数量: {selected_quantity}
- 用户手机号: {user_phone}

所有必要信息已齐全，请立即使用 prepare_create_order 工具创建订单。
必须使用的参数：
  user_phone: "{user_phone}"
  items: [{{"product_id": {selected_product_id}, "quantity": {selected_quantity}}}]

不要再询问用户提供手机号或其他信息，直接执行即可。
"""
        elif selected_product_id and not user_phone:
            # 有商品但缺少手机号
            product_hint = f"""

=== 任务链上下文 ===
用户已选择商品（ID: {selected_product_id}，数量: {selected_quantity}），但缺少手机号。
请向用户索要手机号以完成订单创建。
"""

        # 清理消息序列，移除孤立的 ToolMessage
        # 这确保符合 OpenAI API 的格式要求：
        # "messages with role 'tool' must be a response to a preceeding message with 'tool_calls'"
        cleaned_messages = clean_messages_for_llm(messages)

        # 构建 Agent 消息
        agent_messages = [
            SystemMessage(content=ORDER_AGENT_SYSTEM_PROMPT + phone_hint + product_hint)
        ]
        agent_messages.extend(cleaned_messages)

        # 添加调试日志
        logger.info(f"🤖 [ORDER_AGENT] 准备调用 LLM")
        logger.info(f"🤖 [ORDER_AGENT] 提取的用户手机号: {user_phone}")
        logger.info(f"🤖 [ORDER_AGENT] 消息数量: {len(cleaned_messages)}")
        if cleaned_messages:
            latest_msg = cleaned_messages[-1]
            logger.info(f"🤖 [ORDER_AGENT] 最新消息类型: {type(latest_msg).__name__}")
            logger.info(f"🤖 [ORDER_AGENT] 最新消息内容: {latest_msg.content[:100] if hasattr(latest_msg, 'content') else 'N/A'}...")

        # 调用 LLM
        response = self.llm_with_tools.invoke(agent_messages)

        # 添加调试日志
        logger.info(f"🤖 [ORDER_AGENT] LLM 响应类型: {type(response).__name__}")
        logger.info(f"🤖 [ORDER_AGENT] 是否有工具调用: {hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"🤖 [ORDER_AGENT] 工具调用数量: {len(response.tool_calls)}")
            for tc in response.tool_calls:
                logger.info(f"  - 工具名称: {tc.get('name', 'N/A')}")
                logger.info(f"    参数: {tc.get('args', {})}")

        # 处理工具调用
        if hasattr(response, "tool_calls") and response.tool_calls:
            # 检查是否是 prepare_* 操作（需要确认）
            needs_confirmation = False
            confirmation_data = None

            # 执行工具调用并构建 ToolMessage
            tool_messages = []
            tool_used_info = []

            for tool_call in response.tool_calls:
                tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
                if tool:
                    try:
                        tool_result = tool.invoke(tool_call["args"])

                        # 检查是否需要确认
                        if tool_call["name"] in ["prepare_cancel_order", "prepare_create_order"]:
                            needs_confirmation = True

                            # 解析工具结果以获取展示信息
                            try:
                                parsed_result = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
                            except:
                                parsed_result = {}

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

                        # 构建 ToolMessage
                        tool_messages.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call["id"],
                            )
                        )
                        tool_used_info.append({
                            "agent": self.name,
                            "tool": tool_call["name"],
                            "args": tool_call["args"],
                        })
                    except Exception as e:
                        # 工具调用失败，也需要返回 ToolMessage
                        tool_messages.append(
                            ToolMessage(
                                content=f"错误: {str(e)}",
                                tool_call_id=tool_call["id"],
                            )
                        )

            # 构建后续消息列表（包含 tool_calls 的 assistant 消息 + ToolMessage）
            followup_messages = agent_messages + [response] + tool_messages

            # 再次调用 LLM 生成最终回复
            final_response = self.llm.invoke(followup_messages)

            # 构建返回 - 重要：必须包含完整的消息序列
            # 包括：1. response (包含 tool_calls 的 AIMessage)
            #      2. tool_messages (ToolMessage 列表)
            #      3. final_response (最终回复)
            # 这样可以确保 OpenAI API 的消息格式要求：
            # "messages with role 'tool' must be a response to a preceeding message with 'tool_calls'"
            result = {
                "messages": messages + [response] + tool_messages + [final_response],
                "current_agent": self.name,
                "tools_used": state.get("tools_used", []) + tool_used_info,
            }

            # 如果需要确认，通过 ConfirmationManager 创建确认请求
            if needs_confirmation and confirmation_data:
                confirmation = await self.confirmation_manager.request_confirmation(
                    session_id=session_id,
                    action_type=confirmation_data["action_type"],
                    action_data=confirmation_data["action_data"],
                    agent_name=self.name,
                    display_message=confirmation_data["display_message"],
                    display_data=confirmation_data["display_data"],
                )

                # 在返回中包含确认信息供前端使用
                result["confirmation_pending"] = {
                    "confirmation_id": confirmation.confirmation_id,
                    "action_type": confirmation.action_type,
                    "display_message": confirmation.display_message,
                    "display_data": confirmation.display_data,
                }

            return result

        # 无工具调用，直接返回响应
        return {
            "messages": messages + [response],
            "current_agent": self.name,
        }
