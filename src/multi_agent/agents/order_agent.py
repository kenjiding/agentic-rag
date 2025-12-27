"""Order Agent - 订单管理 Agent

处理与订单相关的查询和操作：
- 订单查询（列表、详情）
- 订单取消（需要用户确认）
- 订单创建（需要用户确认）
"""

import json
import re
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage

from src.tools.order_tools import get_order_tools
from src.multi_agent.state import MultiAgentState
from src.confirmation import get_confirmation_manager, ConfirmationManager, ConfirmationStatus


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


# 确认相关的关键词
CONFIRM_YES = ["确认", "是", "好的", "可以", "同意", "下单", "执行", "继续"]
CONFIRM_NO = ["不", "否", "取消", "不要", "算了"]


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

        Args:
            user_input: 用户输入文本

        Returns:
            True: 确认
            False: 否认
            None: 无法判断（非确认相关输入）
        """
        user_input_lower = user_input.strip().lower()

        # 检查确认
        for keyword in CONFIRM_YES:
            if keyword in user_input_lower:
                return True

        # 检查否认
        for keyword in CONFIRM_NO:
            if keyword in user_input_lower:
                return False

        return None

    def _extract_user_phone(self, messages: list) -> str | None:
        """从消息历史中提取用户手机号

        Args:
            messages: 消息列表

        Returns:
            手机号或 None
        """
        # 手机号正则
        phone_pattern = r"1[3-9]\d{9}"

        # 从最新消息开始查找
        for msg in reversed(messages):
            if hasattr(msg, "content"):
                content = msg.content
                if isinstance(content, str):
                    phones = re.findall(phone_pattern, content)
                    if phones:
                        return phones[0]

        # 从 state metadata 中查找
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

        # === 优先从 context_data 获取上下文信息（任务链模式）===
        context_data = state.get("context_data", {})

        # 优先从context_data获取手机号（任务链传递的），其次从messages中提取
        user_phone = context_data.get("user_phone") or self._extract_user_phone(messages)

        selected_product_id = context_data.get("selected_product_id")
        selected_quantity = context_data.get("quantity", 1)

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

        # 构建 Agent 消息
        agent_messages = [
            SystemMessage(content=ORDER_AGENT_SYSTEM_PROMPT + phone_hint + product_hint)
        ]
        agent_messages.extend(messages)

        # 调用 LLM
        response = self.llm_with_tools.invoke(agent_messages)

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

            # 构建返回
            result = {
                "messages": messages + [final_response],
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

    def _execute_confirm_action(self, action_type: str, action_data: dict) -> str:
        """执行确认后的操作

        Args:
            action_type: 操作类型 (cancel_order, create_order)
            action_data: 操作参数

        Returns:
            操作结果
        """
        # 查找对应的 confirm_* 工具
        tool_name = f"confirm_{action_type}"
        tool = next((t for t in self.tools if t.name == tool_name), None)

        if not tool:
            return f"❌ 找不到确认操作工具: {tool_name}"

        try:
            result = tool.invoke(action_data)
            return result
        except Exception as e:
            return f"❌ 执行操作时出错: {str(e)}"


# 兼容 LangGraph 节点函数
async def order_agent_node(state: MultiAgentState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """LangGraph 节点函数 - 订单 Agent (异步)

    Args:
        state: 当前状态
        config: 配置（可包含 llm 实例和 session_id）

    Returns:
        状态更新
    """
    llm = config.get("llm") if config else None
    session_id = config.get("session_id", "default") if config else "default"
    agent = OrderAgent(llm=llm)
    return await agent.invoke(state, session_id=session_id)
