"""Order Agent - 订单管理 Agent

处理与订单相关的查询和操作：
- 订单查询（列表、详情）
- 订单取消（需要用户确认）
- 订单创建（需要用户确认）
"""

import json
import re
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage

from src.tools.order_tools import get_order_tools
from src.multi_agent.state import MultiAgentState


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
    1. prepare_* 操作后，进入"等待确认"状态
    2. 用户回复后，判断是否确认
    3. 确认后执行 confirm_* 操作
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        tools: list | None = None,
    ):
        """初始化 Order Agent

        Args:
            llm: LangChain LLM 实例
            tools: 订单工具列表，默认使用内置工具
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )
        self.tools = tools or get_order_tools()
        self.name = "order_agent"

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

    def invoke(self, state: MultiAgentState) -> Dict[str, Any]:
        """执行订单操作

        Args:
            state: 当前多 Agent 状态

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

        # 检查是否在等待确认
        awaiting_confirmation = state.get("confirmation_pending")
        if awaiting_confirmation:
            # 检查用户确认
            if hasattr(latest_message, "content"):
                user_input = latest_message.content
                confirmation = self._check_confirmation(user_input)

                if confirmation is True:
                    # 用户确认，执行确认操作
                    action_type = awaiting_confirmation.get("action_type")
                    action_data = awaiting_confirmation.get("data", {})

                    result = self._execute_confirm_action(action_type, action_data)

                    return {
                        "messages": messages + [AIMessage(content=result)],
                        "current_agent": self.name,
                        "confirmation_pending": None,
                    }
                elif confirmation is False:
                    # 用户取消
                    return {
                        "messages": messages + [
                            AIMessage(content="👌 已取消操作，请问还有其他需要帮助的吗？")
                        ],
                        "current_agent": self.name,
                        "confirmation_pending": None,
                    }
                # 无法判断，继续正常处理

        # 构建提取用户手机号的提示
        user_phone = self._extract_user_phone(messages)
        phone_hint = f"\n用户手机号: {user_phone}" if user_phone else "\n注意: 需要用户提供手机号才能查询订单"

        # 构建 Agent 消息
        agent_messages = [
            SystemMessage(content=ORDER_AGENT_SYSTEM_PROMPT + phone_hint)
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
                        result = tool.invoke(tool_call["args"])

                        # 检查是否需要确认
                        if tool_call["name"] in ["prepare_cancel_order", "prepare_create_order"]:
                            needs_confirmation = True
                            confirmation_data = {
                                "action_type": tool_call["name"].replace("prepare_", ""),
                                "data": tool_call["args"],
                            }

                        # 构建 ToolMessage
                        tool_messages.append(
                            ToolMessage(
                                content=str(result),
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

            # 如果需要确认，设置确认状态
            if needs_confirmation and confirmation_data:
                result["confirmation_pending"] = confirmation_data

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
def order_agent_node(state: MultiAgentState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """LangGraph 节点函数 - 订单 Agent

    Args:
        state: 当前状态
        config: 配置（可包含 llm 实例）

    Returns:
        状态更新
    """
    llm = config.get("llm") if config else None
    agent = OrderAgent(llm=llm)
    return agent.invoke(state)
